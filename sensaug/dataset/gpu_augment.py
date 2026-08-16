"""GPU-batched application of the differentiable ops (`diff32`).

The ops in `sensaug.dataset.differentiable_augmentations{,_aa}` are batched GPU
tensor functions -- that is their stated contract, and it is how the reference
implementation (Shu et al., ICRA 2021) applies them: to each batch, inside the
training step, not per-image in a data-loading pipeline.

This module is that call site. It hangs the ops off `SegDataPreProcessor`, which
already runs on GPU after collation and already receives a `training` flag, so
the train and eval paths can carry independent augmentation state through one
model instance.

Why not an mmseg pipeline transform: a pipeline transform runs per-image, on
CPU, in a dataloader worker, with a numpy<->torch round-trip per sample. That is
the path `_DiffAugTransform` took and it is what made the `grad_corr` SA
round-eval unaffordable. Nothing here converts to numpy or leaves the GPU.

The de-normalize -> op -> re-normalize arithmetic is deliberately identical to
`sensaug.hooks.grad_hook.CollectGradientHook._grad_for_op`. The probe and the
training pipeline must apply the SAME function to the same tensor, and the
easiest way for that to stop being true is for the two to reconstruct RGB [0, 1]
differently.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mmseg.registry import MODELS
from mmseg.models.data_preprocessor import SegDataPreProcessor

from sensaug.dataset.differentiable_augmentations_aa import (
    DIFF32_OPS,
    GEOMETRIC_OP_KEYS,
    geometric_affine_matrix,
    warp_image_and_label,
)

__all__ = [
    "GpuAugSegDataPreProcessor",
    "set_train_spec",
    "set_eval_spec",
    "clear_spec",
    "NO_OP",
]

# The sentinel `RandomAlphaTrainTransform` and `RandomTrainTransformNew` both
# use for "leave this image alone". Kept as the same string so the pdf dicts
# published by the SA loop need no translation on the way in -- ("none", 0) is
# a real key in them, and Lever 3 holds it fixed on purpose.
NO_OP = "none"


def _unwrap(model):
    """The underlying model, through DDP if present.

    Every rank runs its own preprocessor, so every rank has to be told about the
    augmentation state separately -- `runner.model` is the DDP wrapper and does
    not forward attribute writes to the module it wraps.
    """
    return model.module if hasattr(model, "module") else model


def _preprocessor(runner):
    dp = _unwrap(runner.model).data_preprocessor
    if not isinstance(dp, GpuAugSegDataPreProcessor):
        raise TypeError(
            "GPU augmentation was requested but the model's data_preprocessor is "
            f"{type(dp).__name__}, not GpuAugSegDataPreProcessor. train.py sets "
            "cfg.model.data_preprocessor.type when the resolved perturbation set "
            "is 'diff32'; this means that did not happen."
        )
    return dp


def set_train_spec(
    runner,
    pdf_dict: Optional[Dict] = None,
    geometric_only: bool = False,
    photometric_only: bool = False,
) -> None:
    """Install the training-time augmentation policy.

    `pdf_dict` given -> sample (op, level) pairs from it, matching
    RandomTrainTransformNew. Omitted -> uniform over the op bank plus "none",
    with a uniform level, matching RandomAlphaTrainTransform.
    """
    dp = _preprocessor(runner)
    if pdf_dict is not None:
        dp.set_train_pdf(pdf_dict)
    else:
        dp.set_train_uniform(geometric_only, photometric_only)


def set_eval_spec(runner, op_name: str, magnitude: float) -> None:
    """Install the eval-time perturbation: one op at one magnitude, applied to
    every image. This is the GPU counterpart of inserting a single perturbation
    transform into the val dataloader's pipeline."""
    _preprocessor(runner).set_eval(op_name, magnitude)


def clear_spec(runner, training: bool) -> None:
    """Remove the train or eval policy, restoring clean images on that path."""
    dp = _preprocessor(runner)
    if training:
        dp.set_train_none()
    else:
        dp.set_eval_none()


@MODELS.register_module()
class GpuAugSegDataPreProcessor(SegDataPreProcessor):
    """`SegDataPreProcessor` that applies a `diff32` op to the batch on GPU.

    Two independent slots, because the train and eval paths share one model
    instance: the SA round-eval sets a perturbation for evaluation while the
    training pipeline is separately sampling from its pdf, and neither may see
    the other's state. `forward`'s `training` flag selects between them.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (kind, payload): ("uniform", (geo, photo)) | ("pdf", pdf_dict) | None
        self._train_spec: Optional[Tuple[str, object]] = None
        # (op_name, magnitude) | None
        self._eval_spec: Optional[Tuple[str, float]] = None
        # Sampling RNG is numpy's global, matching the transforms this replaces
        # (RandomAlphaTrainTransform / RandomTrainTransformNew both call
        # np.random.*), so seeding behaviour is unchanged.

    # --- state ---------------------------------------------------------------

    def set_train_uniform(self, geometric_only=False, photometric_only=False):
        self._train_spec = ("uniform", (geometric_only, photometric_only))

    def set_train_pdf(self, pdf_dict: Dict):
        probs = np.asarray(list(pdf_dict.values()), dtype=np.float64)
        total = probs.sum()
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"pdf_dict probabilities sum to {total}, not 1.0"
            )
        self._train_spec = ("pdf", (list(pdf_dict.keys()), probs / total))

    def set_train_none(self):
        self._train_spec = None

    def set_eval(self, op_name: str, magnitude: float):
        if op_name != NO_OP and op_name not in DIFF32_OPS:
            raise KeyError(
                f"unknown op {op_name!r}; expected one of {sorted(DIFF32_OPS)}"
            )
        self._eval_spec = (op_name, float(magnitude))

    def set_eval_none(self):
        self._eval_spec = None

    # --- sampling ------------------------------------------------------------

    def _op_bank(self, geometric_only: bool, photometric_only: bool) -> List[str]:
        names = list(DIFF32_OPS)
        if geometric_only:
            return [n for n in names if n in GEOMETRIC_OP_KEYS]
        if photometric_only:
            return [n for n in names if n not in GEOMETRIC_OP_KEYS]
        return names

    def _sample(self, batch_size: int) -> List[Tuple[str, float]]:
        """One (op, magnitude) per image.

        Per-IMAGE, not per-batch. The transforms this replaces sampled inside
        `transform(results)`, i.e. once per sample, so collapsing to one draw per
        batch would quietly cut augmentation diversity by a factor of the batch
        size and make this arm non-comparable with the ones that did not change.
        """
        kind, payload = self._train_spec
        if kind == "uniform":
            geometric_only, photometric_only = payload
            bank = self._op_bank(geometric_only, photometric_only) + [NO_OP]
            return [
                (str(np.random.choice(bank)), float(np.random.uniform(0.0, 1.0)))
                for _ in range(batch_size)
            ]

        keys, probs = payload
        out = []
        for _ in range(batch_size):
            op, level = keys[np.random.choice(len(keys), p=probs)]
            if op != NO_OP:
                # Same jitter RandomTrainTransformNew applies, so the pdf's
                # discrete levels do not become the only magnitudes ever trained on.
                level = float(np.clip(level + np.random.normal(0, scale=0.1), 0.0, 1.0))
            out.append((op, float(level)))
        return out

    # --- application ---------------------------------------------------------

    @staticmethod
    def _gather_labels(data_samples: Sequence) -> Optional[torch.Tensor]:
        """(B, 1, H, W) label batch, or None when this batch carries no labels
        or they are not mutually stackable (differently-sized val images)."""
        if not data_samples:
            return None
        labels = []
        for sample in data_samples:
            gt = getattr(sample, "gt_sem_seg", None)
            if gt is None:
                return None
            labels.append(gt.data)
        if len({tuple(x.shape) for x in labels}) != 1:
            return None
        return torch.stack(labels, dim=0)

    @staticmethod
    def _scatter_labels(data_samples: Sequence, labels: torch.Tensor) -> None:
        for sample, label in zip(data_samples, labels):
            sample.gt_sem_seg.data = label

    def _apply(self, rgb01, labels, specs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply each sampled op to the sub-batch that drew it.

        Grouped rather than looped per image so the ops still run batched, which
        is the entire reason this lives here instead of in a pipeline transform.
        """
        groups: Dict[str, List[Tuple[int, float]]] = {}
        for i, (name, magnitude) in enumerate(specs):
            if name != NO_OP:
                groups.setdefault(name, []).append((i, magnitude))

        for name, members in groups.items():
            idx = torch.as_tensor([i for i, _ in members], device=rgb01.device)
            mags = torch.as_tensor(
                [m for _, m in members], dtype=rgb01.dtype, device=rgb01.device
            )
            sub = rgb01[idx]

            if name in GEOMETRIC_OP_KEYS:
                # Geometric ops move pixels, so the label has to move with them.
                # Called INSTEAD of DIFF32_OPS[name], not in addition to it: the
                # op is `warp_affine(images, matrix)` and this is the same warp
                # with the label carried along, so the image result is identical.
                matrix = geometric_affine_matrix(name, sub, mags)
                if labels is not None:
                    sub_out, sub_labels = warp_image_and_label(
                        sub, labels[idx], matrix
                    )
                    labels = labels.index_copy(0, idx, sub_labels)
                else:
                    # No labels in this batch (inference). Reproduce the op's own
                    # image warp rather than skipping the op entirely.
                    sub_out = DIFF32_OPS[name](sub, mags)
            else:
                sub_out = DIFF32_OPS[name](sub, mags)

            rgb01 = rgb01.index_copy(0, idx, sub_out)

        return rgb01, labels

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training)

        spec = self._train_spec if training else self._eval_spec
        if spec is None:
            return data

        inputs = data["inputs"]
        data_samples = data.get("data_samples") or []

        if training:
            specs = self._sample(int(inputs.shape[0]))
        else:
            op_name, magnitude = spec
            if op_name == NO_OP:
                return data
            specs = [(op_name, magnitude)] * int(inputs.shape[0])

        # De-normalize to the ops' [0, 1] RGB contract, apply, re-normalize.
        # Identical arithmetic to CollectGradientHook._grad_for_op -- see module
        # docstring for why that matters.
        mean = self.mean.view(1, -1, 1, 1)
        std = self.std.view(1, -1, 1, 1)

        # no_grad: nothing differentiates the augmentation during training (the
        # loss is backpropagated to the model weights, and the graph stops at the
        # input). Only the probe needs the derivative, and it builds its own.
        with torch.no_grad():
            rgb01 = ((inputs * std + mean) / 255.0).clamp(0.0, 1.0)
            labels = self._gather_labels(data_samples)
            rgb01, labels = self._apply(rgb01, labels, specs)
            data["inputs"] = (rgb01 * 255.0 - mean) / std

        if labels is not None:
            self._scatter_labels(data_samples, labels)

        return data
