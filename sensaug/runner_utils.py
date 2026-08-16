from typing import List, Dict
from copy import deepcopy

import torch

from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.loops import _InfiniteDataloaderIterator
from mmseg.registry import DATASETS

# Local imports
from sensaug.dataset.augmentations import *
from sensaug.dataset.gpu_augment import (
    clear_spec,
    set_eval_spec,
    set_train_spec,
)


def _is_gpu_set(perturbation_set) -> bool:
    """Whether this set's ops are applied batched on GPU rather than by inserting
    a transform into a dataloader pipeline.

    Every `apply_*` below branches on this FIRST. The GPU set shares all 32 keys
    with the CPU 32-op set, so anything reached by name alone would resolve to a
    CPU transform and silently augment differently from what the probe measures.
    """
    return perturbation_set == GPU_PERTURBATION_SET  # noqa: F405


def _perturbation_transform_cfg(p_type, value, perturbation_set: str = None):
    """Map a perturbation name + magnitude onto a registered transform cfg.

    `perturbation_set` disambiguates: "legacy20" and "non-diff32" are different
    vocabularies, and a name found in one is not necessarily meant for the other.
    Left as None (every pre-existing caller) the old name-based order applies.

    Never reached for the GPU set -- those ops have no pipeline transform at all,
    and the callers below return before building a pipeline. The explicit raise is
    a tripwire for a future caller that forgets to branch.
    """
    if _is_gpu_set(perturbation_set):
        raise ValueError(
            f"perturbation_set={perturbation_set!r} has no pipeline transforms; "
            "its ops are applied batched on GPU by sensaug.dataset.gpu_augment. "
            "The caller should have taken the GPU branch instead of building a "
            "perturbed pipeline."
        )
    if p_type in IMAGENETC_NAME_FN_DICT.keys():  # noqa: F405
        return "ImageNetCTransform", dict(
            type="ImageNetCTransform", name=p_type, severity=value
        )
    if perturbation_set is not None and p_type in resolve_perturbation_set(  # noqa: F405
        perturbation_set
    ):
        transform_cls, _ = resolve_perturbation_set(perturbation_set)[p_type]  # noqa: F405
        return transform_cls.__name__, dict(
            type=transform_cls.__name__, magnitude=value
        )
    if p_type in LEGACY20_OPS.keys():  # noqa: F405
        return p_type, dict(type=p_type, magnitude=value)
    if p_type in NON_DIFF32_OPS.keys():  # noqa: F405
        # Unlike LEGACY20_OPS, the key is NOT the registered type name here
        # (`lighter_R` -> `LighterR`), so the class name has to be read off the
        # registry rather than reused. Returning the key would register a transform
        # that does not exist and trip _assert_transforms_present, which compares
        # against `t.__class__.__name__`.
        transform_cls, _ = NON_DIFF32_OPS[p_type]  # noqa: F405
        return transform_cls.__name__, dict(
            type=transform_cls.__name__, magnitude=value
        )
    if p_type == "combination":
        return "CombinationPerturbation", dict(
            type="CombinationPerturbation", choice=value
        )
    raise ValueError(f"transform name doesn't match anything: {p_type}")


def _build_perturbed_pipeline(
    dataloader_cfg, perturb_levels: Dict, train: bool, perturbation_set: str = None
):
    """Insert perturbation transforms into dataloader_cfg's pipeline, in place.

    Returns the list of transform type names that were inserted.
    """
    pipeline = dataloader_cfg.dataset.pipeline

    insert_index = -1
    if len(perturb_levels) != 0:
        if not train:
            for i, aug in enumerate(pipeline):
                if aug["type"] == "Resize":
                    pipeline.insert(1, pipeline.pop(i))
                    break

        for i, aug in enumerate(pipeline):  # put transform after load annotations
            if aug["type"] == "LoadAnnotations":
                pipeline.insert(1, pipeline.pop(i))
                break

    inserted: List[str] = []
    for p_type, value in perturb_levels.items():
        transform_name, kwargs = _perturbation_transform_cfg(
            p_type, value, perturbation_set=perturbation_set
        )
        pipeline.insert(insert_index, kwargs)
        inserted.append(transform_name)

    return inserted


def _assert_transforms_present(dataloader, expected: List[str], tag: str):
    """Guard: the loader we are about to hand to the workers really carries the
    perturbations we asked for. Mutating `dataset.pipeline` on an already-running
    loader does not reach worker processes when persistent_workers=True, so a
    silently-clean dataloader is the failure mode this catches.
    """
    present = [t.__class__.__name__ for t in dataloader.dataset.pipeline.transforms]
    missing = [name for name in expected if name not in present]
    if missing:
        raise RuntimeError(
            f"{tag} dataloader is missing perturbation transforms {missing} after "
            f"rebuild; pipeline is {present}. Perturbations would silently not be "
            f"applied and every perturbed eval would return clean metrics."
        )


def _rebuild_loader(runner: Runner, loop, dataloader_cfg, expected: List[str], tag: str):
    """Rebuild a loop's dataloader from cfg so fresh workers pick up the pipeline."""
    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )
    _assert_transforms_present(new_dataloader, expected, tag)
    loop.dataloader = new_dataloader
    return new_dataloader


def _rebind_train_iterator(runner: Runner):
    """IterBasedTrainLoop caches an iterator over the *old* dataloader at __init__
    time and, with InfiniteSampler, never re-creates it. Swapping
    train_loop.dataloader alone therefore changes nothing.
    """
    loop = runner.train_loop
    if hasattr(loop, "dataloader_iterator"):
        loop.dataloader_iterator = _InfiniteDataloaderIterator(loop.dataloader)


def verify_perturbation_effective(
    runner: Runner,
    p_type: str,
    magnitude: float = 1.0,
    sample_idx: int = 0,
    perturbation_set: str = None,
):
    """Functional self-test: at full magnitude, a perturbation must change pixels.

    Cheap (one sample, main process) and run once per SA pass. Uses magnitude=1.0
    because the adaptive search can select magnitudes small enough to vanish under
    uint8 rounding, which would make a strict pixel-difference check flaky.
    """
    if _is_gpu_set(perturbation_set):
        # Nothing is inserted into a pipeline on this path, so the dataloader
        # comparison below cannot say anything. Check the op itself instead: the
        # equivalent failure is an op that silently returns its input.
        op = resolve_perturbation_set(perturbation_set)[p_type]  # noqa: F405
        probe = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            perturbed = op(probe, float(magnitude))
        if perturbed.shape == probe.shape and torch.allclose(probe, perturbed):
            raise RuntimeError(
                f"Perturbation {p_type} at magnitude {magnitude} left its input "
                f"unchanged. Sensitivity analysis would measure clean performance "
                f"for this perturbation. Refusing to produce a meaningless SA curve."
            )
        print_log(
            f"GPU perturbation verified: {p_type}@{magnitude} alters input data",
            logger="current",
        )
        return

    clean_cfg = deepcopy(runner.cfg.val_dataloader)
    pert_cfg = deepcopy(runner.cfg.val_dataloader)
    _build_perturbed_pipeline(
        pert_cfg, {p_type: magnitude}, train=False, perturbation_set=perturbation_set
    )

    clean = DATASETS.build(clean_cfg.dataset)[sample_idx]["inputs"]
    perturbed = DATASETS.build(pert_cfg.dataset)[sample_idx]["inputs"]

    if clean.shape == perturbed.shape and torch.equal(clean, perturbed):
        raise RuntimeError(
            f"Perturbation {p_type} at magnitude {magnitude} left the input sample "
            f"bit-identical to clean. The perturbation pipeline is not taking effect; "
            f"sensitivity analysis would measure clean performance for every "
            f"perturbation. Refusing to produce a meaningless SA curve."
        )

    print_log(
        f"Perturbation pipeline verified: {p_type}@{magnitude} alters input data",
        logger="current",
    )


def create_union_test_set_new(runner: Runner, perturb_levels: Dict = {}):
    dataset_cfgs: List = []
    dataloader_cfg = deepcopy(runner.cfg.test_dataloader)

    insert_index = -1
    for i, aug in enumerate(
        dataloader_cfg.dataset.pipeline
    ):  # put LoadAnnotations at the top of the pipeline
        if aug["type"] == "LoadAnnotations":
            dataloader_cfg.pipeline.insert(1, dataloader_cfg.pipeline.pop(i))
            # insert_index = i + 1
            break

    # create one dataset for each perturbation type
    for p_type, level in perturb_levels.items():
        transform_cls, is_parameterized = LEGACY20_OPS[p_type]
        transform = transform_cls(magnitude=level)

        # make dataset cfg
        p_dataset_cfg = deepcopy(dataloader_cfg.dataset)
        p_dataset_cfg.pipeline.insert(
            insert_index, dict(type="PerturbationTransform", transform=transform)
        )

        # add dataset cfg to list
        dataset_cfgs.append(p_dataset_cfg)

    # create clean dataset
    p_dataset_cfg = deepcopy(dataloader_cfg.dataset)
    dataset_cfgs.append(p_dataset_cfg)

    # create a concat dataset type and merge the datasets
    dataloader_cfg.dataset = dict(type="ConcatDataset", datasets=dataset_cfgs)

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )

    # set to test loader in runner
    runner.test_loop.dataloader = new_dataloader  # type: ignore


def create_union_train_set_new(runner: Runner, perturb_levels: Dict = {}):
    dataloader_cfg = deepcopy(runner.cfg.train_dataloader)
    dataset_cfgs: List = [dataloader_cfg.dataset]

    insert_index = -1
    for i, aug in enumerate(
        dataloader_cfg.dataset.pipeline
    ):  # put transform after load annotations
        if aug["type"] == "LoadAnnotations":
            dataloader_cfg.pipeline.insert(1, dataloader_cfg.pipeline.pop(i))
            # insert_index = i + 1
            break

    # create one dataset for each perturbation type
    for p_type, level in perturb_levels.items():
        transform_cls, is_parameterized = LEGACY20_OPS[p_type]  # noqa: F405
        transform = transform_cls(magnitude=level)

        # make dataset cfg
        p_dataset_cfg = deepcopy(dataloader_cfg.dataset)
        p_dataset_cfg.pipeline.insert(
            insert_index, dict(type="PerturbationTransform", transform=transform)
        )

        # add dataset cfg to list
        dataset_cfgs.append(p_dataset_cfg)

    # create a concat dataset type and merge the datasets
    dataloader_cfg.dataset = dict(type="ConcatDataset", datasets=dataset_cfgs)

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )

    # set to train loader in runner
    runner.train_loop.dataloader = new_dataloader  # type: ignore


def apply_perturbations_dataloader(
    runner: Runner,
    train: bool = False,
    perturb_levels: Dict = {},
    perturbation_set: str = None,
):
    """General perturation dataloader utility function.
    Changes the perturbations applied on the dataloader, given an existing runner object.
    Modifies 'runner' directly, returns nothing.

    The dataloader is rebuilt rather than having its dataset.pipeline reassigned:
    the configs set persistent_workers=True, so worker processes hold their own copy
    of the dataset and never observe an in-place pipeline swap.

    `perturbation_set` names the vocabulary `perturb_levels`' keys are drawn from;
    see _perturbation_transform_cfg for why the names alone are not enough. Callers
    resetting to clean (`perturb_levels={}`) need not pass it -- nothing is resolved.

    Under the GPU set no dataloader is rebuilt at all: the perturbation becomes
    state on the model's data preprocessor. That is the whole point -- the SA
    round-eval used to rebuild the val dataloader once per (op, level) pair, and
    each rebuild dragged the per-image CPU path along with it.
    """
    if _is_gpu_set(perturbation_set):
        if not perturb_levels:
            clear_spec(runner, training=train)
            return
        if train:
            # No caller does this: the training policy is a distribution, set by
            # apply_random_*_train_* below, not a single pinned perturbation. If
            # one ever wants it, decide there whether the pdf jitter applies.
            raise NotImplementedError(
                "pinning a single perturbation on the TRAIN path is not supported "
                "for the GPU set; use apply_random_alpha_training_augmentations or "
                "apply_random_perturbations_train_dataloader_new"
            )
        if len(perturb_levels) != 1:
            raise ValueError(
                "the GPU eval path applies one op at a time; got "
                f"{len(perturb_levels)} in perturb_levels={perturb_levels!r}"
            )
        ((op_name, magnitude),) = perturb_levels.items()
        set_eval_spec(runner, op_name, magnitude)
        print_log(
            f"GPU eval perturbation: {op_name}@{float(magnitude):.4g}",
            logger="current",
        )
        return

    if train:
        dataloader_cfg = deepcopy(runner.cfg.train_dataloader)
        inserted = _build_perturbed_pipeline(
            dataloader_cfg, perturb_levels, train=True, perturbation_set=perturbation_set
        )
        print_log(
            f"Rebuilding train loop dataloader: {dataloader_cfg.dataset.pipeline}",
            logger="current",
        )
        _rebuild_loader(runner, runner.train_loop, dataloader_cfg, inserted, "train")
        _rebind_train_iterator(runner)
        return

    # val and test loops are both driven off val_dataloader cfg, as before
    for tag, loop in (("val", runner.val_loop), ("test", runner.test_loop)):
        dataloader_cfg = deepcopy(runner.cfg.val_dataloader)
        inserted = _build_perturbed_pipeline(
            dataloader_cfg,
            perturb_levels,
            train=False,
            perturbation_set=perturbation_set,
        )
        print_log(
            f"Rebuilding {tag} loop dataloader: {dataloader_cfg.dataset.pipeline}",
            logger="current",
        )
        _rebuild_loader(runner, loop, dataloader_cfg, inserted, tag)


def apply_random_perturbations_train_dataloader_new(
    runner: Runner, pdf_dict: Dict, perturbation_set: str = "legacy20"
):
    if _is_gpu_set(perturbation_set):
        # The pdf becomes preprocessor state; the dataloader is left alone. Note
        # this is why the GPU set does not need the dataloader rebuilt on every SA
        # round -- the pdf can change without touching the workers at all.
        set_train_spec(runner, pdf_dict=pdf_dict)
        print_log(
            f"GPU train pdf updated ({len(pdf_dict)} entries)", logger="current"
        )
        return

    dataloader_cfg = deepcopy(runner.cfg.train_dataloader)

    pipeline = [
        x
        for x in dataloader_cfg.dataset.pipeline
        if x["type"] != "RandomAlphaTrainTransform"
    ]

    insert_index = -1
    for i, aug in enumerate(pipeline):  # put transform after load annotations
        if aug["type"] == "LoadAnnotations":
            pipeline.insert(1, pipeline.pop(i))
            break

    pipeline.insert(
        insert_index,
        dict(
            type="RandomTrainTransformNew",
            pdf_dict=pdf_dict,
            # The pdf's op names come from whichever vocabulary SA ran over; the
            # transform has to look them up in the same one or it KeyErrors on the
            # first augmented image.
            perturbation_set=perturbation_set,
        ),
    )

    dataloader_cfg.dataset.pipeline = pipeline

    print_log(f"Rebuilding train loop dataloader: {pipeline}", logger="current")
    _rebuild_loader(
        runner, runner.train_loop, dataloader_cfg, ["RandomTrainTransformNew"], "train"
    )
    _rebind_train_iterator(runner)


def apply_random_alpha_training_augmentations(
    runner: Runner,
    geometric_only=False,
    photometric_only=False,
    perturbation_set: str = "legacy20",
):
    if _is_gpu_set(perturbation_set):
        set_train_spec(
            runner,
            geometric_only=geometric_only,
            photometric_only=photometric_only,
        )
        print_log("GPU train policy set to uniform", logger="current")
        return

    dataloader_cfg = deepcopy(runner.cfg.train_dataloader)

    pipeline = dataloader_cfg.dataset.pipeline

    insert_index = -1
    for i, aug in enumerate(pipeline):  # put transform after load annotations
        if aug["type"] == "LoadAnnotations":
            pipeline.insert(1, pipeline.pop(i))
            break

    pipeline.insert(
        insert_index,
        dict(
            type="RandomAlphaTrainTransform",
            geometric_only=geometric_only,
            photometric_only=photometric_only,
            perturbation_set=perturbation_set,
        ),
    )

    dataloader_cfg.dataset.pipeline = pipeline

    print_log(f"Rebuilding train loop dataloader: {pipeline}", logger="current")
    _rebuild_loader(
        runner, runner.train_loop, dataloader_cfg, ["RandomAlphaTrainTransform"], "train"
    )
    _rebind_train_iterator(runner)


def apply_random_perturbations_test_dataloader(
    runner: Runner, pdf_dict: Dict, perturbation_set: str = "legacy20"
):
    """Applies the random perturbation dataloader based on a given PDF dictionary.
    Specifically, applies 'RandomTrainTransform' from bp.robustness.augmentations to the current
    Runner's *test* dataloader.
    """

    dataloader_cfg = deepcopy(runner.cfg.test_dataloader)

    pipeline = dataloader_cfg.dataset.pipeline

    insert_index = -1
    for i, aug in enumerate(pipeline):  # put transform after load annotations
        if aug["type"] == "LoadAnnotations":
            pipeline.insert(1, pipeline.pop(i))
            # insert_index = i + 1
            break

    pipeline.insert(
        insert_index,
        dict(
            type="RandomTrainTransformNew",
            pdf_dict=pdf_dict,
            perturbation_set=perturbation_set,
        ),
    )

    dataloader_cfg.dataset.pipeline = pipeline

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )

    runner.test_loop.dataloader = new_dataloader  # type: ignore

    return new_dataloader
