from typing import List, Dict
from copy import deepcopy

import torch

from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.runner.loops import _InfiniteDataloaderIterator
from mmseg.registry import DATASETS

# Local imports
from sensaug.dataset.augmentations import *


def _perturbation_transform_cfg(p_type, value, perturbation_set: str = None):
    """Map a perturbation name + magnitude onto a registered transform cfg.

    `perturbation_set` disambiguates. DIFF_PERTURBATIONS and ALIGNED_PERTURBATIONS
    share all 32 keys and mean different classes under each -- `lighter_R` is the
    per-image torch wrapper DiffLighterR in one and the plain cv2 LighterR in the
    other. Dispatching on the name alone would silently resolve every aligned-set
    sweep onto the diff wrappers, i.e. onto the 40-150x slower path the aligned
    vocabulary exists to avoid, and nothing downstream would say so: the transform
    names it inserts are real and _assert_transforms_present would pass.

    Left as None (every pre-existing caller) the old name-based order applies.
    """
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
    if p_type in NEW_PERTURBATIONS.keys():  # noqa: F405
        return p_type, dict(type=p_type, magnitude=value)
    if p_type in DIFF_PERTURBATIONS.keys():  # noqa: F405
        # Unlike NEW_PERTURBATIONS, the key is NOT the registered type name here
        # (`lighter_R` -> `DiffLighterR`), so the class name has to be read off the
        # registry rather than reused. Returning the key would register a transform
        # that does not exist and trip _assert_transforms_present, which compares
        # against `t.__class__.__name__`.
        transform_cls, _ = DIFF_PERTURBATIONS[p_type]  # noqa: F405
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
        transform_cls, is_parameterized = NEW_PERTURBATIONS[p_type]
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
        transform_cls, is_parameterized = NEW_PERTURBATIONS[p_type]  # noqa: F405
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
    """

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
    runner: Runner, pdf_dict: Dict, perturbation_set: str = "new"
):
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
    perturbation_set: str = "new",
):
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
    runner: Runner, pdf_dict: Dict, perturbation_set: str = "new"
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
