from typing import List, Dict
from copy import deepcopy

from mmengine.logging import print_log
from mmengine.runner import Runner, BaseLoop
from mmengine.dataset.base_dataset import Compose

# Local imports
from sensaug.dataset.augmentations import *


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
    runner: Runner, train: bool = False, perturb_levels: Dict = {}
):
    """General perturation dataloader utility function.
    Changes the perturbations applied on the dataloader, given an existing runner object.
    Modifies 'runner' directly, returns nothing.
    """

    dataloader_cfg = (
        deepcopy(runner.cfg.train_dataloader)
        if train
        else deepcopy(runner.cfg.val_dataloader)
    )

    pipeline = dataloader_cfg.dataset.pipeline

    insert_index = -1
    if len(perturb_levels) != 0:
        if train == False:
            for i, aug in enumerate(pipeline):  # put transform after load annotations
                if aug["type"] == "Resize":
                    pipeline.insert(1, pipeline.pop(i))
                    break

        for i, aug in enumerate(pipeline):  # put transform after load annotations
            if aug["type"] == "LoadAnnotations":
                pipeline.insert(1, pipeline.pop(i))
                break

    for p_type, value in perturb_levels.items():
        transform_name = None
        kwargs = {}

        if p_type in IMAGENETC_NAME_FN_DICT.keys():  # noqa: F405
            transform_name = "ImageNetCTransform"
            kwargs = {"type": transform_name, "name": p_type, "severity": value}
        elif p_type in NEW_PERTURBATIONS.keys():  # noqa: F405
            transform_name = p_type
            kwargs = {"type": transform_name, "magnitude": value}
        elif p_type == "combination":
            transform_name = "CombinationPerturbation"
            kwargs = {"type": transform_name, "choice": value}

        assert transform_name is not None, "transform name doesn't match anything"

        pipeline.insert(
            insert_index, dict(**kwargs)
        )  # before load annotations and pack seg inputs

    built_pipeline = Compose(pipeline)

    if train:
        print_log(f"Setting pipeline for train loop: {pipeline}", logger="current")

        if isinstance(runner._train_loop, BaseLoop):
            runner.train_loop.dataloader.dataset.pipeline = built_pipeline  # type: ignore
        else:
            dataloader_cfg.dataset.pipeline = pipeline
            diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
            new_dataloader = runner.build_dataloader(
                dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
            )
            runner.train_loop.dataloader = new_dataloader  # type: ignore
    else:
        print_log(f"Setting pipeline for val loop: {pipeline}", logger="current")
        if isinstance(runner._val_loop, BaseLoop):
            runner.val_loop.dataloader.dataset.pipeline = built_pipeline  # type: ignore
        else:
            dataloader_cfg.dataset.pipeline = pipeline
            diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
            new_dataloader = runner.build_dataloader(
                dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
            )
            runner.val_loop.dataloader = new_dataloader  # type: ignore

        print_log(f"Setting pipeline for test loop: {pipeline}", logger="current")
        if isinstance(runner._test_loop, BaseLoop):
            runner.test_loop.dataloader.dataset.pipeline = built_pipeline  # type: ignore
        else:
            dataloader_cfg.dataset.pipeline = pipeline
            diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
            new_dataloader = runner.build_dataloader(
                dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
            )
            runner.test_loop.dataloader = new_dataloader  # type: ignore


def apply_random_perturbations_train_dataloader_new(runner: Runner, pdf_dict: Dict):
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
        insert_index, dict(type="RandomTrainTransformNew", pdf_dict=pdf_dict)
    )

    built_pipeline = Compose(pipeline)

    # dataloader_cfg.dataset.pipeline = pipeline
    # diff_rank_seed = runner._randomness_cfg.get(
    #     'diff_rank_seed', False)
    # new_dataloader = runner.build_dataloader(
    #     dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed)

    if isinstance(runner._train_loop, BaseLoop):
        print_log(f"Setting pipeline for train loop: {pipeline}", logger="current")
        runner.train_loop.dataloader.dataset.pipeline = built_pipeline  # type: ignore
    # runner.train_loop.dataloader = new_dataloader # type: ignore


def apply_random_alpha_training_augmentations(
    runner: Runner, geometric_only=False, photometric_only=False
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
        ),
    )

    # dataloader_cfg.dataset.pipeline = pipeline
    # diff_rank_seed = runner._randomness_cfg.get(
    #     'diff_rank_seed', False)
    # new_dataloader = runner.build_dataloader(
    #     dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed)
    new_pipeline = Compose(pipeline)

    # runner.train_loop.dataloader = new_dataloader # type: ignore
    if isinstance(runner._train_loop, BaseLoop):
        print_log(f"Setting pipeline for train loop: {pipeline}", logger="current")
        runner.train_loop.dataloader.dataset.pipeline = new_pipeline  # type: ignore


def apply_random_perturbations_test_dataloader(runner: Runner, pdf_dict: Dict):
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
        insert_index, dict(type="RandomTrainTransformNew", pdf_dict=pdf_dict)
    )

    dataloader_cfg.dataset.pipeline = pipeline

    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
    new_dataloader = runner.build_dataloader(
        dataloader_cfg, seed=runner.seed, diff_rank_seed=diff_rank_seed
    )

    runner.test_loop.dataloader = new_dataloader  # type: ignore

    return new_dataloader
