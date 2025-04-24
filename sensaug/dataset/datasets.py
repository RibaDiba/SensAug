# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.datasets.cityscapes import CityscapesDataset


@DATASETS.register_module()
class A2I2HazeDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to 'c.jpg' and ``seg_map_suffix`` is
    fixed to 'c_labelTrainIds.png' for Cityscapes dataset.
    """

    METAINFO = dict(
        classes=(
            "sedan",
            "van",
            "utv",
            "pickup",
            "tool",
            "rope",
            "extinguisher",
            "case",
            "barrel",
            "helmet",
            "backpack",
            "survivor",
            "truss",
            "barrier",
        ),
        palette=[
            [0, 0, 142],
            [0, 0, 90],
            [0, 0, 110],
            [0, 0, 70],
            [153, 153, 153],
            [220, 220, 0],
            [119, 11, 32],
            [244, 35, 232],
            [0, 60, 100],
            [250, 170, 30],
            [107, 142, 35],
            [220, 20, 60],
            [70, 70, 70],
            [180, 165, 180],
        ],
    )

    def __init__(
        self, img_suffix="c.jpg", seg_map_suffix="c_labelTrainIds.png", **kwargs
    ) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class ACDCDataset(CityscapesDataset):
    """ACDCDataset dataset."""

    def __init__(
        self,
        img_suffix="_rgb_anon.png",
        seg_map_suffix="_gt_labelTrainIds.png",
        **kwargs,
    ) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)


@DATASETS.register_module()
class IDDDataset(CityscapesDataset):
    """IDDDataset dataset."""

    def __init__(
        self,
        img_suffix="_leftImg8bit.png",
        seg_map_suffix="_gtFine_labelTrainIds.png",
        **kwargs,
    ) -> None:
        super().__init__(img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
