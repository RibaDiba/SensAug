# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import os
import cv2
import mmcv
import numpy as np
import torch
from mmengine.dist import master_only
from mmengine.structures import PixelData
from mmengine.visualization import Visualizer

from mmseg.registry import VISUALIZERS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@VISUALIZERS.register_module()
class BPSegLocalVisualizer(SegLocalVisualizer):
    """Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        classes (list, optional): Input classes for result rendering, as the
            prediction of segmentation model is a segment map with label
            indices, `classes` is a list which includes items responding to the
            label indices. If classes is not defined, visualizer will take
            `cityscapes` classes by default. Defaults to None.
        palette (list, optional): Input palette for result rendering, which is
            a list of color palette responding to the classes. Defaults to None.
        dataset_name (str, optional): `Dataset name or alias <https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/class_names.py#L302-L317>`_
            visulizer will use the meta information of the dataset i.e. classes
            and palette, but the `classes` and `palette` have higher priority.
            Defaults to None.
        alpha (int, float): The transparency of segmentation mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import PixelData
        >>> from mmseg.structures import SegDataSample
        >>> from mmseg.visualization import SegLocalVisualizer

        >>> seg_local_visualizer = SegLocalVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> gt_sem_seg_data = dict(data=torch.randint(0, 2, (1, 10, 12)))
        >>> gt_sem_seg = PixelData(**gt_sem_seg_data)
        >>> gt_seg_data_sample = SegDataSample()
        >>> gt_seg_data_sample.gt_sem_seg = gt_sem_seg
        >>> seg_local_visualizer.dataset_meta = dict(
        >>>     classes=('background', 'foreground'),
        >>>     palette=[[120, 120, 120], [6, 230, 230]])
        >>> seg_local_visualizer.add_datasample('visualizer_example',
        ...                         image, gt_seg_data_sample)
        >>> seg_local_visualizer.add_datasample(
        ...                        'visualizer_example', image,
        ...                         gt_seg_data_sample, show=True)
    """  # noqa

    @master_only
    def add_datasample(
        self,
        name: str,
        image: List[np.ndarray],
        data_sample: Optional[SegDataSample] = None,
        draw_gt: bool = True,
        draw_pred: bool = True,
        show: bool = False,
        wait_time: float = 0,
        # TODO: Supported in mmengine's Viusalizer.
        out_file: Optional[str] = None,
        step: int = 0,
        with_labels: Optional[bool] = True,
    ) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Defaults to True.
        """

        assert len(image) == 2, (
            f"BPSegLocalVisualizer takes two images: the original image and the augmented version. \
            The provided input has {len(image)} image(s)."
        )

        image_aug = image[1]  # augmented version of image
        image = image[0]  # original image

        classes = self.dataset_meta.get("classes", None)
        palette = self.dataset_meta.get("palette", None)

        gt_img_data = image_aug
        pred_img_data = image_aug

        if draw_gt and data_sample is not None:
            if "gt_sem_seg" in data_sample:
                assert classes is not None, (
                    "class information is "
                    "not provided when "
                    "visualizing semantic "
                    "segmentation results."
                )
                gt_img_data = self._draw_sem_seg(
                    image, data_sample.gt_sem_seg, classes, palette, with_labels
                )

            if "gt_depth_map" in data_sample:
                gt_img_data = gt_img_data if gt_img_data is not None else image
                gt_img_data = self._draw_depth_map(
                    gt_img_data, data_sample.gt_depth_map
                )

        if draw_pred and data_sample is not None:
            if "pred_sem_seg" in data_sample:
                assert classes is not None, (
                    "class information is "
                    "not provided when "
                    "visualizing semantic "
                    "segmentation results."
                )

                pred_img_data = self._draw_sem_seg(
                    image_aug, data_sample.pred_sem_seg, classes, palette, with_labels
                )

            if "pred_depth_map" in data_sample:
                pred_img_data = (
                    pred_img_data if pred_img_data is not None else image_aug
                )
                pred_img_data = self._draw_depth_map(
                    pred_img_data, data_sample.pred_depth_map
                )

        if show:
            self.show(gt_img_data, win_name=name + "_gt", wait_time=wait_time)
            self.show(pred_img_data, win_name=name + "_pred", wait_time=wait_time)

        if out_file is not None:
            split_filename = out_file.split(".")
            filename, ext = ".".join(split_filename[:-1]), split_filename[-1]
            gt_filename = f"{filename}_gt.{ext}"
            pred_filename = f"{filename}_pred.{ext}"
            mmcv.imwrite(mmcv.rgb2bgr(gt_img_data), gt_filename)
            mmcv.imwrite(mmcv.rgb2bgr(pred_img_data), pred_filename)
        else:
            self.add_image(name + "_gt", gt_img_data, step)
            self.add_image(name + "_pred", pred_img_data, step)
