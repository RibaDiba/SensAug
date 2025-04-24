# The new config inherits a base config to highlight the necessary modification
_base_ = "mmdet/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py"

# training schedule for 1x
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001),
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1))
)

# Modify dataset related settings
data_root = (
    "/fs/nexus-projects/robustness_datasets/iclr_detection/datasets/balloon/balloon/"
)
dataset_type = "CocoDataset"

metainfo = {
    "classes": ("balloon",),
    "palette": [
        (220, 20, 60),
    ],
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        ann_file="train/annotation_coco.json",
        data_prefix=dict(img="train/"),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        metainfo=metainfo,
        ann_file="val/annotation_coco.json",
        data_prefix=dict(img="val/"),
    )
)
test_dataloader = val_dataloader
# test_dataloader = train_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val/annotation_coco.json")
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
