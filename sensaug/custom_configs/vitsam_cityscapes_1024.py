# /home/lyzheng/projects/vision_robustness/custom_configs/mmseg/_base_/models/upernet_vit-b16_ln_mln.py
# /home/lyzheng/projects/vision_robustness/custom_configs/mmseg/vit/vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512.py

_base_ = [
    # './mmseg/_base_/models/upernet_vit-b16_ln_mln.py',
    "./mmseg/_base_/datasets/cityscapes.py",
    "./mmseg/_base_/default_runtime.py",
    "./mmseg/_base_/schedules/schedule_80k.py",
]
crop_size = (1024, 1024)

data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)
pretrained_sam = "https://download.openmmlab.com/mmclassification/v1/vit_sam/vit-base-p16_sam-pre_3rdparty_sa1b-1024px_20230411-2320f9cc.pth"
norm_cfg = dict(requires_grad=True, type="BN")
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    auxiliary_head=dict(
            align_corners=False,
            channels=256,
            concat_input=False,
            dropout_ratio=0.1,
            in_channels=256,
            in_index=2,
            loss_decode=dict(
                loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=norm_cfg,
            num_classes=19,
            num_convs=1,
            type='FCNHead'),
    backbone = dict(
            type="mmpretrain.ViTSAM",  # Using MobileNetV3 from mmpretrain
            arch="base",
            img_size=1024,
            patch_size=16,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
            init_cfg=dict(
                type="Pretrained", checkpoint=pretrained_sam, prefix="backbone."
            ),
            out_indices=(
                3,
                7,
                11,
                12,
            ),
            # out_indices = [3,7,21,24],
            out_type="raw",  # (B, C, H, W)
        ),  # The pre-trained weights of backbone network in mmpretrain have prefix='backbone.'. The prefix in the keys will be removed so that these weights can be normally loaded.
    decode_head = dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[
            256,
            256,
            256,
        ],
        in_index=[
            0,
            1,
            2,
        ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=norm_cfg,
        num_classes=19,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='UPerHead'),
    neck = dict(
        type='MultiLevelNeck',
        in_channels=[256, 256, 256],
        out_channels=256,
        scales=[2, 1, 0.5]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_embed": dict(decay_mult=0.0),
            "cls_token": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
