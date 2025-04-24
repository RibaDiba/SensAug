_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/a2i2haze.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]
    ),
    auxiliary_head=dict(in_channels=256, channels=64,
                        loss_decode=[
                            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
                            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
                        ]
    ))
    