_base_ = "./pspnet_r50-d8_4xb2-80k_cityscapes-512x1024.py"
model = dict(
    pretrained="open-mmlab://resnet18_v1c",
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64),
)
# base_batch_size = 4 GPUs x 2 samples; enables auto LR scaling on fewer GPUs
auto_scale_lr = dict(base_batch_size=8)
