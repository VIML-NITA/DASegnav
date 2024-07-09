_base_ = [
    '../_base_/models/segnavex.py', '../_base_/datasets/yamaha_group6.py',
    '../_base_/default_runtime.py'
]


optimizer = dict(type='SGD', lr=0.093, weight_decay=4e-5)

optimizer_config = dict()

runner = dict(type='IterBasedRunner', max_iters=240)
total_iters = 240
checkpoint_config = dict(by_epoch=False, interval=16)
evaluation = dict(interval=240, metric='mIoU')

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    by_epoch=False)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6)


loss_decode=dict(
        type='DiceLoss',
        use_sigmoid=True,
        loss_weight=1.0
    )
