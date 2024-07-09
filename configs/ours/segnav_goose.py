_base_ = [
    '../_base_/models/segnavex.py', '../_base_/datasets/goose_group6.py',
    '../_base_/default_runtime.py'
]


# optimizer = dict(type='SGD', lr=0.09, weight_decay=0.01)
optimizer = dict(type='SGD', lr=0.093, weight_decay=4e-5)  # Adjusted learning rate
# optimizer = dict(type='SGD', lr=0.12, weight_decay=0.001)

optimizer_config = dict()
# learning policy
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=240000)
total_iters = 240000
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=240000, metric='mIoU')

# optimizer
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    by_epoch=False)


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

# loss_decode=dict(
#     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
loss_decode=dict(
        type='DiceLoss',  # Example: Using Dice loss instead of CrossEntropyLoss
        use_sigmoid=True,
        loss_weight=1.0
    )
