_base_ = [
    '../_base_/models/segnavex.py', '../_base_/datasets/rugd_group6_new2.py',
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
    samples_per_gpu=2,
    workers_per_gpu=6)

# loss_decode=dict(
#     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
loss_decode=dict(
        type='DiceLoss',  # Example: Using Dice loss instead of CrossEntropyLoss
        use_sigmoid=True,
        loss_weight=1.0
    )

# lr=0.95 and 0.0001 gave 84.74 in aAcc and miou of 56.17
# optimizer = dict(type='SGD', lr=0.09, weight_decay=4e-5)  gave best results
# +-------+-------+------+
# |  aAcc |  mIoU | mAcc |
# +-------+-------+------+
# | 81.75 | 57.48 | 68.7 |

# lr = 0.093
# +-------+-------+-------+
# |  aAcc |  mIoU |  mAcc |
# +-------+-------+-------+
# | 83.88 | 58.72 | 68.15 |
# +-------+-------+-------+

# [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 1401/1401, 25.7 task/s, elapsed: 55s, ETA:     0s2024-03-31 16:05:36,531 - mmseg - INFO - per class results:
# 2024-03-31 16:05:36,532 - mmseg - INFO - 
# +-----------------+-------+-------+
# |      Class      |  IoU  |  Acc  |
# +-----------------+-------+-------+
# |    background   | 80.99 |  89.1 |
# |      stable     |  81.9 | 91.98 |
# |     granular    | 82.68 | 88.95 |
# |  poor foothold  | 70.89 | 90.94 |
# | high resistance | 87.45 | 92.04 |
# |     obstacle    | 90.16 |  96.2 |
# +-----------------+-------+-------+
# 2024-03-31 16:05:36,532 - mmseg - INFO - Summary:
# 2024-03-31 16:05:36,532 - mmseg - INFO - 
# +------+-------+-------+
# | aAcc |  mIoU |  mAcc |
# +------+-------+-------+
# | 93.2 | 82.34 | 91.54 |
# +------+-------+-------+


# ./configs/ours/segnav_cfg.py