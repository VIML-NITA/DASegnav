norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        # embed_dims=64,
        embed_dims=128,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        # num_layers=[3, 3, 3, 3],  # Adjusted number of layers
        # num_heads=[2, 4, 8, 16],  # Adjusted number of heads
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        # drop_rate=0.0,
        drop_rate=0.1,  # Adjusted dropout rate
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='APFormerHead',  #FeedFormerHeadUNet, FeedFormerHeadUNetPlus, FeedFormerHead32, FeedFormerHead32_new'
        feature_strides=[2, 4, 8, 16],
        # in_channels=[64, 128, 320, 512],
        in_channels=[128, 256, 640, 1024],
        in_index=[0, 1, 2, 3],
        channels=256,
        # decoder_params=dict(embed_dim=768, 
        #                     num_heads=[24, 12, 6, 3],
        #                     pool_ratio=[1, 2, 4, 8]),
        num_classes=6
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        
        ),
    
    auxiliary_head=[
        dict(
            type='FCNHead',
            # in_channels=320,
            in_channels=640,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='DiceLoss', use_sigmoid=True, loss_weight=0.4)),
        dict(
            type='FCNHead',
            # in_channels=128,
            in_channels=256,
            channels=32,
            num_convs=1,
            num_classes=6,
            in_index=-3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    ],
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
