dataset_type = 'ChaseDB1Dataset'
data_root = 'data/CHASE_DB1'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (960, 999)
crop_size = (128, 128)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(960, 999), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(128, 128), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(128, 128), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(960, 999),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=400,
        dataset=dict(
            type='ChaseDB1Dataset',
            data_root='data/CHASE_DB1',
            img_dir='images/training',
            ann_dir='annotations/training',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='Resize',
                    img_scale=(960, 999),
                    ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(128, 128),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(128, 128), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ])),
    val=dict(
        type='ChaseDB1Dataset',
        data_root='data/CHASE_DB1',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 999),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ChaseDB1Dataset',
        data_root='data/CHASE_DB1',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 999),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[65, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=1)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='SyncBN', requires_grad=True)
drop_probs = (0.1, 0.1, 0.1, -1)
channels = (128, 128, 256, 512)
decoder_channel = 128
auxiliary_channel = 256
blocks = (2, 2, 2, 2)
m_module_paths = 8
m_module_groups = 4
m_module_expand_ratio = 2
act_cfg = [dict(type='ReLU'), dict(type='ReLU')]
upsample_cfg = dict(mode='bilinear', align_corners=False)
num_classes = 2
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='TwistNet_ResNet',
        slave_depth=50,
        slave_strides=(1, 2, 2, 2),
        slave_dilations=(1, 1, 1, 1),
        slave_avg_down=False,
        slave_conv_cfg=None,
        slave_norm_cfg=dict(type='SyncBN', requires_grad=True),
        slave_norm_eval=False,
        slave_act_cfg=dict(type='ReLU'),
        slave_drop_prob=0.1,
        slave_init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/new_one/RGB_weight_hb2next_resnet50.pth'),
        master_channels=(128, 128, 256, 512),
        master_blocks=(2, 2, 2, 2),
        master_strides=(2, 2, 1, 1),
        master_dilations=(1, 1, 2, 4),
        m_module_paths=8,
        m_module_groups=4,
        m_module_expand_ratio=2,
        ct_module_compress_ratios=(1, 1),
        bt_modes=(None, 'dual_mul', 'dual_mul', 'dual_mul_e'),
        down_type=None,
        upsample_cfg=dict(mode='bilinear', align_corners=False),
        master_order=('conv', 'norm', 'act'),
        master_conv_cfg=None,
        master_norm_cfg=dict(type='SyncBN', requires_grad=True),
        master_norm_eval=False,
        master_act_cfg=dict(type='ReLU'),
        master_drop_prob=0.1,
        with_cp=False,
        multi_modals=1),
    decode_head=dict(
        type='TwistNet_Head',
        in_channels=(512, 256, 128, 128),
        channels=128,
        blocks=(2, 2, 2, 2),
        num_classes=2,
        m_module_paths=8,
        m_module_groups=4,
        m_module_expand_ratio=2,
        conv_cfg=None,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        order=('conv', 'norm', 'act'),
        drop_prob=0.1,
        dropout_ratio=-1,
        in_index=(0, 1, 2, 3),
        input_transform='multiple_select',
        align_corners=False,
        loss_decode=dict(type='DiceLoss', loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(128, 128), stride=(85, 85)))
samples_per_gpu = 16
max_epochs = 100
evaluation = dict(
    interval=1, metric=['mIoU', 'mDice', 'mFscore'], save_best='mDice')
work_dir = 'my_work_mmseg/hb2nextv2/chasedb1/ours'
gpu_ids = range(0, 2)
auto_resume = False
