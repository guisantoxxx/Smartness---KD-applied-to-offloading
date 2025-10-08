_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/atss/atss_r101_fpn_1x_coco/atss_r101_fpn_1x_20200825-dfcadd6f.pth'

# Guilherme: Configurações gerais de treino, modelo, dados, etc.
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32
)

# Guilherme: Definição do modelo e seus componentes (backbone, neck, head)
model = dict(
    type='CrossKDATSS',  # modelo customizado CrossKD, ao passar esse tipo, o modelo se comportará de acordo com o comportamente esperado da classe crossKD definida em mmdet/models
    data_preprocessor=data_preprocessor,

    # Guilherme: Aqui ele define quem sera o professor e suas configs
    teacher_config='configs/atss/atss_r101_fpn_1x_coco.py',
    teacher_ckpt=teacher_ckpt,

    # Daqui em diante define como serao as configs do estudante (backbone, neck e head)

    # Backbone: extrator de features
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),

    # Neck: FPN para combinar features multi-escala
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5
    ),

    # Head: cabeçalho do detector (ATSS)
    bbox_head=dict(
        type='ATSSHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        )
    ),

    # Configurações de Knowledge Distillation, define as loss functions usadas
    kd_cfg=dict(
        loss_cls_kd=dict(type='KDQualityFocalLoss', beta=1, loss_weight=1.0),
        loss_reg_kd=dict(type='GIoULoss', loss_weight=1.0),
        loss_center_kd=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        reused_teacher_head_idx=3
    ),

    # Configs de treino (assigner, etc)
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),

    # Configs de teste (NMS, score threshold, etc)
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100
    )
)

# Guilherme: Otimizador e wrapper
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# Guilherme: Dataloader para treino
train_dataloader = dict(batch_size=8, num_workers=4)

# Guilherme: Auto escala do learning rate
auto_scale_lr = dict(enable=True, base_batch_size=16)
