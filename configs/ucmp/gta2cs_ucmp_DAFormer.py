_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/daformer_sepaspp_mitb5.py',
    '../_base_/datasets/uda_gta_to_cityscapes_512x512.py',
    '../_base_/uda/dacs_a999_fdthings.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

# Random Seed
seed = 2  

# DAFormer Configuration
model = dict(
    type='EncoderDecoder',
    test_cfg=dict(mode='whole')  
)


data = dict(
    train=dict(
        rare_class_sampling=dict(
            min_pixels=3000,
            class_temp=0.01,
            min_crop_ratio=0.5
        ),
    ),
    workers_per_gpu=1,
    samples_per_gpu=2,
)


uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',              
    mask_pseudo_threshold='same',    
    mask_lambda=1,                  

    mask_generator=dict(
        type='block',
        mask_ratio=0.7,
        mask_block_size=32,
        _delete_=True
    ),

    legacy_scalar_pseudo_weight=False,

    uncertainty_weight=dict(
        enable=True,
        type='entropy',
        alpha=2.0,
        gamma=2.0,
        w_min=0.2,
        use_exp=True,
    ),

    mean_calibration=dict(
        enable=True,
    ),

    calib_gain=1.1,

    prototype_distill=dict(
        enable=True,
        feat_level=-1,
        min_pixels=8,
        visible_ratio_thr=0.35,
        proto_lambda=0.12,
        rel_lambda=0.0,
        use_cosine=True,
    ),

    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,

    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120
)

optimizer_config = None
optimizer = dict(
    lr=6e-05, 
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')

name = 'gta2cs_ucmp_daformer_s2'
exp = 'basic'
name_dataset = 'gta2cityscapes_512x512'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fdthings_rcs0.01-0.5_m32-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'