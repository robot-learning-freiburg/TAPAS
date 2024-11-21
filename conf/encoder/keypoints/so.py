from tapas_gmm.encoder.keypoints import (
    EncoderConfig,
    KeypointsConfig,
    KeypointsPredictorConfig,
    PreTrainingConfig,
    PriorTypes,
    ProjectionTypes,
)

keypoints_config = KeypointsConfig()

pretraining_config = PreTrainingConfig()

encoder_config = EncoderConfig(
    normalize_images=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    descriptor_dim=64,
    projection=ProjectionTypes.EGO,
    vision_net="Resnet101_8s",
    prior_type=PriorTypes.PARTICLE_FILTER,
    descriptor_distance_for_outside_pixels=(1.0, 1.5),
    filter_noise_scale=0.01,
    taper_sm=4,
    cosine_distance=False,
    use_motion_model=True,
    use_gripper_motion=True,
    gripper_motion_prob=0.25,
    motion_model_noisy=True,
    motion_model_kernel=9,
    motion_model_sigma=2,
    predefined_same_objectness=True,
    sample_from_each_obs=True,
    keypoints=keypoints_config,
    return_spread=False,
    add_noise_scale=None,
    clip_projection=False,
)

keypoints_predictor_config = KeypointsPredictorConfig(
    encoder=encoder_config,
    pretraining=pretraining_config,
    debug_kp_selection=False,
    debug_filter=False,
)
