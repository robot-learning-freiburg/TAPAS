from tapas_gmm.encoder.keypoints import (
    EncoderConfig,
    KeypointsConfig,
    KeypointsPredictorConfig,
    PreTrainingConfig,
    PriorTypes,
    ProjectionTypes,
    ReferenceSelectionTypes,
)
from tapas_gmm.filter.particle_filter import ParticleFilterConfig

keypoints_config = KeypointsConfig()

pretraining_config = PreTrainingConfig(
    ref_selection=ReferenceSelectionTypes.MANUAL,
)

encoder_config = EncoderConfig(
    normalize_images=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    descriptor_dim=64,
    projection=ProjectionTypes.EGO,
    vision_net="Resnet101_8s",
    prior_type=PriorTypes.PARTICLE_FILTER,
    taper_sm=4,
    cosine_distance=False,
    keypoints=keypoints_config,
    add_noise_scale=None,
)

filter_config = ParticleFilterConfig(
    descriptor_distance_for_outside_pixels=(1.0, 1.5),
    filter_noise_scale=0.01,
    use_gripper_motion=True,
    gripper_motion_prob=0.25,
    sample_from_each_obs=True,
    return_spread=False,
    clip_projection=False,
)


keypoints_predictor_config = KeypointsPredictorConfig(
    encoder=encoder_config,
    pretraining=pretraining_config,
    filter=filter_config,
    debug_kp_selection=False,
)
