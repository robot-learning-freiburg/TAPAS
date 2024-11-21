from conf.encoder.vit_extractor.default import vit_model_config
from tapas_gmm.encoder.keypoints import KeypointsConfig, PriorTypes, ProjectionTypes
from tapas_gmm.encoder.vit_extractor import (
    VitKeypointsEncoderConfig,
    VitKeypointsPredictorConfig,
)
from tapas_gmm.filter.particle_filter import ParticleFilterConfig

encoder_config = VitKeypointsEncoderConfig(
    vit=vit_model_config,
    descriptor_dim=384,
    keypoints=KeypointsConfig(),
    prior_type=PriorTypes.PARTICLE_FILTER,
    projection=ProjectionTypes.EGO,
    taper_sm=25,
    cosine_distance=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    add_noise_scale=None,
)

filter_config = ParticleFilterConfig(
    descriptor_distance_for_outside_pixels=(1,),
    filter_noise_scale=0.01,
    use_gripper_motion=True,
    gripper_motion_prob=0.25,
    sample_from_each_obs=True,
    return_spread=False,
    clip_projection=False,
)

vit_keypoints_predictor_config = VitKeypointsPredictorConfig(
    encoder=encoder_config, filter=filter_config
)
