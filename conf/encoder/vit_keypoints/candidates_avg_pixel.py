from conf.encoder.vit_extractor.default import vit_model_config
from tapas_gmm.encoder.keypoints import KeypointsConfig, PriorTypes, ProjectionTypes
from tapas_gmm.encoder.vit_extractor import (
    PreTrainingConfig,
    ReferenceSelectionTypes,
    VitKeypointsEncoderConfig,
    VitKeypointsPredictorConfig,
)

obj_labels = tuple((13, 14, 15, 16, 17))
robot_labels = tuple(range(1, 12))

kp_config = KeypointsConfig(
    n_sample=len(obj_labels),
)

pretraining_config = PreTrainingConfig(
    ref_selection=ReferenceSelectionTypes.MASK_AVG,
    ref_labels=obj_labels,
    ref_traj_idx=9,
    ref_obs_idx=0,
)

encoder_config = VitKeypointsEncoderConfig(
    vit=vit_model_config,
    descriptor_dim=384,
    keypoints=kp_config,
    prior_type=PriorTypes.NONE,
    projection=ProjectionTypes.NONE,
    taper_sm=25,
    cosine_distance=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    add_noise_scale=None,
)

# TODO: verify center/avg sampling

vit_keypoints_predictor_config = VitKeypointsPredictorConfig(
    encoder=encoder_config,
    pretraining=pretraining_config,
    debug_kp_selection=False,
)
