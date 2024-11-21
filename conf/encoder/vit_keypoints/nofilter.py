from conf.encoder.vit_extractor.default import vit_model_config
from tapas_gmm.encoder.keypoints import (
    KeypointsConfig,
    PriorTypes,
    ProjectionTypes,
    ReferenceSelectionTypes,
)
from tapas_gmm.encoder.vit_extractor import (
    PreTrainingConfig,
    VitKeypointsEncoderConfig,
    VitKeypointsPredictorConfig,
)
from tapas_gmm.viz.keypoint_selector import (
    DisplayConfig,
    DistanceMetricConfig,
    ManualKeypointSelectorConfig,
)

keypoints_conf = KeypointsConfig(
    n_sample=5,
)

encoder_config = VitKeypointsEncoderConfig(
    vit=vit_model_config,
    descriptor_dim=384,
    keypoints=keypoints_conf,
    prior_type=PriorTypes.NONE,
    projection=ProjectionTypes.GLOBAL_HARD,
    taper_sm=25,
    cosine_distance=True,
    use_spatial_expectation=False,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    add_noise_scale=None,
)

ref_selector_config = ManualKeypointSelectorConfig(
    distance=DistanceMetricConfig(
        metric="cosine" if encoder_config.cosine_distance else "euclidean",
        norm_by_descr_dim=not encoder_config.cosine_distance,
        norm_multiplier=None if encoder_config.cosine_distance else 20,
    ),
    display=DisplayConfig(taper=encoder_config.taper_sm),
)

pretraining_config = PreTrainingConfig(
    ref_selection=ReferenceSelectionTypes.MANUAL,
    ref_selector=ref_selector_config,
)

vit_keypoints_predictor_config = VitKeypointsPredictorConfig(
    encoder=encoder_config,
    image_dim=(480, 640),
    pretraining=pretraining_config,
    debug_kp_selection=True,
    debug_kp_encoding=True,
)
