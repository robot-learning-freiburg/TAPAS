from tapas_gmm.encoder.keypoints import (
    EncoderConfig,
    KeypointsConfig,
    KeypointsPredictorConfig,
    PreTrainingConfig,
    PriorTypes,
    ProjectionTypes,
    ReferenceSelectionTypes,
)
from tapas_gmm.viz.keypoint_selector import (
    DisplayConfig,
    DistanceMetricConfig,
    ManualKeypointSelectorConfig,
)

keypoints_config = KeypointsConfig()

encoder_config = EncoderConfig(
    image_dim=(256, 256),
    normalize_images=True,
    use_spatial_expectation=True,
    threshold_keypoint_dist=None,
    overshadow_keypoints=False,
    descriptor_dim=64,
    projection=ProjectionTypes.UVD,
    vision_net="Resnet101_8s",
    prior_type=PriorTypes.NONE,
    taper_sm=4,
    cosine_distance=False,
    keypoints=keypoints_config,
    add_noise_scale=None,
)

ref_selector_config = ManualKeypointSelectorConfig(
    distance=DistanceMetricConfig(
        metric="cosine" if encoder_config.cosine_distance else "euclidean",
        norm_by_descr_dim=not encoder_config.cosine_distance,
        norm_multiplier=None if encoder_config.cosine_distance else 0.5,
    ),
    display=DisplayConfig(),
)

pretraining_config = PreTrainingConfig(
    ref_selection=ReferenceSelectionTypes.MANUAL,
    ref_selector=ref_selector_config,
)


keypoints_predictor_config = KeypointsPredictorConfig(
    encoder=encoder_config,
    pretraining=pretraining_config,
    filter=None,
    debug_kp_selection=False,
)
