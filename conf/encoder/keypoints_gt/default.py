from tapas_gmm.encoder.keypoints import KeypointsConfig, KeypointsTypes, ProjectionTypes
from tapas_gmm.encoder.keypoints_gt import (
    EncoderConfig,
    GTKeypointsPredictorConfig,
    PreTrainingConfig,
    ReferenceSelectionTypes,
)
from tapas_gmm.viz.keypoint_selector import (
    DisplayConfig,
    DistanceMetricConfig,
    ManualKeypointSelectorConfig,
)

encoder_config = EncoderConfig(
    debug=False,
    descriptor_dim=3,  # DUMMY value
    projection=ProjectionTypes.EGO,
    keypoints=KeypointsConfig(
        type=KeypointsTypes.SD,
        n_sample=16,  # larger when using WSD, eg. 100
        n_keep=5,  # only for SDS, WSDS
    ),
)

ref_selector_config = ManualKeypointSelectorConfig(  # DUMMY values
    distance=DistanceMetricConfig(
        metric="cosine",
        norm_by_descr_dim=False,
        norm_multiplier=None,
    ),
    display=DisplayConfig(),
)

pretraining_config = PreTrainingConfig(
    ref_selection=ReferenceSelectionTypes.MANUAL,
    ref_selector=ref_selector_config,
)

keypoints_gt_config = GTKeypointsPredictorConfig(
    pretraining=pretraining_config,
    encoder=encoder_config,
)
