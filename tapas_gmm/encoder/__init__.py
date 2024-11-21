from loguru import logger

from tapas_gmm.encoder.bvae import BVAE
from tapas_gmm.encoder.cnn import CNN, CNNDepth
from tapas_gmm.encoder.keypoints import KeypointsPredictor, KeypointsPredictorConfig
from tapas_gmm.encoder.keypoints_gt import (
    GTKeypointsPredictor,
    GTKeypointsPredictorConfig,
)
from tapas_gmm.encoder.monet import Monet
from tapas_gmm.encoder.representation_learner import (
    RepresentationLearner,
    RepresentationLearnerConfig,
)
from tapas_gmm.encoder.transporter import Transporter
from tapas_gmm.encoder.vit_extractor import (
    VitFeatureEncoder,
    VitFeatureEncoderConfig,
    VitKeypointsPredictor,
    VitKeypointsPredictorConfig,
)

encoder_switch = {
    "transporter": Transporter,
    "bvae": BVAE,
    "monet": Monet,
    "keypoints": KeypointsPredictor,
    "keypoints_gt": GTKeypointsPredictor,
    "cnn": CNN,
    "cnnd": CNNDepth,
    "vit_extractor": VitFeatureEncoder,
    "vit_keypoints": VitKeypointsPredictor,
}

encoder_names = list(encoder_switch.keys())


# TODO: add the remaining encoders. Some (eg CNN) do not have a specific config yet.
encoder_config_map = {
    KeypointsPredictorConfig: KeypointsPredictor,
    GTKeypointsPredictorConfig: GTKeypointsPredictor,
    VitFeatureEncoderConfig: VitFeatureEncoder,
    VitKeypointsPredictorConfig: VitKeypointsPredictor,
}


def get_image_encoder_class(encoder_config: RepresentationLearnerConfig | None):
    """
    Get the image encoder class from the encoder config.

    Parameters
    ----------
    encoder_name : RepresentationLearnerConfig
        The encoder config.

    Returns
    -------
    Encoder
        The encoder class.
    """
    if encoder_config is None:
        logger.info("No encoder config provided. Using None.")
        Encoder = None
    else:
        Encoder = encoder_config_map.get(type(encoder_config), None)

        if Encoder is None:
            raise ValueError(f"Encoder for config {encoder_config} not found.")

    return Encoder
