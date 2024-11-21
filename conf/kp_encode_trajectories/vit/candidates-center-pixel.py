from conf._machine import data_naming_config
from conf.encoder.vit_keypoints.candidates_center_pixel import (
    vit_keypoints_predictor_config,
)
from tapas_gmm.kp_encode_trajectories import Config
from tapas_gmm.policy.encoder import PseudoEncoderPolicyConfig
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="demos",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("base",),
    image_crop=None,
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

policy_config = PseudoEncoderPolicyConfig(
    encoder_name="vit_keypoints",
    encoder_config=vit_keypoints_predictor_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    observation=observation_config,
)

config = Config(
    policy=policy_config,
    data_naming=data_naming_config,
    observation=observation_config,
)
