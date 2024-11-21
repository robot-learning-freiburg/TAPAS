from conf._machine import data_naming_config
from conf.behavior_cloning.keypoints.default import (
    dataloader_config,
    encoder_naming_config,
    observation_config,
    policy_training_config,
    training_config,
)
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.observation.wrist_nocrop import observation_config
from tapas_gmm.behavior_cloning import Config
from tapas_gmm.dataset.bc import BCDataConfig
from tapas_gmm.policy.encoder import EncoderPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.observation import MaskTypes

policy_config = EncoderPolicyConfig(
    encoder_name="keypoints",
    encoder_config=keypoints_predictor_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    observation=observation_config,
    kp_pre_encoded=True,
    suffix=None,
    visual_embedding_dim=SET_PROGRAMMATICALLY,  # Set by encoder
    proprio_dim=7,
    action_dim=7,
    lstm_layers=2,
    use_ee_pose=True,
    add_gripper_state=False,
    add_object_poses=False,
    training=policy_training_config,
)

bc_data_config = BCDataConfig(
    fragment_length=30,
    pre_embedding=False,
    kp_pre_encoding=None,
    only_use_labels=(13,),
    mask_type=MaskTypes.GT,
)

config = Config(
    policy_type="encoder",
    policy=policy_config,
    training=training_config,
    # observation=observation_config,
    data_naming=data_naming_config,
    data_loader=dataloader_config,
    bc_data=bc_data_config,
)
