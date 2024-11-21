from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.env.rlbench.headless import rlbench_env_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.encoder import EncoderPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

eval = EvalConfig(
    n_episodes=200,
    seed=1,
    obs_dropout=None,
    viz=False,
    kp_per_channel_viz=False,
    show_channels=None,
)

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="pretrain_manual",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

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
    training=None,
)


config = Config(
    env_config=rlbench_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
)
