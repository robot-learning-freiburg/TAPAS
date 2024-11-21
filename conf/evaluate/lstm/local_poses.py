from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.env.rlbench.lstm_eval import rlbench_env_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.lstm import LSTMPolicyConfig
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
    feedback_type="demos",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

policy_config = LSTMPolicyConfig(
    suffix=None,
    visual_embedding_dim=7,  # Set by encoder
    proprio_dim=7,
    action_dim=7,
    lstm_layers=2,
    use_ee_pose=True,
    add_gripper_state=False,
    add_object_poses=True,
    training=None,
    action_scaling=True,
    poses_in_ee_frame=False,
)

config = Config(
    env_config=rlbench_env_config,
    eval=eval,
    policy=policy_config,
    policy_type="lstm",
    data_naming=data_naming_config,
)
