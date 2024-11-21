from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.env.panda.handguide import franka_env_config
from conf.kp_encode_trajectories.vit.franka import policy_config as vit_config
from conf.policy.models.tpgmm.auto_test import auto_tpgmm_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.gmm import GMMPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

franka_env_config.teleop = True
franka_env_config.eval = True

eval = EvalConfig(
    n_episodes=200,
    horizon=500,
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

policy_config = GMMPolicyConfig(
    suffix=None,
    model=auto_tpgmm_config,
    dbg_prediction=True,
    binary_gripper_action=True,
    batch_predict_in_t_models=True,
    topp_in_t_models=False,
    predict_dx_in_xdx_models=False,
    time_based=True,
    force_overwrite_checkpoint_config=True,
    pos_lag_thresh=0.01,
    quat_lag_thresh=0.4,
    pos_change_thresh=0.002,
    quat_change_thresh=0.04,
    binary_gripper_closed_threshold=-0.5,
    encoder=vit_config,
    encoder_path="data/Banana/demos_vit_keypoints_encoder-test.pt",
    # topp_in_t_models=True,
    # batch_t_max=1.25,
    # pos_change_thresh=None,
    # quat_change_thresh=None,
    # pos_lag_thresh=None,
    # quat_lag_thresh=None,
)


config = Config(
    env_config=franka_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
    policy_type="gmm",
)
