from conf._machine import data_naming_config

# from conf.kp_encode_trajectories.vit.franka import policy_config as vit_config
from conf.encoder.vit_keypoints.nofilter import (
    vit_keypoints_predictor_config as vit_config,
)
from conf.env.panda.handguide_coffee import franka_env_config
from conf.policy.models.tpgmm.auto_test import auto_tpgmm_config
from tapas_gmm.encoder.encoder import ObservationEncoderConfig
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy import PolicyEnum
from tapas_gmm.policy.gmm import GMMPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

franka_env_config.teleop = True
franka_env_config.eval = True
# franka_env_config.scale_action = False
franka_env_config.action_is_absolute = True
franka_env_config.postprocess_actions = False
franka_env_config.dynamic_camera_visualization = True

eval = EvalConfig(
    n_episodes=200,
    horizon=3000,
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

obs_encoder = ObservationEncoderConfig(
    image_encoder=vit_config,
)

policy_config = GMMPolicyConfig(
    suffix=None,
    model=auto_tpgmm_config,
    dbg_prediction=False,
    binary_gripper_action=True,
    batch_predict_in_t_models=True,
    topp_in_t_models=True,
    topp_supersampling=0.05,
    return_full_batch=True,
    predict_dx_in_xdx_models=False,
    time_based=True,
    time_scale=1.0,
    force_overwrite_checkpoint_config=True,
    pos_lag_thresh=None,  # 0.01,
    quat_lag_thresh=None,  # 0.4,
    pos_change_thresh=None,  # 0.002,
    quat_change_thresh=None,  # 0.04,
    binary_gripper_closed_threshold=0,
    encoder=vit_config,
    encoder_path="data/Coffee/demos_vit_keypoints_encoder-exp.pt",
    postprocess_prediction=False,  # Predicts abs pose if False
    obs_encoder=obs_encoder,
    # topp_in_t_models=True,
    # batch_t_max=1.25,
    # pos_change_thresh=None,
    # quat_change_thresh=None,
    # pos_lag_thresh=None,
    # quat_lag_thresh=None,
)


config = Config(
    env=franka_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
    policy_type="gmm",
)
