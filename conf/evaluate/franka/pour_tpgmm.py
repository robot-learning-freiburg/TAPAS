from conf._machine import data_naming_config
from conf.env.panda.handguide_banana import franka_env_config
from conf.kp_encode_trajectories.vit.franka import policy_config as vit_config
from conf.policy.models.tpgmm.auto_test import auto_tpgmm_config
from tapas_gmm.env.franka import IK_Solvers, IKConfig
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.gmm import GMMPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

franka_env_config.teleop = True
# franka_env_config.neutral_joints = (
#     -2.8432,
#     -0.5859,
#     1.4025,
#     -2.4130,
#     1.4827,
#     1.3504,
#     -1.6179,
# )
franka_env_config.neutral_joints = (
    0.034685710431713805,
    0.30546978389085466,
    -1.515821759533733,
    -2.220055133810161,
    1.2059611979222047,
    1.5495597872535385,
    -1.6113782127739653,
)
franka_env_config.eval = True
# franka_env_config.scale_action = False
franka_env_config.action_is_absolute = True
franka_env_config.postprocess_actions = False

franka_env_config.ik = IKConfig(
    solver=IK_Solvers.GN,
    max_iterations=1000,
    max_searches=50,
    tolerance=1e-4,
)

eval = EvalConfig(
    n_episodes=200,
    horizon=2500,
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
    dbg_prediction=False,
    binary_gripper_action=True,
    batch_predict_in_t_models=False,
    return_full_batch=True,
    topp_in_t_models=True,
    topp_supersampling=0.1,  # 0.005,
    predict_dx_in_xdx_models=False,
    time_based=True,
    time_scale=1.0,
    force_overwrite_checkpoint_config=True,
    pos_lag_thresh=None,  # 0.002,
    quat_lag_thresh=None,  # 0.4,
    pos_change_thresh=None,  # 0.0005,
    quat_change_thresh=None,  # 0.04,
    binary_gripper_closed_threshold=-0.8,
    encoder=vit_config,
    encoder_path="data/Pour/demos_vit_keypoints_encoder-exp.pt",
    postprocess_prediction=False,  # Predicts abs pose if False
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
