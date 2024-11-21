from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.env.rlbench.gmm import rlbench_env_config
from conf.policy.models.tpgmm.auto_tgrip_wine import auto_tpgmm_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.gmm import GMMPolicyConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

eval = EvalConfig(
    n_episodes=200,
    horizon=700,
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

policy_config = GMMPolicyConfig(
    suffix=None,
    model=auto_tpgmm_config,
    predict_dx_in_xdx_models=True,
    binary_gripper_action=True,
    binary_gripper_closed_threshold=0.95,
    dbg_prediction=False,
    # the kinematics model in RLBench is just to unreliable -> leads to mistakes
    topp_in_t_models=False,
    topp_supersampling=0.025,
    # time_scale=0.25,
)


config = Config(
    env_config=rlbench_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
    policy_type="gmm",
)
