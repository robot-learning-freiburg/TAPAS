from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.env.maniskill.gmm import maniskill_env_config
from conf.policy.models.tpgmm.test import tpgmm_config
from tapas_gmm.evaluate import Config, EvalConfig
from tapas_gmm.policy.gmm import GMMPolicyConfig
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

policy_config = GMMPolicyConfig(
    suffix=None,
    model=tpgmm_config,
)


config = Config(
    env_config=maniskill_env_config,
    eval=eval,
    policy=policy_config,
    data_naming=data_naming_config,
    policy_type="gmm",
)
