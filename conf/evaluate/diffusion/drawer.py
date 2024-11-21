from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.drawer import policy
from conf.env.rlbench.lstm_eval import rlbench_env_config
from conf.evaluate.lstm.rlbech_headless import eval
from tapas_gmm.evaluate import Config

rlbench_env_config.absolute_action_mode = True
# rlbench_env_config.planning_action_mode=True
rlbench_env_config.action_frame = "world"

eval.horizon = int(700 / policy.n_action_steps)

config = Config(
    env_config=rlbench_env_config,
    eval=eval,
    policy=policy,
    policy_type="diffusion",
    data_naming=data_naming_config,
)
