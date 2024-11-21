from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.default import bc_data, bc_training
from conf.behavior_cloning.diffusion.five_demos.default import data_loader
from conf.behavior_cloning.diffusion.sweep import policy
from tapas_gmm.behavior_cloning import Config

policy.training.lr_warmup_steps = 50

config = Config(
    policy_type="diffusion",
    policy=policy,
    training=bc_training,
    data_naming=data_naming_config,
    data_loader=data_loader,
    bc_data=bc_data,
)
