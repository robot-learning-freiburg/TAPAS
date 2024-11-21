from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.default import (
    batch_size,
    bc_data,
    bc_training,
    policy,
)
from tapas_gmm.behavior_cloning import Config, DataLoaderConfig

data_loader = DataLoaderConfig(
    train_split=1.0,
    batch_size=batch_size,
    eval_batchsize=batch_size,
    train_workers=2,
)

policy.training.lr_warmup_steps = 50

config = Config(
    policy_type="diffusion",
    policy=policy,
    training=bc_training,
    data_naming=data_naming_config,
    data_loader=data_loader,
    bc_data=bc_data,
)
