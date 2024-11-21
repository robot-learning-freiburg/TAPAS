from conf._machine import data_naming_config
from tapas_gmm.env import Environment
from tapas_gmm.tsdf_fusion import Config

config = Config(
    data_naming=data_naming_config,
    env=Environment.RLBENCH,
)
