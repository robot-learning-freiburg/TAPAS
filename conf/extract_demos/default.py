from conf._machine import data_naming_config
from conf.dataset.scene.maniskill import scene_dataset_config
from conf.env.maniskill.extract_demos import maniskill_env_config
from tapas_gmm.extract_demos import Config

data_naming_config.feedback_type = "demos"


config = Config(
    num_demos=10,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    vis=False,
    verbose=False,
    env_config=maniskill_env_config,
)
