from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.visualize_demo import Config

data_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="demos",
    data_root="data",
)

config = Config(
    encoder_name="demos_vit_keypoints_encoder",
    encoding_name="dbg",
    data_naming=data_naming_config,
)
