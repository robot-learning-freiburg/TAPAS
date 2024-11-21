from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from tapas_gmm.embed_trajectories import Config
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="pretrain_manual",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

config = Config(
    encoder_name="keypoints",
    encoder_config=keypoints_predictor_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    data_naming=data_naming_config,
    observation=observation_config,
)
