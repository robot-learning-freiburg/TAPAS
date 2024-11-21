from conf._machine import data_naming_config
from tapas_gmm.summarize_encoding import Config
from tapas_gmm.utils.observation import ObservationConfig

observation_config = ObservationConfig(
    cameras=("base",),
    image_crop=None,
)

config = Config(
    data_naming=data_naming_config,
    observation=observation_config,
    show_kp=True,
    annotate_kp=True,
    show_heatmap=False,
    show_3d_traj=False,
    d_kp=None,
    strict_shape_checking=False,
    index_step=10,
    n_cols=6,
)
