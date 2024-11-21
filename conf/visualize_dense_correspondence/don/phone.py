from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig
from tapas_gmm.visualize_dense_correspondence import Config, EncoderConfig
from tapas_gmm.viz.live_heatmap_visualization import (
    DisplayConfig,
    DistanceMetricConfig,
    HeatmapVisualizationConfig,
    SamplingConfig,
)

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="pretrain_manual",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
)

display_config = DisplayConfig(
    taper=4,
    normalize=False,
)

sampling_config = SamplingConfig()

distance_metric_config = DistanceMetricConfig(
    norm_by_descr_dim=False,
    norm_multiplier=None,
    metric="cosine",
)

viz_config = HeatmapVisualizationConfig(
    distance=(distance_metric_config,),
    display=display_config,
    sampling=sampling_config,
)

encoder_config = EncoderConfig(
    encoder_name="keypoints",
    encoder_config=keypoints_predictor_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    encoder_path=None,
    vit_extr=False,
)

config = Config(
    encoder=(encoder_config,),
    viz=viz_config,
    data_naming=data_naming_config,
    dc_data=None,
    observation=observation_config,
)
