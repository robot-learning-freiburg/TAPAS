from conf._machine import data_naming_config
from conf.encoder.vit_extractor.default import (
    vit_feature_encoder_config as stride8_config,
)
from conf.encoder.vit_extractor.stride4 import (
    vit_feature_encoder_config as stride4_config,
)
from conf.visualize_dense_correspondence.vit.dinov1 import (
    distance_metric_config as v1_config,
)
from conf.visualize_dense_correspondence.vit.dinov2 import (
    distance_metric_config as v2_config,
)
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig
from tapas_gmm.visualize_dense_correspondence import Config, EncoderConfig
from tapas_gmm.viz.live_heatmap_visualization import (
    DisplayConfig,
    HeatmapVisualizationConfig,
    SamplingConfig,
)

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="pretrain_manual",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("base",),
    image_crop=None,
)

display_config = DisplayConfig()

sampling_config = SamplingConfig()

viz_config = HeatmapVisualizationConfig(
    distance=(v1_config, v2_config),
    display=display_config,
    sampling=sampling_config,
)

encoder_config = EncoderConfig(
    encoder_name="vit_extractor",
    encoder_config=stride8_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    encoder_path=None,
    vit_extr=True,
)

encoder_config_v2 = EncoderConfig(
    encoder_name="vit_extractor",
    encoder_config=stride4_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    encoder_path=None,
    vit_extr=True,
)

config = Config(
    encoder=(encoder_config, encoder_config_v2),
    viz=viz_config,
    data_naming=data_naming_config,
    dc_data=None,
    observation=observation_config,
)
