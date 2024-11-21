from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.dc import DCDataConfig, DenseCorrespondenceDataset
from tapas_gmm.encoder import KeypointsPredictor, VitFeatureEncoder

# from tapas_gmm.encoder.keypoints import KeypointsPredictorConfig
# from tapas_gmm.encoder.vit_extractor import VitFeatureEncoderConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import value_not_set
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import ObservationConfig
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.viz.live_heatmap_visualization import (
    HeatmapVisualization,
    HeatmapVisualizationConfig,
)


@dataclass
class EncoderConfig:
    encoder_name: str
    encoder_config: Any  # KeypointsPredictorConfig | VitFeatureEncoderConfig
    encoder_suffix: str | None
    encoder_naming: DataNamingConfig
    encoder_path: str | None
    vit_extr: bool


@dataclass
class Config:
    encoder: tuple[EncoderConfig, ...]

    viz: HeatmapVisualizationConfig

    data_naming: DataNamingConfig
    dc_data: DCDataConfig | None
    observation: ObservationConfig

    seed: int = 0

    # HACK
    encoder_name: Any = None
    encoder_config: Any = None
    encoder_suffix: Any = None
    encoder_naming: Any = None
    encoder_path: Any = None
    vit_extr: Any = None


def main(config: Config) -> None:
    scene_data = load_scene_data(config.data_naming)
    scene_data.update_camera_crop(config.observation.image_crop)
    config.observation.image_dim = scene_data.image_dimensions

    Encoders = [
        VitFeatureEncoder if c.vit_extr else KeypointsPredictor for c in config.encoder
    ]

    sample_type = Encoders[0].sample_type  # all should have DC sample type
    dc_data = DenseCorrespondenceDataset(
        scene_data,
        config.dc_data,
        sample_type=sample_type,
        cameras=config.observation.cameras,
    )

    file_names = tuple(
        c.encoder_path or pretrain_checkpoint_name(c) for c in config.encoder
    )

    # HACK
    encoders = []
    for Enc, enc_conf in zip(Encoders, config.encoder):
        conf = deepcopy(config)
        conf.encoder_name = enc_conf.encoder_name
        conf.encoder_config = enc_conf.encoder_config
        conf.encoder_suffix = enc_conf.encoder_suffix
        conf.encoder_naming = enc_conf.encoder_naming
        conf.encoder_path = enc_conf.encoder_path
        conf.vit_extr = enc_conf.vit_extr
        encoders.append(Enc(conf))
    encoders = tuple(encoders)

    for enc, file_name in zip(encoders, file_names):
        logger.info("Loading encoder from {}", file_name)
        enc.from_disk(file_name)

    live_heatmap = HeatmapVisualization(
        dc_data, encoders, config.viz, cams=config.observation.cameras
    )

    live_heatmap.run()


def complete_config(config: DictConfig) -> DictConfig:
    for encoder_config in config.encoder:
        if value_not_set(encoder_config.encoder_naming.data_root):
            encoder_config.encoder_naming.data_root = config.data_naming.data_root

        if value_not_set(encoder_config.encoder_naming.task):
            encoder_config.encoder_naming.task = config.data_naming.task

        if value_not_set(encoder_config.encoder_naming.feedback_type):
            encoder_config.encoder_naming.feedback_type = (
                config.data_naming.feedback_type
            )

    return config


if __name__ == "__main__":
    args, dict_config = parse_and_build_config()
    dict_config = complete_config(dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    seed = configure_seeds(dict_config.seed)

    print(OmegaConf.to_yaml(dict_config))

    main(config)  # type: ignore
