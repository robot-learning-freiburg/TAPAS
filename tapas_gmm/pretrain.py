# import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

# import config as default_config
import tapas_gmm.utils.logging  # noqa
import wandb
from tapas_gmm.dataset.dc import DCDataConfig, DenseCorrespondenceDataset
from tapas_gmm.encoder import encoder_switch
from tapas_gmm.encoder.representation_learner import RepresentationLearner
from tapas_gmm.policy.encoder import EncoderConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY, save_config_along_path
from tapas_gmm.utils.data_loading import (
    DataLoaderConfig,
    InfiniteDataIterator,
    build_data_iterators,
)
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import ObservationConfig
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.version import get_git_revision_hash

# warnings.filterwarnings("error")  # For getting stack trace on warnings.


# TODO:
# restructurig of conf for pretrained-on, pretrain-feedback


@dataclass
class TrainingConfig:
    steps: int
    eval_freq: int
    save_freq: int
    auto_early_stopping: bool
    seed: int
    wandb_mode: str  # "disabled", "online", "offline"
    save_model: bool = True


@dataclass
class Config:
    encoder_name: str
    encoder_config: EncoderConfig
    encoder_suffix: str | None

    training: TrainingConfig
    observation: ObservationConfig

    data_naming: DataNamingConfig
    dataloader: DataLoaderConfig
    dc_data: DCDataConfig | None

    encoder_naming: DataNamingConfig = MISSING

    commit_hash: str = SET_PROGRAMMATICALLY


def run_training(
    encoder: RepresentationLearner,
    dc_data: DenseCorrespondenceDataset,
    config: Config,
    snapshot_prefix: Path,
) -> None:
    encoder.train()
    current_loss = np.inf

    collate_func = dc_data.get_collate_func()

    train_iterator, val_iterator = build_data_iterators(
        dc_data, collate_func, config.dataloader
    )

    obs_total = len(dc_data)
    train_frac = config.dataloader.train_split
    no_train_obs = int(train_frac * obs_total)
    no_eval_obs = int((1 - train_frac) * obs_total)

    save_freq = config.training.save_freq
    eval_freq = config.training.eval_freq
    early_stopping = config.training.auto_early_stopping
    n_steps = config.training.steps

    try:
        for i in tqdm(range(n_steps)):
            training_metrics = run_training_step(
                encoder, config, train_iterator, no_train_obs
            )

            wandb.log(training_metrics, step=i)

            if i % eval_freq == 0 and val_iterator is not None:
                last_loss = current_loss

                eval_metrics = run_eval_step(encoder, config, val_iterator, no_eval_obs)

                wandb.log(eval_metrics, step=i)

                current_loss = eval_metrics["eval-loss"].cpu().numpy()

                if current_loss > last_loss and early_stopping:
                    logger.info("Started to overfit. Interrupting training.")
                    break

            if save_freq and (i % save_freq == 0):
                file_name = (
                    snapshot_prefix.parent / (snapshot_prefix.name + "_step_" + str(i))
                ).with_suffix(".pt")

                logger.info("Saving intermediate encoder at {}", file_name)
                encoder.to_disk(file_name)

    except KeyboardInterrupt:
        logger.info("Interrupted training.")


def run_training_step(
    encoder: RepresentationLearner,
    config: Config,
    train_iterator: InfiniteDataIterator,
    no_train_obs: int,
) -> dict:
    while True:  # try until non-empty correspondence batch
        batch = next(train_iterator)  # type: ignore

        training_metrics = encoder.update_params(
            batch, dataset_size=no_train_obs, batch_size=config.dataloader.batch_size
        )

        if training_metrics is not None:
            break

    return training_metrics


def run_eval_step(
    encoder: RepresentationLearner,
    config: Config,
    val_iterator: InfiniteDataIterator,
    no_eval_obs: int,
) -> dict:
    encoder.eval()

    with torch.no_grad():
        while True:
            batch = next(val_iterator)  # type: ignore

            eval_metrics = encoder.evaluate(
                batch,
                dataset_size=no_eval_obs,
                batch_size=config.dataloader.eval_batchsize,
            )

            if eval_metrics is not None:
                break

    encoder.train()

    return eval_metrics


def main(config: Config) -> None:
    scene_data = load_scene_data(config.data_naming)
    scene_data.update_camera_crop(config.observation.image_crop)
    config.observation.image_dim = scene_data.image_dimensions

    Encoder = encoder_switch[config.encoder_name]
    encoder = Encoder(config).to(device)

    wandb.watch(encoder, log_freq=100)

    dc_data = DenseCorrespondenceDataset(
        scene_data,
        config.dc_data,
        sample_type=encoder.sample_type,
        cameras=config.observation.cameras,
    )

    encoder.initialize_image_normalization(dc_data, config.observation.cameras)

    if config.dc_data and config.dc_data.contrast_set:
        raise NotImplementedError(
            "Need to update contrast set loading. " "See release-branche for old code."
        )

    file_name = pretrain_checkpoint_name(config)  # type: ignore
    snapshot_prefix = file_name.with_suffix("")
    logger.info("Using snapshot prefix " + str(snapshot_prefix))

    run_training(encoder, dc_data, config, snapshot_prefix)

    if config.training.save_model:
        logger.info("Saving current model checkpoint.")
        encoder.to_disk(file_name)

        save_config_along_path(config, file_name)


def complete_config(config: DictConfig) -> DictConfig:
    if config.encoder_config.end_to_end:
        raise ValueError("Passed end_to_end=True to pretraining script.")

    config.encoder_naming = config.data_naming

    config.commit_hash = get_git_revision_hash()

    return config


def entry_point():
    _, dict_config = parse_and_build_config()
    dict_config = complete_config(dict_config)

    seed = configure_seeds(dict_config.training.seed)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    wandb.init(
        config=OmegaConf.to_container(dict_config, resolve=True),  # type: ignore
        project="bask_pretrain",
        mode=config.training.wandb_mode,
    )  # type: ignore

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
