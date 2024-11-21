from copy import deepcopy
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa
import wandb
from tapas_gmm.dataset.bc import BCDataConfig, BCDataset
from tapas_gmm.encoder.keypoints import PriorTypes
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.encoder import EncoderPolicyConfig

# TODO: split policy into Policy and LearningPolicy, then import LearningPolicy and its config here.
from tapas_gmm.policy.lstm import LSTMPolicy, LSTMPolicyConfig, PolicyConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import (
    SET_PROGRAMMATICALLY,
    save_config_along_path,
    value_not_set,
)
from tapas_gmm.utils.data_loading import (
    DataLoaderConfig,
    InfiniteDataIterator,
    build_infinte_data_iterators,
)
from tapas_gmm.utils.ema import EMAModel
from tapas_gmm.utils.logging import indent_logs
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    load_scene_data,
    policy_checkpoint_name,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import ObservationConfig, collate
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.version import get_git_revision_hash
from tapas_gmm.viz.particle_filter import ParticleFilterViz

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for better CUDA debugging


@dataclass
class EMAConfig:
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    min_value: float = 0.0
    max_value: float = 0.9999


@dataclass
class TrainingConfig:
    steps: int | None
    epochs: int | None
    eval_freq: int
    save_freq: int
    auto_early_stopping: bool
    seed: int | None
    wandb_mode: str
    use_ema: bool = False
    ema: EMAConfig | None = None
    full_set_training: bool = False
    full_set_eval: bool = True


@dataclass
class Config:
    policy_type: str
    policy: PolicyConfig  # EncoderPolicyConfig

    training: TrainingConfig
    # observation: ObservationConfig

    data_naming: DataNamingConfig
    data_loader: DataLoaderConfig
    bc_data: BCDataConfig

    commit_hash: str = SET_PROGRAMMATICALLY


def run_training(
    policy: LSTMPolicy, bc_data: BCDataset, config: Config, snapshot_prefix: str
) -> None:
    if config.bc_data.fragment_length == -1:
        if config.bc_data.subsample_to_common_length:
            collate_func = partial(collate, subsample=True)
        else:
            collate_func = partial(collate, pad=True)
            raise NotImplementedError(
                "Didn't yet implement padded sequence training. Need to pack and "
                "unpack sequences. Also everything we feed into encoder? Or only "
                "image sequence as LSTM will take care of stopping gradients? See "
                "https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html"  # noqa 501
                "https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e"
            )  # noqa 501
    else:
        collate_func = collate

    train_iterator, val_iterator = build_infinte_data_iterators(
        bc_data,
        collate_func,
        config.data_loader,
        full_set_training=config.training.full_set_training,
        full_set_eval=config.training.full_set_eval,
    )

    # TODO: init_params per train step to enable multi-task learning?
    with indent_logs():
        policy.initialize_parameters_via_dataset(
            bc_data, config.bc_data.cameras, epoch_length=train_iterator.epoch_length
        )

    wandb.watch(policy, log_freq=100)

    policy.train()

    current_loss = np.inf

    eval_freq = config.training.eval_freq
    save_freq = config.training.save_freq
    early_stopping = config.training.auto_early_stopping
    n_steps = config.training.steps or config.training.epochs
    epoch_based_training = config.training.full_set_training

    logger.info("Beginning training.")

    # ema_policy = deepcopy(policy) if config.training.use_ema else None
    ema = (
        EMAModel(model=policy, **config.training.ema.__dict__)
        if config.training.use_ema
        else None
    )

    total_steps = 0

    try:
        for i in tqdm(range(1, n_steps + 1)):
            if epoch_based_training:
                while True:
                    try:
                        batch = next(train_iterator)
                    except StopIteration:
                        break
                    training_metrics = policy.update_params(batch)
                    # print(f"{total_steps}: {training_metrics}")
                    wandb.log(training_metrics, step=total_steps)
                    if ema:
                        ema.step(policy)
                    total_steps += 1

            else:
                batch = batch = next(train_iterator)
                training_metrics = policy.update_params(batch)
                wandb.log(training_metrics, step=total_steps)

                if ema:
                    ema.step(policy)

                total_steps += 1

            if i % eval_freq == 0 and val_iterator is not None:
                last_loss = current_loss

                eval_metrics = run_eval_epoch(policy, val_iterator)

                wandb.log(eval_metrics, step=total_steps)

                # logger.info(f"Val loss {eval_metrics['eval-loss']}")

                if ema:
                    ema_eval_metrics = run_eval_epoch(ema.averaged_model, val_iterator)

                    ema_eval_metrics = {
                        "ema_" + k: v for k, v in ema_eval_metrics.items()
                    }

                    wandb.log(ema_eval_metrics, step=total_steps)

                    # logger.info(f"Ema val loss {ema_eval_metrics['ema_eval-loss']}")

                current_loss = eval_metrics["eval-loss"].cpu().numpy()

                if current_loss > last_loss and early_stopping:
                    logger.info("Started to overfit. Interrupting training.")
                    break

            if save_freq and (i % save_freq == 0):
                file_name = snapshot_prefix + "_step_" + str(i) + ".pt"
                logger.info("Saving intermediate policy:", file_name)
                with indent_logs():
                    policy.to_disk(file_name)
                    if ema:
                        ema_file_name = snapshot_prefix + "_step_" + str(i) + "_ema.pt"
                        ema.averaged_model.to_disk(ema_file_name)

    except KeyboardInterrupt:
        logger.info("Interrupted training. Saving current model checkpoint.")


def run_eval_epoch(policy: LSTMPolicy, val_iterator: InfiniteDataIterator) -> dict:
    policy.eval()

    with torch.no_grad():
        eval_metrics = []

        # for batch in val_iterator:
        while True:
            try:
                batch = next(val_iterator)
            except StopIteration:
                break
            else:
                eval_metrics.append(policy.evaluate(batch.to(device)))

        eval_metrics = {
            k: torch.cat([d[k].unsqueeze(0) for d in eval_metrics]).mean()
            for k in eval_metrics[0]
        }

    policy.train()

    return eval_metrics


def main(config: Config) -> None:
    scene_data = load_scene_data(config.data_naming)

    is_encoder_policy = config.policy.encoder_name is not None

    # if type(config.policy) is EncoderPolicyConfig:
    if is_encoder_policy:
        scene_data.update_camera_crop(config.policy.observation.image_crop)
        config.policy.observation.image_dim = scene_data.image_dimensions

        encoder_checkpoint = pretrain_checkpoint_name(config.policy)
        encoder_name = encoder_checkpoint.with_suffix("").parts[-1]
    else:
        encoder_name = None

    config.bc_data.encoder_name = encoder_name
    bc_data = BCDataset(scene_data, config.bc_data)

    disk_read = config.bc_data.pre_embedding or (
        config.bc_data.kp_pre_encoding is not None
    )

    Policy = import_policy(config.policy_type, disk_read=disk_read)

    # In kp_encoder_trajectories, we append the kp selection name to the
    # encoder checkpoint. Need to do here AFTER instantiating the replay buffer
    if kp_selection_name := config.bc_data.kp_pre_encoding:
        enc_suffix = config.policy.encoder_suffix
        enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
        config.policy.encoder_suffix = enc_suffix + kp_selection_name
    if is_encoder_policy:
        encoder_checkpoint = pretrain_checkpoint_name(config.policy)
    else:
        encoder_checkpoint = None

    policy = Policy(config.policy, encoder_checkpoint=encoder_checkpoint).to(device)

    file_name, suffix = policy_checkpoint_name(
        config, create_suffix=config.policy.suffix is None
    )  # type: ignore

    wandb.log({"suffix": suffix}, step=0)

    snapshot_prefix = str(file_name.with_suffix(""))

    # TODO: this is quite hacky. Make more elegant.
    if (
        type(config.policy) is EncoderPolicyConfig
        and config.policy.encoder_name == "keypoints"
        and config.policy.encoder_config.encoder.prior_type
        is PriorTypes.PARTICLE_FILTER
        and config.policy.encoder_config.debug_filter
    ):
        policy.encoder.particle_filter_viz = ParticleFilterViz()
        policy.encoder.particle_filter_viz.run()

    run_training(policy, bc_data, config, snapshot_prefix)

    policy.to_disk(file_name)
    save_config_along_path(config, file_name)


# TODO: remove most of this -> should be done directly in config, eg using
# variables that are reused. makes it less confusing.
# Only do some validation here?
def complete_config(config: DictConfig) -> DictConfig:
    assert (config.training.steps is None) ^ (config.training.epochs is None)
    assert (config.training.steps is None) == (config.training.full_set_training)

    # is_encoder_policy = type(config.policy) is EncoderPolicyConfig
    # is_encoder_policy = config.policy.encoder_name is not None
    is_encoder_policy = config.policy.obs_encoder.image_encoder is not None

    if is_encoder_policy:
        if value_not_set(config.policy.encoder_naming.data_root):
            config.policy.encoder_naming.data_root = config.data_naming.data_root

        if value_not_set(config.policy.encoder_naming.task):
            config.policy.encoder_naming.task = config.data_naming.task

        if value_not_set(config.policy.encoder_naming.feedback_type):
            config.policy.encoder_naming.feedback_type = (
                config.data_naming.feedback_type
            )

        cams = config.policy.observation.cameras
        config.bc_data.cameras = cams
        config.bc_data.encoder_name = config.policy.encoder_name
        config.bc_data.pre_embedding = config.policy.observation.disk_read_embedding
        # TODO: disentangle the force_load/force_skip stuff.
        # For non-pre-embedding, need RGB.
        # For filters need depth, ext, int, even when pre-embedding.
        # For GT too.
        config.bc_data.force_load_raw = True

        if (
            config.policy.observation.disk_read_keypoints
            and config.bc_data.kp_pre_encoding is None
        ):
            raise ValueError(
                "disk_read_keypoints is True but no kp_pre_encoding "
                "was specified. Can specify via "
                "--overwrite bc_data.kp_pre_encoding=<name>"
            )
    else:
        config.bc_data.cameras = tuple()
        config.bc_data.encoder_name = None
        config.bc_data.pre_embedding = False

    config.commit_hash = get_git_revision_hash()

    return config


def entry_point():
    args, dict_config = parse_and_build_config()
    dict_config = complete_config(dict_config)  # type: ignore

    seed = configure_seeds(dict_config.training.seed)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    wandb.init(
        config=OmegaConf.to_container(dict_config, resolve=True),  # type: ignore
        project="bask_bc",
        mode=config.training.wandb_mode,
    )  # type: ignore

    # print(OmegaConf.to_yaml(dict_config))

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
