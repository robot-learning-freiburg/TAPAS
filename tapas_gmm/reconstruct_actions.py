import argparse

# import matplotlib.pyplot as plt
from dataclasses import dataclass

import lovely_tensors
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.bc import BCDataConfig, BCDataset
from tapas_gmm.encoder.keypoints import PriorTypes
from tapas_gmm.policy.encoder import EncoderPseudoPolicy, PseudoEncoderPolicyConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY, value_not_set
from tapas_gmm.utils.data_loading import DataLoaderConfig, build_data_loaders
from tapas_gmm.utils.franka import reconstruct_actions_obs
from tapas_gmm.utils.logging import indent_logs
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    apply_machine_config,
    import_config_file,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig, collate

# lovely_tensors.monkey_patch()

# from tapas_gmm.viz.image_single import figure_emb_with_points_overlay

bc_data_config = BCDataConfig(
    fragment_length=-1, force_load_raw=True, mask_type=MaskTypes.GT
)

data_loader_config = DataLoaderConfig(
    train_split=1.0, batch_size=1, eval_batchsize=-1, shuffle=False  # placeholder
)


@dataclass
class Config:
    data_naming: DataNamingConfig
    observation: ObservationConfig

    data_loader: DataLoaderConfig = data_loader_config
    bc_data: BCDataConfig = bc_data_config


def reconstruct_action_trajectories(
    bc_data: BCDataset,
    config: Config,
) -> None:
    collate_func = collate  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(bc_data, collate_func, config.data_loader)
    train_iterator = iter(train_loader)

    logger.info("Beginning encoding.")

    n_trajs = len(bc_data)

    for traj_no, batch in tqdm(enumerate(train_iterator), total=n_trajs):
        time_steps = batch.shape[1]

        actions = reconstruct_actions_obs(batch.squeeze(0))

        for step in tqdm(range(time_steps), leave=False):
            bc_data.update_traj_attr(
                traj_no,
                step,
                "action",
                actions[step],
            )


def main(config: Config) -> None:
    scene_data = load_scene_data(config.data_naming)
    # TODO: need to update crop for real world data here, too?
    # config.policy.observation.image_dim = scene_data.image_dimensions

    bc_data = BCDataset(scene_data, config.bc_data)

    reconstruct_action_trajectories(bc_data, config)


def complete_config(arg: argparse.Namespace, config: DictConfig) -> DictConfig:
    config.bc_data.force_load_raw = True

    return config


if __name__ == "__main__":
    args, dict_config = parse_and_build_config()
    dict_config = complete_config(args, dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore
