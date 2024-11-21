from dataclasses import dataclass

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.bc import BCDataConfig, BCDataset
from tapas_gmm.encoder import encoder_switch
from tapas_gmm.encoder.representation_learner import RepresentationLearner
from tapas_gmm.policy.encoder import EncoderConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.data_loading import DataLoaderConfig, build_data_loaders
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig, collate
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.version import get_git_revision_hash

bc_data_config = BCDataConfig(
    fragment_length=-1, force_load_raw=True, mask_type=MaskTypes.GT
)

data_loader_config = DataLoaderConfig(
    train_split=1.0, batch_size=1, eval_batchsize=-1, shuffle=False  # placeholder
)


@dataclass
class Config:
    encoder_name: str
    encoder_config: EncoderConfig
    encoder_suffix: str | None
    encoder_naming: DataNamingConfig

    data_naming: DataNamingConfig
    observation: ObservationConfig

    data_loader: DataLoaderConfig = data_loader_config
    bc_data: BCDataConfig = bc_data_config

    seed: int = 0

    commit_hash: str = SET_PROGRAMMATICALLY


@torch.no_grad()  # otherwise there's a memory leak somwhere. shoulf fix!
def embed_trajectories(
    encoder: RepresentationLearner,
    bc_data: BCDataset,
    config: Config,
    encoder_name: str,
) -> None:
    collate_func = collate  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(bc_data, collate_func, config.data_loader)
    train_iterator = iter(train_loader)

    cam_names = config.observation.cameras
    n_cams = len(cam_names)

    logger.info("Beginning embedding.")

    n_trajs = len(bc_data)

    for traj_no, batch in tqdm(enumerate(train_iterator), total=n_trajs):
        batch = batch.to(device)

        time_steps = batch.shape[1]

        for step in tqdm(range(time_steps), leave=False):
            obs = batch[:, step, ...]
            embedding, info = encoder.encode(obs)

            if config.encoder_name == "keypoints":
                save_descriptor(bc_data, encoder_name, cam_names, traj_no, step, info)

            else:
                save_encoding(
                    bc_data,
                    config,
                    encoder_name,
                    cam_names,
                    n_cams,
                    traj_no,
                    step,
                    embedding,
                    info,
                )


def save_encoding(
    bc_data: BCDataset,
    config: Config,
    encoder_name: str,
    cam_names: tuple[str],
    n_cams: int,
    traj_no: int,
    obs_no: int,
    embedding: torch.Tensor,
    info: dict,
) -> None:
    cam_embeddings = embedding.squeeze(0).detach().chunk(n_cams, -1)

    if config.encoder_name == "transporter":
        heatmaps = [h.squeeze(0).detach().cpu() for h in info["heatmap"]]

    for i, cn, e in zip(range(n_cams), cam_names, cam_embeddings):
        bc_data.add_embedding(traj_no, obs_no, cn, "descriptor", e, encoder_name)

        if config.encoder_name == "transporter":
            bc_data.add_embedding(
                traj_no, obs_no, cn, "heatmap", heatmaps[i], encoder_name
            )


def save_descriptor(
    bc_data: BCDataset,
    encoder_name: str,
    cam_names: tuple[str],
    traj_no: int,
    obs_no: int,
    info: dict,
) -> None:
    descriptor = (e.squeeze(0).detach() for e in info["descriptor"])

    for cn, d in zip(cam_names, descriptor):
        bc_data.add_embedding(traj_no, obs_no, cn, "descriptor", d, encoder_name)


def main(config: Config, copy_selection_from: str | None = None) -> None:
    scene_data = load_scene_data(config.data_naming)
    scene_data.update_camera_crop(config.observation.image_crop)
    config.observation.image_dim = scene_data.image_dimensions

    bc_data = BCDataset(scene_data, config.bc_data)

    Encoder = encoder_switch[config.encoder_name]
    encoder = Encoder(config).to(device)

    file_name = pretrain_checkpoint_name(config)
    encoder.from_disk(file_name)

    encoder_name = file_name.with_suffix("").parts[-1]

    if config.encoder_name == "keypoints":
        # Set ref descriptors to zero to avoid out-of-bounds erros.
        # That's easier than skipping the kp-computation for embedding.
        encoder._reference_descriptor_vec = torch.zeros_like(
            encoder._reference_descriptor_vec
        )
    else:
        # kp_gt needs dataset init
        if (ckpt := copy_selection_from) is not None:
            logger.info("Copying reference positions and descriptors from {}", ckpt)
            state_dict = torch.load(ckpt, map_location=device)
            for attr in [
                "ref_pixels_uv",
                "ref_pixel_world",
                "ref_depth",
                "ref_int",
                "ref_ext",
            ]:
                setattr(encoder, attr, state_dict[attr].to(device))
            setattr(
                encoder,
                "ref_object_poses",
                state_dict["_extra_state"]["ref_object_poses"].to(device),
            )
        else:
            encoder.initialize_parameters_via_dataset(
                bc_data, config.observation.cameras[0]
            )

    encoder.eval()

    bc_data.add_embedding_config(encoder_name, config)

    embed_trajectories(encoder, bc_data, config, encoder_name)

    # save if kp_gt model as we need the reference selection
    if config.encoder_name == "keypoints_gt":
        encoder.to_disk(file_name)


def complete_config(config: DictConfig) -> DictConfig:
    if config.encoder_naming.data_root is None:
        config.encoder_naming.data_root = config.data_naming.data_root

    if config.encoder_naming.task is None:
        config.encoder_naming.task = config.data_naming.task

    if config.encoder_naming.feedback_type is None:
        config.encoder_naming.feedback_type = config.data_naming.feedback_type

    cams = config.observation.cameras
    config.bc_data.cameras = cams
    config.bc_data.encoder_name = config.encoder_name
    config.bc_data.pre_embedding = False
    config.bc_data.force_load_raw = True

    config.commit_hash = get_git_revision_hash()

    return config


def entry_point():
    extra_args = (
        {
            "name": "--copy_selection_from",
            "required": False,
            "help": "Path to an encoder from which to copy the reference positions. "
            "For GT-KP model only (to compare different projections).",
        },
    )

    args, dict_config = parse_and_build_config(extra_args=extra_args)
    dict_config = complete_config(dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    seed = configure_seeds(dict_config.seed)

    main(config, args.copy_selection_from)  # type: ignore


if __name__ == "__main__":
    entry_point()
