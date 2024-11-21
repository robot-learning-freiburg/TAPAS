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
from tapas_gmm.utils.logging import indent_logs
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    apply_machine_config,
    import_config_file,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig, collate
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.version import get_git_revision_hash

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
    policy: PseudoEncoderPolicyConfig

    data_naming: DataNamingConfig
    observation: ObservationConfig

    kp_selection_name: str | None = None
    copy_selection_from: str | None = None

    data_loader: DataLoaderConfig = data_loader_config
    bc_data: BCDataConfig = bc_data_config

    constant_encode_with_first_obs: bool = False

    seed: int = 0
    commit_hash: str = SET_PROGRAMMATICALLY


def encode_trajectories(
    policy: EncoderPseudoPolicy, bc_data: BCDataset, config: Config, encoder_name: str
) -> None:
    collate_func = collate  # works as we don't need padding for bs=1.

    train_loader, _ = build_data_loaders(bc_data, collate_func, config.data_loader)
    train_iterator = iter(train_loader)

    cam_names = config.observation.cameras

    assert config.kp_selection_name is not None

    using_pf = (
        config.policy.encoder_config.encoder.prior_type is PriorTypes.PARTICLE_FILTER
    )
    using_df = (
        config.policy.encoder_config.encoder.prior_type is PriorTypes.DISCRETE_FILTER
    )
    dbg = config.policy.encoder_config.debug_kp_encoding

    logger.info("Beginning encoding.")

    n_trajs = len(bc_data)

    for traj_no, batch in tqdm(enumerate(train_iterator), total=n_trajs):
        time_steps = batch.shape[1]

        if config.constant_encode_with_first_obs:
            batch = batch[:, :1, ...]
        batch = batch.to(device)

        policy.reset_episode()

        for step in tqdm(range(time_steps), leave=False):
            if not config.constant_encode_with_first_obs or step == 0:
                obs = batch[:, step, ...]

                with torch.inference_mode():
                    encoding, info = policy.encoder.encode(obs)

            bc_data.add_encoding(
                traj_no,
                step,
                None,
                "kp",
                encoding,
                encoder_name,
                config.kp_selection_name,
            )

            if using_pf and dbg:
                save_particle_debug(
                    bc_data,
                    encoder_name,
                    config.kp_selection_name,
                    cam_names,
                    traj_no,
                    step,
                    info,
                )

            elif using_df and dbg:
                save_discrete_filter_debug(
                    bc_data,
                    encoder_name,
                    config.kp_selection_name,
                    cam_names,
                    traj_no,
                    step,
                    info,
                )
            elif dbg:
                save_kp_debug(
                    bc_data,
                    encoder_name,
                    config.kp_selection_name,
                    cam_names,
                    traj_no,
                    step,
                    info,
                )


def save_kp_debug(
    bc_data: BCDataset,
    encoder_name: str,
    kp_selection_name: str,
    cam_names: tuple[str, ...],
    traj_no: int,
    obs_no: int,
    info: dict,
) -> None:
    for i, cn in enumerate(cam_names):
        bc_data.add_encoding(
            traj_no,
            obs_no,
            cn,
            "2d_locations",
            info["kp_raw_2d"][i].squeeze(0),
            encoder_name,
            kp_selection_name,
        )
        bc_data.add_encoding(
            traj_no,
            obs_no,
            cn,
            "post",
            info["post"][i].squeeze(0),
            encoder_name,
            kp_selection_name,
        )


def save_particle_debug(
    bc_data: BCDataset,
    encoder_name: str,
    kp_selection_name: str,
    cam_names: tuple[str, ...],
    traj_no: int,
    obs_no: int,
    info: dict,
) -> None:
    for i, cn in enumerate(cam_names):
        bc_data.add_encoding(
            traj_no,
            obs_no,
            cn,
            "heatmaps",
            info["particles_2d"][i].squeeze(0),
            encoder_name,
            kp_selection_name,
        )
        bc_data.add_encoding(
            traj_no,
            obs_no,
            cn,
            "2d_locations",
            info["keypoints_2d"][i].squeeze(0),
            encoder_name,
            kp_selection_name,
        )

        bc_data.add_encoding(
            traj_no,
            obs_no,
            cn,
            "diff",
            info["diff"][i].squeeze(0),
            encoder_name,
            kp_selection_name,
        )

        logger.info(f" Cam {cn}")
        logger.info(f"Descr lik: {info['descr_likelihood'][i]}")
        logger.info(f"Depth lik: {info['depth_likelihood'][i]}")
        logger.info(f"Occlu lik: {info['occlusion_likelihood'][i]}")
        logger.info(f"Descr diff: {info['diff'][i]}")

        # for j, (heatmap, kp_pos) in enumerate(zip(
        #     info["particles_2d"][i].squeeze(0),
        #     info["keypoints_2d"][i].squeeze(0))):
        #     fig, extent = figure_emb_with_points_overlay(
        #         heatmap, kp_pos, None, None, None, is_img=False,
        #         rescale=False, colors='y')

        #     bc_data.add_encoding_fig(
        #         traj_no, obs_no, cn, "heatmap_" + str(j), fig,
        #         encoder_name, kp_selection_name, bbox=extent)
        #     plt.close(fig)

    bc_data.add_encoding(
        traj_no,
        obs_no,
        None,
        "kp_world_coordinates",
        info["world_coordinates"][i].squeeze(0),
        encoder_name,
        kp_selection_name,
    )

    bc_data.add_encoding(
        traj_no,
        obs_no,
        None,
        "particle_var",
        info["particle_var"].squeeze(0),
        encoder_name,
        kp_selection_name,
    )


def save_discrete_filter_debug(
    bc_data: BCDataset,
    encoder_name: str,
    kp_selection_name: str,
    cam_names: tuple[str, ...],
    traj_no: int,
    step: int,
    info: dict,
) -> None:
    if (prior := info["prior"])[0] is not None:
        prior = (p.squeeze(0) for p in prior)
    else:
        prior = tuple(None for _ in range(len(cam_names)))

    sm = (s.squeeze(0) for s in info["sm"])
    post = (p.squeeze(0) for p in info["post"])

    for (
        cn,
        pr,
        so,
        po,
    ) in zip(cam_names, prior, sm, post):
        bc_data.add_encoding(
            traj_no, step, cn, "prior", pr, encoder_name, kp_selection_name
        )
        bc_data.add_encoding(
            traj_no, step, cn, "sm", so, encoder_name, kp_selection_name
        )
        bc_data.add_encoding(
            traj_no, step, cn, "post", po, encoder_name, kp_selection_name
        )


def main(config: Config) -> None:
    scene_data = load_scene_data(config.data_naming)
    # TODO: need to update crop for real world data here, too?
    config.policy.observation.image_dim = scene_data.image_dimensions

    encoder_checkpoint = pretrain_checkpoint_name(config.policy)
    encoder_name = encoder_checkpoint.with_suffix("").parts[-1]
    config.bc_data.encoder_name = encoder_name
    bc_data = BCDataset(scene_data, config.bc_data)

    policy = EncoderPseudoPolicy(
        config.policy,
        encoder_checkpoint=encoder_checkpoint,
        copy_selection_from=config.copy_selection_from,
    )

    with indent_logs():
        policy.initialize_parameters_via_dataset(bc_data, config.bc_data.cameras)

    # if config["policy"]["encoder"] == "keypoints" and \
    #      config["policy"]["encoder_config"]["encoder"][
    #          "prior_type"] is PriorTypes.PARTICLE_FILTER and \
    #         config["policy"]["encoder_config"]["training"].get("debug"):
    #     policy.encoder.particle_filter_viz = ParticleFilterViz()
    #     policy.encoder.particle_filter_viz.run()

    bc_data.add_encoding_config(encoder_name, config.kp_selection_name, config)

    encode_trajectories(policy, bc_data, config, encoder_name)

    enc_suffix = config.policy.encoder_suffix
    enc_suffix = "" if enc_suffix is None else enc_suffix + "-"
    config.policy.encoder_suffix = enc_suffix + config.kp_selection_name

    file_name = pretrain_checkpoint_name(config.policy)

    policy.encoder_to_disk(file_name)


def complete_config(arg: argparse.Namespace, config: DictConfig) -> DictConfig:
    if value_not_set(config.policy.encoder_naming.data_root):
        config.policy.encoder_naming.data_root = config.data_naming.data_root

    if value_not_set(config.policy.encoder_naming.task):
        config.policy.encoder_naming.task = config.data_naming.task

    if value_not_set(config.policy.encoder_naming.feedback_type):
        config.policy.encoder_naming.feedback_type = config.data_naming.feedback_type

    cams = config.policy.observation.cameras
    config.bc_data.cameras = cams
    config.bc_data.encoder_name = config.policy.encoder_name
    config.bc_data.pre_embedding = config.policy.observation.disk_read_embedding
    # TODO: disentangle the force_load/force_skip stuff.
    # For non-pre-embedding, need RGB.
    # For filters need depth, ext, int, even when pre-embedding.
    # For GT too.
    config.bc_data.force_load_raw = True

    config.kp_selection_name = config.kp_selection_name or arg.selection_name

    config.copy_selection_from = config.copy_selection_from or arg.copy_selection_from

    if config.policy.kp_pre_encoded:
        raise ValueError("kp_pre_encoded must be False for kp encoding.")

    if config.kp_selection_name is None:
        raise ValueError("kp_selection_name must be specified.")

    if (
        config.policy.encoder_config.debug_kp_encoding
        and config.policy.encoder_config.encoder.prior_type
        is PriorTypes.PARTICLE_FILTER
    ):
        config.policy.encoder_config.filter.debug = True

    config.commit_hash = get_git_revision_hash()

    return config


def entry_point():
    extra_args = (
        {
            "name": "--selection_name",
            "required": True,
            "help": "Name by which to identify the kp selection.",
        },
        {
            "name": "--copy_selection_from",
            "required": False,
            "help": "Path to an encoder from which to copy the reference positions. "
            "For GT-KP model only (to compare different projections).",
        },
    )

    args, dict_config = parse_and_build_config(extra_args=extra_args)
    dict_config = complete_config(args, dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    seed = configure_seeds(dict_config.seed)

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
