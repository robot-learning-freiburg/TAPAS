import argparse
from dataclasses import dataclass

import lovely_tensors
import matplotlib.pyplot as plt
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.geometry_torch import quaternion_to_matrix
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    apply_machine_config,
    import_config_file,
    load_scene_data,
    pretrain_checkpoint_name,
)
from tapas_gmm.viz.operations import channel_front2back
from tapas_gmm.viz.threed import plot_coordindate_frame

# lovely_tensors.monkey_patch()


@dataclass
class Config:
    encoder_name: str
    encoding_name: str

    data_naming: DataNamingConfig

    kp_selection_name: str | None = None
    copy_selection_from: str | None = None

    traj_idx: int = 0
    camera: str = "wrist"


def main(config):
    scene_data = load_scene_data(config.data_naming)

    encoding_attrs = {}

    for attr in ("2d_locations", "post"):
        c_attr = scene_data._get_cam_attr_name(config.camera, attr)
        encoding_attrs[c_attr] = scene_data._get_cam_attr_name(config.camera, attr)
    encoding_attrs["kp"] = "kp"

    batch = scene_data._get_bc_traj(
        traj_idx=config.traj_idx,  # type: ignore
        cams=(config.camera,),
        encoder_name=config.encoder_name,
        encoding_name=config.encoding_name,
        encoding_attr=encoding_attrs,
        mask_type=None,
        skip_rgb=False,
        skip_generic_attributes=False,
        as_tensor_dict=True,
    )

    ee_traj = batch["ee_pose"][:, :3]

    first_obs = batch[0]

    cam_obs = first_obs["cameras"][config.camera]

    rgb = channel_front2back(cam_obs.rgb)
    depth = cam_obs.depth
    extr = cam_obs.extr

    ee_pose = first_obs["ee_pose"]
    kp_3d = torch.stack(first_obs["kp"].squeeze(0).chunk(3, dim=0), dim=1)

    kp_u, kp_v = first_obs[
        scene_data._get_cam_attr_name(config.camera, "2d_locations")
    ].chunk(2)

    kp_u = (kp_u / 2 + 0.5) * rgb.shape[1]
    kp_v = (kp_v / 2 + 0.5) * rgb.shape[0]
    kp_2d = torch.stack([kp_u, kp_v], dim=1)

    lik = first_obs[scene_data._get_cam_attr_name(config.camera, "post")]

    lik = (
        torch.nn.functional.interpolate(
            lik.unsqueeze(0),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(0)
        .sum(dim=0)
    )  # average over channels/kps

    fig = plt.figure()
    add_2d_subplot(fig, rgb, None, kp_2d, 1)
    add_2d_subplot(fig, rgb, lik, kp_2d, 2)
    add_2d_subplot(fig, depth, None, kp_2d, 3)
    add_2d_subplot(fig, depth, lik, kp_2d, 4)

    add_3d_subplot(fig, ee_pose, extr, kp_3d, ee_traj, 5)

    plt.show()


def add_2d_subplot(fig, img, lik, kp_2d, idx):
    ax = fig.add_subplot(3, 2, idx)
    ax.imshow(img)
    if lik is not None:
        ax.imshow(lik, alpha=0.7)
    ax.scatter(kp_2d[:, 0], kp_2d[:, 1], s=5, c="r")


def add_3d_subplot(fig, ee_pose, extr, kp_3d, ee_traj, idx):
    ax = fig.add_subplot(3, 2, idx, projection="3d")
    ee_pos = ee_pose[:3]
    ee_rot = quaternion_to_matrix(ee_pose[3:])
    plot_coordindate_frame(
        ax,
        origin=ee_pos,
        rotation=ee_rot,
        arrow_length=0.5,
        linewidth=1,
        annotation="ee",
        annotation_size=10,
    )

    plot_coordindate_frame(
        ax,
        transformation=extr.double(),
        arrow_length=0.5,
        linewidth=1,
        annotation="camera",
        annotation_size=10,
    )

    ax.scatter(kp_3d[:, 0], kp_3d[:, 1], kp_3d[:, 2], s=5, c="r")

    ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], c="b")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def complete_config(args: argparse.Namespace, config: DictConfig) -> DictConfig:
    config.encoder_name = args.encoder_name
    config.encoding_name = args.encoding_name
    config.traj_idx = args.traj

    return config


if __name__ == "__main__":
    extra_args = (
        {
            "name": "--encoder_name",
            "required": True,
        },
        {
            "name": "--encoding_name",
            "required": True,
        },
        {
            "name": "--traj",
            "required": False,
            "help": "trajectory index",
            "default": 0,
            "type": int,
        },
    )
    args, dict_config = parse_and_build_config(extra_args=extra_args)
    dict_config = complete_config(args, dict_config)  # type: ignore

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    print(config)

    main(config)  # type: ignore
