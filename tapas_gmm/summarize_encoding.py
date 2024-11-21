import argparse
import pathlib
from dataclasses import dataclass
from enum import Enum
from functools import partial
from math import ceil, sqrt
from typing import Any

import matplotlib
import matplotlib.patches as mpatches
import torch
import torchvision.transforms as T
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode
from tensordict import TensorDict

from tapas_gmm.dataset.scene import SceneDataset
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.misc import DataNamingConfig, load_scene_data
from tapas_gmm.utils.observation import ObservationConfig, tensorclass_from_tensordict
from tapas_gmm.viz.operations import channel_front2back, rgb2gray

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

tab_colors = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


class Outside(Enum):
    SKIP = 0
    PLOT = 1
    CLAMP = 2


@dataclass
class Config:
    data_naming: DataNamingConfig
    observation: ObservationConfig

    # set via command line
    encoding_paths: tuple[str] | None = None
    gt_path: str | None = None
    gt_path_world: str | None = None
    gt_attr_name: str = "descriptor"
    heatmap_attr_name: str = "heatmaps"  # for pf. for df: sm, prior, post
    kp_attr_name: str = "kp"
    kp_pf_3d_attr_name: str = "kp_world_coordinates"
    traj_idx: int | None = None

    show_kp: bool = True
    annotate_kp: bool = True
    annotation_color: str = "w"
    show_img: bool = True
    img_color: bool = False
    show_heatmap: bool = False
    enum_steps: bool = False
    show_2d_traj: bool = True
    show_3d_traj: bool = False
    show_pf_dbg_3d_traj: bool = False

    outside: Outside = Outside.PLOT
    channel: int | None = None
    min_index: int | None = None
    max_index: int | None = None
    index_step: int = 1

    n_cols: int = 16

    strict_shape_checking: bool = True
    n_kp: int = 16
    d_kp: Any = 3
    # d_kp: int | tuple[int, ...] | None = 3
    d_kp_gt: int = 3

    markersize: int = int(512 / n_cols)
    markerwidth: int = 3
    # NOTE: markers and colors are per encoder and per keypoint, ie
    # tuple[tuple[str, ...], ...], but OmegaConf does not support nesting yet.
    markers: Any = (tuple(["D"] * 8 + ["X"] * 8),)
    colors: Any = tuple((tuple([c] * 16) for c in tab_colors))
    gt_markers: tuple[str, ...] = tuple(["o"] * 16)
    gt_colors: tuple[str, ...] = tuple(["r"] * 16)
    legend: bool = True  # assumes one color per encoder

    save: str | None = None


def subplots(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.set_tight_layout({"pad": 0})
    return fig, axes


def savefig(filename, dir="/home/hartzj/", *args, **kwargs):
    # plt.savefig(dir + filename + '.pdf', *args, **kwargs)
    # plt.savefig(dir + filename + '.pgf', *args, **kwargs)
    plt.savefig(dir + filename + ".png", *args, **kwargs)


def outside(loc, min, max):
    out = torch.logical_or(
        torch.logical_or(loc[:, 0] < min[1], loc[:, 0] > max[1]),
        torch.logical_or(loc[:, 1] < min[0], loc[:, 1] > max[0]),
    )

    return out


def unflatten_keypoints(
    kp: torch.Tensor, kp_dim: int, drop_z: bool, n_kp: int | None = None
):
    if kp_dim is None:
        logger.info("Did not specify kp_dim. Inferring automatically.")
        if n_kp is None:
            raise ValueError("Need to specify either kp_dim or n_kp.")
        assert n_kp is not None
        kp_dim = kp.shape[-1] // n_kp
    # unflatten from stacked x, y, (z) to n_kp, d_kp
    if len(kp.shape) == 1:
        idx = 2 if drop_z else 3
        return torch.stack(torch.chunk(kp, kp_dim, dim=0)[:idx], dim=0)
    elif len(kp.shape) == 2:
        idx = 2 if drop_z else 3
        return torch.stack(torch.chunk(kp, kp_dim, dim=1)[:idx], dim=2)
    else:
        raise NotImplementedError


def main(config: Config) -> None:
    assert not config.show_3d_traj or not config.show_pf_dbg_3d_traj

    scene_data = load_scene_data(config.data_naming)
    scene_data.update_camera_crop(config.observation.image_crop)
    config.observation.image_dim = scene_data.image_dimensions

    split_paths = tuple(path.split("/") for path in config.encoding_paths)
    encoder_names = tuple(t[0] for t in split_paths)
    kp_selection_names = tuple(t[1] for t in split_paths)

    n_cams = len(config.observation.cameras)

    if len(config.markers) == 1:
        config.markers = tuple(list(config.markers) * len(encoder_names))

    # tuple (over encoders) of trajectories as tensorclasses
    trajectories = get_trajs(config, scene_data, encoder_names, kp_selection_names)
    trajectories_ss = subsample_trajs(config, trajectories)

    if config.show_kp:
        kp = get_kp(config, encoder_names, n_cams, trajectories_ss)
    else:
        kp = None

    if config.gt_path is not None:
        gt_kp_per_cam = get_gt_kp(config, scene_data, encoder_names, config.gt_path)
    else:
        gt_kp_per_cam = None

    if config.show_2d_traj:
        plot_2d(config, encoder_names, n_cams, trajectories_ss, kp, gt_kp_per_cam)

    if config.show_3d_traj or config.show_pf_dbg_3d_traj:
        assert config.d_kp == 3

        if config.gt_path_world is not None:
            gt_kp = get_gt_kp(config, scene_data, encoder_names, config.gt_path_world)
            assert len(gt_kp) == 1
            gt_kp = unflatten_keypoints(gt_kp[0], 3, False)
        else:
            gt_kp = None

        if config.show_pf_dbg_3d_traj:
            kp_3d = tuple(
                getattr(trajectories[e], config.kp_pf_3d_attr_name).cpu().numpy()
                for e in range(len(encoder_names))
            )
        else:
            assert n_cams == 1
            kp_3d = tuple(e[0] for e in kp)

        plot_3d(config, encoder_names, kp_3d, gt_kp)

        plt.show()


def get_kp(
    config: Config,
    encoder_names: tuple[str, ...],
    n_cams: int,
    trajectories_ss: tuple[Any, ...],
):
    check_kp_shape(config, n_cams, trajectories_ss)
    # tuple (over encoders) of tuple (over cameras) of tensors
    kp_flat_per_cam = tuple(
        torch.chunk(getattr(t, config.kp_attr_name), n_cams, dim=-1)
        for t in trajectories_ss
    )
    kp = tuple(
        tuple(
            unflatten_keypoints(
                kp_flat_per_cam[e][c].squeeze(1), config.d_kp[e], False, config.n_kp
            )
            for c in range(n_cams)
        )
        for e in range(len(encoder_names))
    )

    return kp


def plot_3d(
    config: Config,
    encoder_names: tuple[str, ...],
    kp: tuple[torch.Tensor, ...],
    gt_kp: torch.Tensor | None,
) -> None:
    n_plots = config.n_kp
    n_cols = n_rows = ceil(sqrt(n_plots))

    fig, axes = subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 4),
        subplot_kw=dict(projection="3d"),
    )

    ax_min = min([t.min() for t in kp])
    ax_max = max([t.max() for t in kp])

    ax_max *= 1.1 if ax_max > 0 else 0.9
    ax_min *= 1.1 if ax_min < 0 else 0.9

    for i in range(config.n_kp):
        ax = axes.flat[i]
        for e in range(len(encoder_names)):
            traj = kp[e][:, i]
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                traj[:, 2],
                color=config.colors[e][0],
                label=config.encoding_paths[e] if i == 0 else None,
            )
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                traj[0, 2],
                color=config.colors[e][0],
                marker="o",
            )

        if gt_kp is not None:
            ax.plot(
                gt_kp[:, i, 0],
                gt_kp[:, i, 1],
                gt_kp[:, i, 2],
                color=config.gt_colors[0],
                label=config.gt_path if i == 0 else None,
            )
            ax.scatter(
                gt_kp[0, i, 0],
                gt_kp[0, i, 1],
                gt_kp[0, i, 2],
                color=config.gt_colors[0],
                marker="o",
            )

        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_zlim(ax_min, ax_max)

    fig.legend(loc="outside upper center")


def plot_2d(
    config: Config,
    encoder_names: tuple[str, ...],
    n_cams: int,
    trajectories: tuple[Any, ...],
    kp_per_cam: tuple[tuple[torch.Tensor, ...], ...],
    gt_kp_per_cam,
) -> None:
    if any((d is not None and d > 2 for d in config.d_kp)):
        logger.info(
            f"Specified {config.d_kp}D keypoints. Plotting "
            "in 2D. Is proper projection used? (Ego or None)"
        )

    len_t = len(trajectories[0])
    n_plots = len_t * n_cams
    n_cols = config.n_cols
    n_rows = n_plots // n_cols

    if n_plots % n_cols != 0:
        n_rows += 1

    fig, axes = subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for i in range(len_t):
        for c, cam_name in enumerate(config.observation.cameras):
            ax = axes.flat[i + c * len_t]
            if config.show_img:
                img = trajectories[0].camera_obs[c].rgb[i]
                img = channel_front2back(img)
                if not config.img_color:
                    img = rgb2gray(img)
                ax.imshow(img, alpha=0.5, cmap="gray" if not config.img_color else None)
                heatmap_alpha = 0.7
            else:
                heatmap_alpha = 1.0

            if config.show_heatmap:
                if len(encoder_names) > 1:
                    raise NotImplementedError("Heatmap only for one encoder.")

                for e in range(len(encoder_names)):
                    heatmap = getattr(trajectories[e], config.heatmap_attr_name).get(
                        cam_name
                    )[i]
                    if heatmap.shape != img.shape:
                        heatmap = T.Resize(img.shape[1:])(heatmap)
                    ax.imshow(heatmap.mean(0), alpha=heatmap_alpha, cmap="viridis")

            if config.show_kp:
                for e in range(len(encoder_names)):
                    kp = kp_per_cam[e][c][i].swapaxes(0, 1)
                    scatter_kp(config, ax, kp, config.colors[e], config.markers[e])

            if config.gt_path is not None:
                assert gt_kp_per_cam is not None
                gt_kp_flat = gt_kp_per_cam[c][i]
                gt_kp = unflatten_keypoints(gt_kp_flat, config.d_kp_gt, True)
                scatter_kp(config, ax, gt_kp, config.gt_colors, config.gt_markers)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.axis("off")
            if config.enum_steps:
                ax.set_title(cam_name + " " + str(i))
            #  ax.set_aspect('equal')
            ax.set_anchor("N")

    if config.legend:
        patches = [
            mpatches.Patch(color=c[0], label=l)
            for c, l in zip(config.colors, config.encoding_paths)
        ]
        if config.gt_path is not None:
            patches.append(
                mpatches.Patch(color=config.gt_colors[0], label=config.gt_path)
            )
        fig.legend(handles=patches, loc="outside upper center")
        # fig.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=3,
        #            handles=patches)

    fig.subplots_adjust(wspace=0.04, hspace=0.04)

    if config.save is not None:
        savefig(config.save, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def get_trajs(
    config: Config,
    scene_data: SceneDataset,
    encoder_names: tuple[str, ...],
    kp_selection_names: tuple[str, ...],
) -> tuple[Any, ...]:
    encoding_attrs = {config.kp_attr_name: config.kp_attr_name}

    if config.show_pf_dbg_3d_traj:
        encoding_attrs[config.kp_pf_3d_attr_name] = config.kp_pf_3d_attr_name

    if config.show_heatmap:
        for c in config.observation.cameras:
            c_attr = scene_data._get_cam_attr_name(c, config.heatmap_attr_name)
            encoding_attrs[c_attr] = c_attr

    # Get as tensor_DICT, bcs heatmap is not designated in the SceneObservation
    # tensorclass. Convert to a bespoke TensorClass afterwards.
    trajectories = tuple(
        scene_data._get_bc_traj(
            traj_idx=config.traj_idx,  # type: ignore
            cams=config.observation.cameras,
            encoder_name=e,
            encoding_name=n,
            encoding_attr=encoding_attrs,
            mask_type=None,
            skip_rgb=not config.show_img or i > 0,
            skip_generic_attributes=True,
            as_tensor_dict=True,
        )
        for i, (e, n) in enumerate(zip(encoder_names, kp_selection_names))
    )

    if config.show_heatmap:
        for t in trajectories:
            t[config.heatmap_attr_name] = {}
            for c in config.observation.cameras:
                c_attr = scene_data._get_cam_attr_name(c, config.heatmap_attr_name)
                heatmaps = t.pop(c_attr)
                t[config.heatmap_attr_name][c] = heatmaps

    data_class = tensorclass_from_tensordict(trajectories[0], "BespokeObs")
    trajectories = tuple(data_class(**t, batch_size=t.batch_size) for t in trajectories)

    return trajectories


def subsample_trajs(config: Config, trajectories: tuple[Any, ...]) -> tuple[Any, ...]:
    trajectories = tuple(
        t[config.min_index : config.max_index : config.index_step] for t in trajectories
    )

    return trajectories


def check_kp_shape(
    config: Config, n_cams: int, trajectories: tuple[TensorDict, ...]
) -> tuple[Any, ...]:
    if config.strict_shape_checking:
        for e, t in enumerate(trajectories):
            kp = getattr(t, config.kp_attr_name)
            assert kp.shape[-1] == config.n_kp * config.d_kp[e] * n_cams, (
                f"Non-matching keypoint dim {kp.shape[-1]} for config "
                f"n_kp={config.n_kp}, d_kp={config.d_kp[e]}, n_cams={n_cams}"
            )


def get_gt_kp(
    config: Config,
    scene_data: SceneDataset,
    encoder_names: tuple[str, ...],
    gt_path: str,
) -> tuple[torch.Tensor, ...]:
    embedding_attr = {
        c: {config.gt_attr_name: scene_data._get_cam_attr_name(c, config.gt_attr_name)}
        for c in config.observation.cameras
    }

    gt_trajectory = scene_data._get_bc_traj(
        traj_idx=config.traj_idx,  # type: ignore
        cams=config.observation.cameras,
        encoder_name=gt_path,
        embedding_attr=embedding_attr,
        mask_type=None,
        skip_rgb=len(encoder_names) > 0,
        skip_generic_attributes=True,
    )

    gt_trajectory = gt_trajectory[
        config.min_index : config.max_index : config.index_step
    ]
    gt_kp_per_cam = tuple(
        getattr(t, config.gt_attr_name) for t in gt_trajectory.camera_obs
    )

    if config.strict_shape_checking:
        for t in gt_kp_per_cam:
            assert t.shape[-1] == config.n_kp * config.d_kp_gt, (
                f"Non-matching GT keypoint dim {t.shape[-1]} for "
                f"config n_kp={config.n_kp}, d_kp={config.d_kp_gt}"
            )

    return gt_kp_per_cam


def scatter_kp(
    config: Config,
    ax: plt.Axes,
    kp: torch.Tensor,
    colors: tuple[str, ...],
    markers: tuple[str, ...],
) -> None:
    if config.channel is not None:
        kp = kp[:, config.channel].unsqueeze(1)

    assert config.observation.image_dim is not None

    # remap from [-1, 1] to image size
    kp_remapped = torch.stack(
        (
            (kp[0] + 1) * config.observation.image_dim[1] / 2,
            (kp[1] + 1) * config.observation.image_dim[0] / 2,
        ),
        dim=1,
    )

    if config.outside == Outside.CLAMP:
        logger.warning("Not tested clamping code yet.")
        kp_remapped = torch.stack(
            (
                torch.clamp(kp_remapped[0], 0, config.observation.image_dim[1]),
                torch.clamp(kp_remapped[1], 0, config.observation.image_dim[0]),
            ),
            dim=0,
        )
        skip = torch.zeros_like(kp_remapped[:, 0]).bool()
    elif config.outside == Outside.SKIP:
        skip = outside(kp_remapped, (0, 0), config.observation.image_dim)
    else:
        skip = torch.zeros_like(kp_remapped[:, 0]).bool()

    for j, s, k, c, m in zip(range(config.n_kp), skip, kp_remapped, colors, markers):
        if not s:
            ax.scatter(
                k[0],
                k[1],
                marker=m,  # type: ignore
                edgecolors=c,
                facecolor="none",
                s=config.markersize,
                linewidths=config.markerwidth,
            )
            if config.annotate_kp:
                ax.annotate(
                    str(j), (k[0], k[1]), color=config.annotation_color  # type: ignore
                )


def complete_config(args: argparse.Namespace, config: DictConfig) -> DictConfig:
    config.encoding_paths = args.encoding_paths
    config.gt_path = args.gt_path
    config.gt_path_world = args.gt_path_world
    config.traj_idx = args.traj

    if type(config.d_kp) is int or config.d_kp is None:
        config.d_kp = tuple([config.d_kp] * len(config.encoding_paths))

    return config


if __name__ == "__main__":
    extra_args = (
        {
            "name": "--encoding_paths",
            "required": True,
            "nargs": "+",
            "help": "One or multiple in form encoder_name/kp_selection_name",
        },
        {"name": "--gt_path", "required": False, "help": "gt_encoder_name"},
        {
            "name": "--gt_path_world",
            "required": False,
            "help": "gt_encoder_name in world frame",
        },
        {"name": "--traj", "required": True, "help": "trajectory index"},
    )

    args, dict_config = parse_and_build_config(extra_args=extra_args)
    dict_config = complete_config(args, dict_config)  # type: ignore

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore
