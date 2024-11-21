from dataclasses import dataclass, field

import numpy as np
import torch
import tsdf.fusion as fusion
import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.scene import SubSampleTypes
from tapas_gmm.env import Environment
from loguru import logger
from omegaconf import OmegaConf, SCMode
from tapas_gmm.tsdf.cluster import Cluster  # type: ignore
from tapas_gmm.tsdf.filter import (
    coordinate_boxes,
    filter_plane_from_mesh_and_pointcloud,
    gripper_dists,
)

from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.cuda import (
    try_debug_memory,
    try_destroy_context,
    try_empty_cuda_cache,
    try_make_context,
)
from tapas_gmm.utils.misc import DataNamingConfig, apply_machine_config, load_scene_data
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.select_gpu import device

# import viz.image_series as viz_image_series


@dataclass
class FusionConfig:
    subsample_type: dict[str, SubSampleTypes] = field(
        default_factory=lambda: {
            "wrist": SubSampleTypes.POSE,
            "overhead": SubSampleTypes.CONTENT,
        }
    )
    fusion_cams: tuple[str] = ("wrist",)
    scene_cams: tuple[str, ...] | None = None  # if None, all cams are used


@dataclass
class Config:
    data_naming: DataNamingConfig

    env: Environment

    fusion: FusionConfig = FusionConfig()


@torch.no_grad()
@logger.contextualize(filter=False)
def main(config: Config):
    scene_data = load_scene_data(config.data_naming)
    out_dir = scene_data.initialize_scene_reconstruction()

    if out_dir is None:
        return

    scene_cams = config.fusion.scene_cams or scene_data.camera_names
    fusion_cams = config.fusion.fusion_cams

    coordinate_box = coordinate_boxes[config.env]
    gripper_dist = gripper_dists[config.env]

    for t in range(len(scene_data)):
        # Load scene only with fusion cams and subsampled for reconstruction.
        fusion_views = scene_data.get_scene(
            traj_idx=t, cams=fusion_cams, subsample_types=config.fusion.subsample_type
        )
        fusion_scene = torch.cat([v for _, v in fusion_views.items()])

        # fusion_scene is SingleCamSceneObservation (stacked from all cams)
        rgb = fusion_scene.rgb.numpy()  # type: ignore
        depth = fusion_scene.depth.numpy()  # type: ignore
        extrinsics = fusion_scene.extr.numpy()  # type: ignore
        intrinsics = fusion_scene.intr.numpy()  # type: ignore
        n_imgs = len(rgb)

        H, W = rgb.shape[-2:]

        try_empty_cuda_cache()

        logger.info("Images to fuse: {} from cams {}", n_imgs, fusion_cams)

        context = try_make_context(device)

        tsdf_vol = fusion.fuse(
            rgb, depth, intrinsics, extrinsics, coordinate_box, gripper_dist
        )
        tsdf_point_cloud = tsdf_vol.get_point_cloud()[:, :3]

        # We do not need the meshes and point clouds in the dataset, so
        # save them externally for verification purposes. Can be skipped.
        traj_name = scene_data._paths[t].parts[-1]
        mesh_path = out_dir / ("mesh_" + traj_name + ".ply")
        _, faces = fusion.write_mesh(tsdf_vol, mesh_path)
        fusion.write_pc(tsdf_vol, out_dir / ("pc_" + traj_name + ".ply"))

        try_destroy_context(context)

        vertices, faces = filter_plane_from_mesh_and_pointcloud(tsdf_point_cloud, faces)

        logger.info(
            "Clustering point cloud of size {}, type {}, ...",
            vertices.shape[0],
            vertices.dtype,
        )
        cluster = Cluster(eps=0.03, min_samples=5000)
        fitted_cluster = cluster.fit(vertices)
        # cluster labels start at zero, noisy is -1, so + 1 for object labels
        pc_labels = fitted_cluster.labels_ + 1

        # Load entire scene for mask generation.
        scene_views = scene_data.get_scene(
            traj_idx=t, cams=scene_cams, subsample_types=None
        )
        scene = torch.cat([v for _, v in scene_views.items()])

        extrinsics = scene.extr.to(device)  # type: ignore
        intrinsics = scene.intr  # type: ignore
        n_imgs = extrinsics.shape[0]

        try_empty_cuda_cache()

        try_debug_memory(device)

        logger.info("Generating masks for {} images from cams {}", n_imgs, scene_cams)
        mask, labels = fusion.build_masks_via_mesh(
            vertices, faces, pc_labels, intrinsics, extrinsics, H, W
        )

        labels = np.delete(labels, np.argwhere(labels == 0)).tolist()
        logger.info("Generated labels {}", labels)

        logger.info("Writing masks to disk ...")
        traj_lens = [len(t.rgb) for _, t in scene_views.items()]
        traj_ends = np.cumsum(traj_lens)
        masks_per_cam = np.split(mask, traj_ends)

        sanity_check = masks_per_cam.pop(-1)
        assert len(sanity_check) == 0

        for c, m in zip(scene_cams, masks_per_cam):
            scene_data.add_tsdf_masks(t, c, m, labels)  # type: ignore

        # NOTE: these are useful for debugging/inspection
        # viz_image_series.vis_series_w_mask(scene_data.camera_obs_w[t],
        #                                    mask)
        # write_mask(mask, out_dir + "/masks_" + str(t) + ".pkl")

        try_empty_cuda_cache()


if __name__ == "__main__":
    args, dict_config = parse_and_build_config()

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore
