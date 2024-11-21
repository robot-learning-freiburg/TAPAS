from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig

import tapas_gmm.encoder.keypoints as keypoints
import tapas_gmm.encoder.models.keypoints.keypoints as keypoints_model
import tapas_gmm.encoder.models.keypoints.model_based_vision as model_based_vision
import tapas_gmm.encoder.representation_learner as representation_learner
from tapas_gmm.utils.geometry_torch import (
    append_depth_to_uv,
    batched_project_onto_cam,
    batched_rigid_transform,
    conjugate_quat,
    hard_pixels_to_3D_world,
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    invert_intrinsics,
    quaternion_to_matrix,
)
from tapas_gmm.utils.observation import SampleTypes

# from tapas_gmm.utils.debug import nan_hook, summarize_tensor
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.torch import eye_like
from tapas_gmm.viz.image_single import image_with_points_overlay_uv_list
from tapas_gmm.viz.surface import depth_map_with_points_overlay_uv_list, scatter3d

# import viz.keypoint_selector


KeypointsTypes = keypoints_model.KeypointsTypes
# ManualKeypointSelectorConfig = viz.keypoint_selector.ManualKeypointSelectorConfig
ProjectionTypes = keypoints.ProjectionTypes
ReferenceSelectionTypes = keypoints.ReferenceSelectionTypes
PriorTypes = keypoints.PriorTypes


@dataclass
class PreTrainingConfig:
    ref_selection: ReferenceSelectionTypes = ReferenceSelectionTypes.MANUAL
    ref_labels: tuple[int, ...] | None = None
    ref_traj_idx: int | None = None
    ref_obs_idx: int | None = None
    # ref_selector: ManualKeypointSelectorConfig | None = None
    ref_selector: Any = None


@dataclass
class EncoderConfig:
    debug: bool
    descriptor_dim: int
    projection: ProjectionTypes
    keypoints: keypoints.KeypointsConfig
    prior_type: PriorTypes = (
        PriorTypes.NONE
    )  # For compatibility with kp encoder. Also set to None.


@dataclass
class GTKeypointsPredictorConfig:
    encoder: EncoderConfig

    pretraining: PreTrainingConfig

    end_to_end: bool = False


class GTKeypointsPredictor(keypoints.KeypointsPredictor):
    sample_type = SampleTypes.GT

    disk_read_embedding = False

    def __init__(self, config: DictConfig) -> None:
        kp_config = config.encoder_config
        assert type(kp_config) is GTKeypointsPredictorConfig

        encoder_config = kp_config.encoder
        self.config = encoder_config
        self.pretrain_config = kp_config.pretraining

        representation_learner.RepresentationLearner.__init__(self, config=config)

        self.n_keypoints = self.get_no_keypoints()

        self.only_use_first_emb = True  # with 3D or ego proj only need one cam

        image_size = config.observation.image_dim

        self.image_height, self.image_width = image_size
        self.descriptor_dimension = encoder_config.descriptor_dim
        self.keypoint_dimension = (
            2 if encoder_config.projection is ProjectionTypes.NONE else 3
        )

        self.debug_kp_selection = encoder_config.debug

        self._register_buffer()

    def _register_buffer(self):
        self.register_buffer("ref_pixels_uv", torch.Tensor(2, self.n_keypoints))
        self.register_buffer("ref_pixel_world", torch.Tensor(self.n_keypoints, 3))

        self.register_buffer(
            "ref_depth", torch.Tensor(self.image_height, self.image_width)
        )
        self.register_buffer("ref_int", torch.Tensor(3, 3))
        self.register_buffer("ref_ext", torch.Tensor(4, 4))

    def set_extra_state(self, state: dict) -> None:
        """
        Buffer functionality in TensorClass does not yet work properly.
        So add the ref_object_poses as an extra state.
        """
        self.ref_object_poses = state["ref_object_poses"]

    def get_extra_state(self) -> dict:
        return {"ref_object_poses": self.ref_object_poses}

    def encode(self, batch):
        camera_obs = batch.camera_obs

        depth = camera_obs[0].depth
        extrinsics = camera_obs[0].extr
        intrinsics = camera_obs[0].intr
        object_poses = batch.object_poses

        if self.disk_read_embedding:
            kp = getattr(camera_obs[0], self.embedding_name)
            descriptor = None
            distance = None
            kp_raw_2d = None
            depth = None
        else:
            kp, descriptor, distance, kp_raw_2d = self.compute_keypoints(
                depth, extrinsics, intrinsics, object_poses
            )

        info = {
            "descriptor": descriptor,
            "distance": distance,
            "kp_raw_2d": kp_raw_2d,
            "depth": depth,
        }

        return kp, info

    def compute_keypoints(
        self,
        depth,
        extrinsics,
        intrinsics,
        cur_object_pose,
        ref_pixel_world=None,
        ref_object_poses=None,
        projection=None,
    ):
        # Can pass other reference pixel, pose than the GT one saved at kp
        # sampling. If None provided, will use those.
        if ref_pixel_world is None:
            ref_pixel_world = self.ref_pixel_world
        if ref_object_poses is None:
            ref_object_poses = self.ref_object_poses

        B = depth.shape[0]

        object_order = ref_object_poses.sorted_keys
        n_objs = len(object_order)

        kp_x = []
        kp_y = []
        kp_z = []
        kp_world = []

        ref_pixel_world = ref_pixel_world.chunk(n_objs, dim=0)

        projection = self.config.projection

        # Iterate over objects as their relative poses can change.
        for i, n in enumerate(object_order):
            # Move them by the pose difference of the object between the ref
            # pose and current pose (poses change between trajectories).
            ref_shift = ref_object_poses[n][:3]
            ref_rot = ref_object_poses[n][3:7]
            cur_shift = cur_object_pose[n][:, :3]
            cur_rot = cur_object_pose[n][:, 3:7]

            # quat_diff = quaternion_multiply(
            #     conjugate_quat(ref_rot).repeat(B, 1), cur_rot
            # )
            # rel_rot_matrix = quaternion_to_matrix(quat_diff)

            # move_back = eye_like(extrinsics)
            # move_back[:, :3, 3] = -ref_shift
            # rel_rot = eye_like(extrinsics)
            # rel_rot[:, :3, :3] = rel_rot_matrix
            # move_forth = eye_like(extrinsics)
            # move_forth[:, :3, 3] = cur_shift
            # ref_to_cur = torch.matmul(move_forth, torch.matmul(rel_rot, move_back))

            logger.error(
                "Updated code without testing this part. Use viz to see if it works."
                " Remove old comments and this error if it does."
            )
            cur_hom = homogenous_transform_from_rot_shift(
                quaternion_to_matrix(cur_rot), cur_shift
            )
            ref_hom = homogenous_transform_from_rot_shift(
                quaternion_to_matrix(ref_rot), ref_shift
            )

            ref_to_cur = torch.matmul(cur_hom, invert_homogenous_transform(ref_hom))

            cur_pixel_world = batched_rigid_transform(ref_pixel_world[i], ref_to_cur)

            clip = projection != ProjectionTypes.EGO

            # Project the new points onto the current camera
            cur_pixel_cam, cur_pixel_depth = batched_project_onto_cam(
                cur_pixel_world,
                depth,
                intrinsics,
                extrinsics,
                clip=clip,
                get_depth=True,
            )

            kp_world.append(cur_pixel_world)

            kp_x.append(cur_pixel_cam[:, :, 0])
            kp_y.append(cur_pixel_cam[:, :, 1])
            kp_z.append(cur_pixel_depth)

        kp = torch.cat(
            (
                torch.cat(kp_x, dim=-1) * 2 / self.image_width - 1,
                torch.cat(kp_y, dim=-1) * 2 / self.image_height - 1,
            ),
            dim=-1,
        ).to(device)

        kp_z = torch.cat(kp_z, dim=-1).to(device)

        kp_world = torch.cat(kp_world, dim=1).to(device)

        distance = torch.zeros(B, self.n_keypoints)

        kp_raw_2d = kp

        if projection == ProjectionTypes.NONE:
            pass
        elif projection in (ProjectionTypes.UVD, ProjectionTypes.EGO):
            kp = torch.cat((kp, kp_z), dim=-1)
        elif projection == ProjectionTypes.GLOBAL_HARD:
            kp = kp_world.permute(0, 2, 1).reshape((B, -1))
        elif projection == ProjectionTypes.LOCAL_HARD:
            # create identity extrinsics
            extrinsics = torch.zeros_like(extrinsics)
            extrinsics[:, range(4), range(4)] = 1
            kp = hard_pixels_to_3D_world(
                kp,
                kp_z,
                extrinsics,
                intrinsics,
                self.image_width - 1,
                self.image_height - 1,
            )
        else:
            raise NotImplementedError

        return kp, torch.zeros_like(depth), distance, kp_raw_2d

    def initialize_parameters_via_dataset(self, replay_memory, cam, **kwargs):
        self.select_reference_descriptors(replay_memory, cam=cam)

    def select_reference_descriptors(
        self, dataset, traj_idx=0, img_idx=0, object_labels=None, cam="wrist"
    ):
        traj_idx = self.pretrain_config.ref_traj_idx or traj_idx
        img_idx = self.pretrain_config.ref_obs_idx or img_idx

        config_labels = self.pretrain_config.ref_labels
        assert config_labels is None or object_labels is None
        object_labels = config_labels or object_labels or dataset.get_object_labels()

        ref_obs = dataset.sample_data_point_with_ground_truth(
            cam=cam, img_idx=img_idx, traj_idx=traj_idx
        )

        self.ref_depth = ref_obs.depth.to(device)
        self.ref_object_poses = ref_obs.object_poses.to(device)
        # self._buffers['ref_object_poses'] = self.ref_object_poses  # HACK
        self.ref_int = ref_obs.intr.to(device)
        self.ref_ext = ref_obs.extr.to(device)

        # dummy tensor
        descriptor = torch.zeros(
            (1, self.descriptor_dimension, self.image_height, self.image_width)
        )

        object_labels = object_labels or dataset.get_object_labels()

        n_keypoints_total = self.config.keypoints.n_sample
        ref_selection = self.pretrain_config.ref_selection
        ref_selector_config = self.pretrain_config.ref_selector

        if (
            ref_selection is not ReferenceSelectionTypes.MANUAL
            and len(object_labels) > 1
        ):
            raise NotImplementedError(
                "The object order in KP selection needs to be identical to the"
                " order in the object-poses dict. Might be possible to sync "
                "using object labels. For now deactivated to avoid unexpected "
                "behavior."
            )

        if ref_selection is ReferenceSelectionTypes.MANUAL:
            from tapas_gmm.viz.keypoint_selector import ManualKeypointSelectorConfig

            assert type(ref_selector_config) is ManualKeypointSelectorConfig

            preview_obs = dataset._get_bc_traj(
                traj_idx, cams=(cam,), fragment_length=-1, force_skip_rgb=False
            )
            indeces = np.linspace(0, stop=preview_obs.shape[0] - 1, num=20)

            indeces = np.round(indeces).astype(int)
            preview_obs = preview_obs.get_sub_tensordict(torch.tensor(indeces))

            preview_frames = preview_obs.camera_obs[0].rgb
            preview_descr = torch.zeros(
                (20, self.descriptor_dimension, self.image_height, self.image_width)
            )
        else:
            preview_frames = None
            preview_descr = None

        object_order = tuple(ref_obs.object_poses.sorted_keys)

        # ref_pixels_uv, reference_descriptor_vec = \
        #     self.sample_keypoints(rgb, descriptor, mask,
        #                           object_labels, n_keypoints_total)

        (
            self.ref_pixels_uv,
            self._reference_descriptor_vec,
        ) = self._select_reference_descriptors(
            ref_obs.rgb,
            descriptor,
            ref_obs.mask,
            object_labels,
            n_keypoints_total,
            ref_selection,
            preview_frames,
            preview_descr,
            object_order=object_order,
            ref_selector_config=ref_selector_config,
        )

        ref_pixels_stacked = torch.cat(
            (self.ref_pixels_uv[0], self.ref_pixels_uv[1]), dim=-1
        )

        # Map the reference pixel to world coordinates.
        ref_pixel_world = model_based_vision.raw_pixels_to_3D_world(
            ref_pixels_stacked.unsqueeze(0),
            self.ref_depth.unsqueeze(0),
            self.ref_ext.unsqueeze(0),
            self.ref_int.unsqueeze(0),
        )

        # Shape is (1, 3*k). Reshape to (k, 3)
        self.ref_pixel_world = torch.stack(
            ref_pixel_world.squeeze(0).chunk(3, dim=-1), dim=1
        ).to(device)

        if self.debug_kp_selection:
            if self.keypoint_dimension == 2:
                image_with_points_overlay_uv_list(
                    ref_obs.rgb.cpu(),
                    (self.ref_pixels_uv[0].numpy(), self.ref_pixels_uv[1].numpy()),
                    mask=ref_obs.mask,
                )
                # descriptor_image_np = descriptor_image_tensor.cpu().numpy()
                # plt.imshow(descriptor_image_np)
                # plt.show()
            elif self.keypoint_dimension == 3:
                depth_map_with_points_overlay_uv_list(
                    ref_obs.depth.cpu().numpy(),
                    (self.ref_pixels_uv[0].numpy(), self.ref_pixels_uv[1].numpy()),
                    mask=ref_obs.mask.cpu().numpy(),
                    object_labels=object_labels,
                    # object_poses=self.ref_object_pose.cpu().numpy()
                )
                local_ref = model_based_vision.raw_pixels_to_camera_frame(
                    ref_pixels_stacked.unsqueeze(0),
                    self.ref_depth.unsqueeze(0),
                    self.ref_int.unsqueeze(0),
                )
                scatter3d(local_ref[0].cpu().numpy())
                scatter3d(self.ref_pixel_world.cpu().numpy())
            else:
                raise ValueError(
                    "No viz for {}d keypoints.".format(self.keypoint_dimension)
                )

    def from_disk(self, chekpoint_path, force_read=False):
        if not force_read:
            logger.info(
                "  GT Keypoints encoder does not need snapshot loading." "Skipping."
            )
        else:
            logger.info(
                "  Force-reading the GT Keypoints encoder from disk. "
                "Should only be needed to preserve the reference selection "
                "in bc if you used embed_trajectories."
            )

            state_dict = torch.load(chekpoint_path, map_location="cpu")
            self.ref_object_poses = state_dict.pop("ref_object_poses")
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("Missing keys: {}".format(missing))
            if unexpected:
                logger.warning("Unexpected keys: {}".format(unexpected))
            self = self.to(device)

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        return keypoints.KeypointsPredictor.get_latent_dim(config, n_cams=1)

    def reset_episode(self):
        """
        Does not need any reset. Overwriting inherited method from keypoints.
        """
        pass
