import hashlib
import itertools
from functools import lru_cache
from typing import Any, Sequence

import numpy as np
import torch
from loguru import logger
from riepybdlib.mappings import s2_id
from sklearn.cluster import DBSCAN

from tapas_gmm.utils.geometry_np import ensure_quaternion_continuity
from tapas_gmm.utils.geometry_torch import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    conjugate_quat,
    cos,
    get_b_from_homogenous_transforms,
    get_R_from_homogenous_transforms,
    hom_to_shift_quat,
    homogenous_transform_from_rot_shift,
    identity_7_pose,
    identity_quaternions,
    invert_homogenous_transform,
    modulo_rotation_angle,
    quarter_rot_angle,
    quaternion_is_unit,
    quaternion_lot_multiply,
    quaternion_multiply,
    quaternion_to_axis_and_angle,
    quaternion_to_matrix,
    remove_quaternion_dim,
    rotate_quat_y180,
    rotate_vector_by_quaternion,
    set_b_in_homogenous_transforms,
    sin,
    standardize_quaternion,
    translation_to_direction_and_magnitude,
)
from tapas_gmm.utils.keypoints import tp_from_keypoints
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.torch import (
    batched_block_diag,
    cat,
    list_or_tensor,
    list_or_tensor_mult_args,
    single_index_any_tensor_dim,
    slice_any_tensor_dim,
    stack,
    to_numpy,
    unsqueeze,
)
from tapas_gmm.utils.typing import NDArrayOrNDArraySeq, TensorOrTensorSeq
from tapas_gmm.viz.quaternion import plot_quat_components

PREFER_SMALLER_ROTS = True  # TODO: Make this a config option


@list_or_tensor
def rotate_frames_180degrees(quats, rot_idx=None, skip_idx=None, axis="y"):
    """
    Rotate a set of quaternions (representing frame orientation) by 180 degrees
    around the specified axis.

    Parameters
    ----------
    quats : torch.Tensor
        Quaternions of shape (n_frames, ..., 4).
    axis : str, optional
        Axis to rotate around, by default 'y'.
    rot_idx : list[int]/None, optional
        Indices of the frames to rotate, by default None.
    skip_idx : list[int]/None, optional
        Indices of the frames to skip, by default None.

    Either rot_idx or skip_idx must be be a list of indices, the other must be
    None.

    Returns
    -------
    torch.Tensor
        Rotated quaternions.
    """
    if axis != "y":
        raise NotImplementedError

    shape = quats.shape[0]

    all_frames = list(range(shape))

    if rot_idx is not None:
        assert skip_idx is None
        skip_idx = [i for i in all_frames if i not in rot_idx]

    elif skip_idx is not None:
        rot_idx = [i for i in all_frames if i not in skip_idx]

    to_keep = quats[skip_idx]
    to_rotate = quats[rot_idx]

    rotated = rotate_quat_y180(to_rotate)

    res = []
    k, r = 0, 0
    k_max, r_max = len(to_keep), len(rotated)
    for i in range(shape):
        if k < k_max and i == skip_idx[k]:
            res.append(to_keep[k])
            k += 1
        elif r <= r_max:
            assert i == rot_idx[r]
            res.append(rotated[r])
            r += 1
        else:
            raise ValueError

    return torch.stack(res)


def configurable_rotate_frames(
    quats, enforce_z_down, enforce_z_up, with_init_ee_pose, with_world_frame
):
    """
    Homogenize the frame orientation based on the given parameters.
    If enforce_z_down is True, the z-axis of the frames will point down,
    if enforce_z_up is True, the z-axis of the frames will point up.
    If neither, do nothing.

    Parameters
    ----------
    quats : torch.Tensor/tuple[torch.Tensor]
        Quaternions of shape (n_frames, ..., 4).
    enforce_z_down : bool
        Enforce that the z-axis of the frames is pointing down.
    enforce_z_up : bool
        Enforce that the z-axis of the frames is pointing up.
    with_world_frame : bool
        Wether the first frame is the world frame.
    with_init_ee_pose : bool
        Wether the the ee_pose_frame is included (first frame if without
        world frame else second frame).

    Returns
    -------
    torch.Tensor/tuple[torch.Tensor]
        Rotated quaternions.
    """
    ee_idx = 1 if with_world_frame else 0

    if enforce_z_down:  # rotate all frames, but ee_pose_frame
        assert not enforce_z_up
        rot_idx = None
        skip_idx = [ee_idx] if with_init_ee_pose else []
    elif enforce_z_up:  # only rotate ee_pose_frame
        rot_idx = [ee_idx] if with_init_ee_pose else []
        skip_idx = None
    else:
        return quats

    return rotate_frames_180degrees(quats, rot_idx, skip_idx)


def get_frames_from_obs(
    obs: SceneObservation,
    frames_from_keypoints: bool,
    add_init_ee_pose_as_frame: bool,
    add_world_frame: bool,
    indeces: Sequence[int] | None,
    add_action_dim: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get object frames from the observation. For online inference. Meant to replicate
    the behavior of the demos class.

    Returns homogenous frame transforms and quaternions.
    """

    # TODO: Modulo rots, enforce z-up/down, etc.
    logger.warning("Implementation lacking modulo rots, enforce z-up/down, etc.")

    if frames_from_keypoints:
        object_poses = [
            p.squeeze(0) for p in tp_from_keypoints(obs.kp.unsqueeze(0), indeces)
        ]
    else:
        object_poses = [v for _, v in obs.object_poses.items()]
        if indeces is not None:
            object_poses = [object_poses[i] for i in indeces]

    if add_world_frame:
        object_poses = [identity_7_pose] + object_poses

    if add_init_ee_pose_as_frame:
        object_poses = [obs.ee_pose] + object_poses

    pose_tensor = torch.stack(object_poses)

    frame_quats = pose_tensor[..., 3:]
    frame_b = pose_tensor[..., :3]
    frame_A = quaternion_to_matrix(frame_quats)

    if add_action_dim:
        logger.warning("Assuming zero frame velocity in Adapter. Should be fixed.")
        frame_quats = torch.stack((frame_quats, frame_quats), dim=-1)

        frame_A = torch.kron(torch.eye(2), frame_A)

        frame_b = torch.cat((frame_b, torch.zeros_like(frame_b)), dim=-1)

    frame_hom = get_frame_transform_flat(frame_A, frame_b, invert=False)

    return frame_hom.numpy(), frame_quats.numpy()


class Demos:
    """
    Convenience class to store and access demonstrations for gmm models.
    """

    def __init__(
        self,
        trajectories: list[SceneObservation],
        meta_data: dict | None = None,
        add_init_ee_pose_as_frame: bool = True,
        add_world_frame: bool = True,
        enforce_z_down: bool = False,
        enforce_z_up: bool = False,
        modulo_object_z_rotation: bool = False,
        frames_from_keypoints: bool = False,
        kp_indeces: Sequence[int] | None = None,
        make_quats_continuous: bool = False,
    ):
        """
        Extract information from a list of BCTrajectories.

        Parameters
        ----------
        trajectories : list[BCTrajectory]
            List of trajectories.
        add_init_ee_pose_as_frame : bool, optional
            Add the initial EE pose as a frame, by default True.
        add_world_frame : bool, optional
            Add a world frame, by default True.
        meta_data : dict, optional
            Meta data for efficient hashing of the class, by default None.
        enforce_z_down : bool, optional
            Enforce that the z-axis of the frames is pointing down, by default
            True.
        enforce_z_up : bool, optional
            Enforce that the z-axis of the frames is pointing up, by default
        frames_from_keypoints : bool, optional
            If true, task-parameters (object frames) are extracted from the
            keypoints, else the GT object poses are used, by default False.

        enforce_z_down and enforce_z_up can be used to homogenize frame
        orientation as the EE frame points down, while object and world frames
        point up.
        """
        assert not (enforce_z_down and enforce_z_up)

        self.meta_data = {} if meta_data is None else meta_data
        self.meta_data["add_init_ee_pose_as_frame"] = add_init_ee_pose_as_frame
        self.meta_data["add_world_frame"] = add_world_frame
        self.meta_data["enforce_z_down"] = enforce_z_down
        self.meta_data["enforce_z_up"] = enforce_z_up
        self.meta_data["modulo_object_z_rotation"] = modulo_object_z_rotation
        self.meta_data["frames_from_keypoints"] = frames_from_keypoints
        self.meta_data["kp_indeces"] = kp_indeces

        if type(trajectories[0].object_poses) is torch.Tensor:
            logger.warning("Legacy support for non-named objects. Auto-naming frames.")
            for t in trajectories:
                t.object_poses = {
                    f"frame_{i}": v for i, v in enumerate(t.object_poses.swapdims(0, 1))
                }

        # Add the EE frame as the first frame. As this will be the obervation
        # we remove it later. However, it needs the same transformations, so
        # we add it here temporarily.
        ee_poses = tuple(o.ee_pose for o in trajectories)
        object_poses = tuple(
            (
                tp_from_keypoints(o.kp.squeeze(1), kp_indeces)
                if frames_from_keypoints
                else [v for _, v in o.object_poses.items()]
            )
            for o in trajectories
        )
        n_obj_frames = len(object_poses[0])
        frame_poses = tuple(
            torch.stack([e] + o) for e, o in zip(ee_poses, object_poses)
        )

        if modulo_object_z_rotation:
            frame_poses = modulo_rotation_angle(
                frame_poses, quarter_rot_angle, 2, skip_first=True
            )

        self.n_trajs = len(frame_poses)

        self.world2frames = []
        self.world2frames_velocities = []
        self.frames2world = []
        self.frames2world_velocities = []
        self.ee_poses = []
        self.ee_poses_raw = tuple(o.ee_pose for o in trajectories)
        self.ee_quats = tuple(o[..., 3:] for o in self.ee_poses_raw)

        if make_quats_continuous:
            self.ee_quats = tuple(
                ensure_quaternion_continuity(e) for e in self.ee_quats
            )

        # import matplotlib.pyplot as plt
        # for dim in range(4):
        #     plt.plot(self.ee_quats[0][:, dim])
        # plt.show()

        frame_quats = []
        if add_world_frame:
            frame_quats.append(
                tuple(identity_quaternions(o[0:1, :, 0].shape) for o in frame_poses)
            )
        if add_init_ee_pose_as_frame:
            frame_quats.append(
                tuple(
                    o[0].unsqueeze(0).unsqueeze(0).repeat(1, o.shape[0], 1)
                    for o in self.ee_quats
                )
            )

        frame_quats.append(tuple(o[1:, :, 3:] for o in frame_poses))
        self.frame_quats = tuple(torch.cat(o) for o in zip(*frame_quats))

        self.frame_quats = configurable_rotate_frames(
            self.frame_quats,
            enforce_z_down,
            enforce_z_up,
            add_init_ee_pose_as_frame,
            add_world_frame,
        )

        # Convert the reference frames and EE pose into homogeneous transforms.
        for i in range(self.n_trajs):
            frame_poses_i = frame_poses[i]
            n_frames, n_steps, len_quat = frame_poses_i.shape
            assert len_quat == 7

            frame_poses_i_b = frame_poses_i[:, :, :3]  # position
            frame_poses_i_q = frame_poses_i[:, :, 3:]  # quaternion

            # First frame in frame_poses_i_q is the EE pose, see line 198.
            # So, need to skip it and rotate the rest. Can achieve this by
            # setting with_init_ee_pose to True and with_world_frame to False.
            frame_poses_i_q = configurable_rotate_frames(
                frame_poses_i_q,
                enforce_z_down,
                enforce_z_up,
                with_init_ee_pose=True,
                with_world_frame=False,
            )

            assert quaternion_is_unit(frame_poses_i_q)
            frame_poses_i_q = standardize_quaternion(frame_poses_i_q)
            # assert quaternion_is_standard(frame_poses_i_q)

            f_b = frame_poses_i_b.reshape(-1, 3)
            f_A = torch.Tensor(quaternion_to_matrix(frame_poses_i_q.reshape(-1, 4)))

            world2frame = get_frame_transform_flat(f_A, f_b).reshape(
                n_frames, n_steps, 4, 4
            )
            world2frame_vel = get_frame_transform_flat(
                f_A, torch.zeros_like(f_b)  # Zero frame velocity
            ).reshape(n_frames, n_steps, 4, 4)
            frame2world = get_frame_transform_flat(f_A, f_b, invert=False).reshape(
                n_frames, n_steps, 4, 4
            )
            frame2world_vel = get_frame_transform_flat(
                f_A, torch.zeros_like(f_b), invert=False
            ).reshape(n_frames, n_steps, 4, 4)

            # Pop out the EE pose
            ee2world = frame2world[0, :, :, :].clone()
            self.ee_poses.append(ee2world)

            # Add world frame and or initial EE pose as frame
            id_frame = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(1, n_steps, 1, 1)
            ee_frame = ee2world[0].unsqueeze(0).unsqueeze(0).repeat(1, n_steps, 1, 1)
            ee_frame_vel = ee_frame.clone()
            ee_frame_vel[:, :, :3, 3] = 0  # Zero frame velocity

            list_world2frames = []
            list_frames2world = []
            list_world2frames_vel = []
            list_frames2world_vel = []
            if add_world_frame:
                list_world2frames.append(id_frame.clone())
                list_frames2world.append(id_frame.clone())
                list_world2frames_vel.append(id_frame.clone())
                list_frames2world_vel.append(id_frame.clone())
            if add_init_ee_pose_as_frame:
                list_world2frames.append(invert_homogenous_transform(ee_frame))
                list_frames2world.append(ee_frame)
                list_world2frames_vel.append(invert_homogenous_transform(ee_frame_vel))
                list_frames2world_vel.append(ee_frame_vel)

            list_frames2world.append(frame2world[1:, :, :, :])
            list_world2frames.append(world2frame[1:, :, :, :])
            list_frames2world_vel.append(frame2world_vel[1:, :, :, :])
            list_world2frames_vel.append(world2frame_vel[1:, :, :, :])

            frame2world = torch.cat(list_frames2world, dim=0)
            world2frame = torch.cat(list_world2frames, dim=0)
            frame2world_vel = torch.cat(list_frames2world_vel, dim=0)
            world2frame_vel = torch.cat(list_world2frames_vel, dim=0)

            self.world2frames.append(world2frame)
            self.frames2world.append(frame2world)
            self.world2frames_velocities.append(world2frame_vel)
            self.frames2world_velocities.append(frame2world_vel)

        self.world2frames = tuple(self.world2frames)
        self.world2frames_velocities = tuple(self.world2frames_velocities)
        self.frames2world = tuple(self.frames2world)
        self.frames2world_velocities = tuple(self.frames2world_velocities)
        self.ee_poses = tuple(self.ee_poses)
        # ee_poses_vel is the EE pose in world frame, but with zero bias (for velocity
        # frame transform). Analog to ee_quats for get_actions_world. Needed because
        # the EE frame transform is static over the trajectory.
        self.ee_poses_vel = tuple(t.clone() for t in self.ee_poses)
        for t in self.ee_poses_vel:
            t[:, :3, 3] = 0

        self.frame_names = []
        if add_world_frame:
            self.frame_names.append("world")
        if add_init_ee_pose_as_frame:
            self.frame_names.append("ee_init")
        self.frame_names += (
            [f"kp {i}" for i in range(n_obj_frames)]
            if frames_from_keypoints
            else [k for k, _ in trajectories[0].object_poses.items()]
        )
        self.frame_names = tuple(self.frame_names)

        self._ee_frame_idx = (
            1 if add_world_frame else 0 if add_init_ee_pose_as_frame else None
        )

        self.traj_lens = tuple(t.shape[0] for t in self.ee_poses)
        self.min_traj_len = min(self.traj_lens)
        self.max_traj_len = max(self.traj_lens)
        self.mean_traj_len = int(np.mean(self.traj_lens))

        actions = [o.action for o in trajectories]
        actions_hom = []
        actions_quats = []

        # Actions are EE-delta, EE-rotation (axis-angle) and gripper action.
        # Convert rotation into the homogeneous transforms as well to simplify
        # projections into local frames.
        if actions[0].shape[1] == 6:  # no gripper action
            self.gripper_actions = (None for _ in actions)
        else:
            self.gripper_actions = tuple(a[:, 6] for a in actions)
        for i in range(self.n_trajs):
            n_steps, len_action = actions[i].shape
            assert len_action <= 7
            a_A = axis_angle_to_matrix(actions[i][:, 3:6].reshape(-1, 3))
            a_b = actions[i][:, :3].reshape(-1, 3)
            actions_hom.append(
                get_frame_transform_flat(a_A, a_b, invert=False).reshape(n_steps, 4, 4)
            )

            actions_quats.append(
                axis_angle_to_quaternion(actions[i][:, 3:6].reshape(-1, 3))
            )

        self.ee_actions = tuple(actions_hom)
        self.ee_actions_quats = tuple(actions_quats)

        self.gripper_states = tuple(o.gripper_state for o in trajectories)

        if not add_world_frame:
            n_frames -= 1
        if add_init_ee_pose_as_frame:
            n_frames += 1

        self.n_frames = n_frames

        self.subsample_to_common_length()

        self.relative_start_time = 0
        self.relative_stop_time = 1
        self.relative_duration = 1

    def debug_trajs(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, self.n_frames + 1, figsize=(10, 10))
        for t in range(self.n_trajs):
            ax[0, 0].plot(self.ee_poses[t][:, 0, 3], color="r")
            ax[0, 0].plot(self.ee_poses[t][:, 1, 3], color="g")
            ax[0, 0].plot(self.ee_poses[t][:, 2, 3], color="b")
            ax[1, 0].plot(self.ee_quats[t][:, 0], color="r")
            ax[1, 0].plot(self.ee_quats[t][:, 1], color="g")
            ax[1, 0].plot(self.ee_quats[t][:, 2], color="b")
            ax[1, 0].plot(self.ee_quats[t][:, 3], color="y")

            for f in range(self.n_frames):
                ax[0, f + 1].plot(self.frames2world[t][f, :, 0, 3], color="r")
                ax[0, f + 1].plot(self.frames2world[t][f, :, 1, 3], color="g")
                ax[0, f + 1].plot(self.frames2world[t][f, :, 2, 3], color="b")
                ax[1, f + 1].plot(self.frame_quats[t][f, :, 0], color="r")
                ax[1, f + 1].plot(self.frame_quats[t][f, :, 1], color="g")
                ax[1, f + 1].plot(self.frame_quats[t][f, :, 2], color="b")
                ax[1, f + 1].plot(self.frame_quats[t][f, :, 3], color="y")

        plt.show()

        for t in range(self.n_trajs):
            if np.isnan(self.frames2world[t]).any():
                print(f"nan in world2frames in traj {t}")
            if np.isnan(self.world2frames_velocities[t]).any():
                print(f"nan in world2frame_velocities in traj {t}")
            if np.isnan(self.ee_poses[t]).any():
                print(f"nan in ee_poses in traj {t}")

    def _subsample(
        self,
        trajectories: Sequence[torch.Tensor] | torch.Tensor,
        indeces: list[Sequence[int]] | None = None,
        dim: int = 1,
    ) -> torch.Tensor:
        """
        Subsample a list of trajectories to a common length using the given
        indeces.
        If no indeces provided, defaults to the indeces in self._ss_idx,
        which are computed in subsample_to_common_length.

        Parameters
        ----------
        trajectories : list[torch.Tensor] or torch.Tensor
            The list of trajectories.
            Shape: (n_trajectories, n_frames, n_steps, ...)
        indeces : list[Iterable[int]], optional
            Subsampling indeces per trajectory.
            Shape: (n_trajectories, n_obserbations/ss_len)
        dim : int, optional
            The dimension along which to subsample.

        Returns
        -------
        torch.Tensor
            Stacked, subsampled trajectories.
        """

        if indeces is None:
            indeces = self._ss_idx

        # return torch.stack(
        #     [t[:, indeces[i], ...] for i, t in enumerate(trajectories)])

        subsampled = [
            torch.index_select(t, dim, indeces[i]) for i, t in enumerate(trajectories)
        ]

        return torch.stack(subsampled)

    def subsample_to_common_length(
        self, use_min: bool = False, use_max: bool = False
    ) -> None:
        """
        Subsample the trajectories to a common length.
        """
        target_len = (
            self.min_traj_len
            if use_min
            else self.max_traj_len if use_max else self.mean_traj_len
        )
        self.ss_len = target_len

        logger.info(
            "Subsampling to length {} using strategy {}-length.",
            target_len,
            "min" if use_min else "mean",
        )

        indeces = [
            get_idx_by_target_len(self.world2frames[i][0].shape[0], target_len)
            for i in range(self.n_trajs)
        ]

        self._ss_idx = indeces

        self.stacked_world2frames = self._subsample(self.world2frames, indeces)
        self.stacked_world2frame_velocities = self._subsample(
            self.world2frames_velocities, indeces
        )
        self.stacked_ee_actions = self._subsample(self.ee_actions, indeces, dim=0)
        self.stacked_ee_poses = self._subsample(self.ee_poses, indeces, dim=0)

        self.stacked_ee_quats = self._subsample(self.ee_quats, indeces, dim=0)
        self.stacked_frame_quats = self._subsample(self.frame_quats, indeces, dim=1)

        self.stacked_ee_actions_quats = self._subsample(
            self.ee_actions_quats, indeces, dim=0
        )

        self.stacked_gripper_actions = self._subsample(
            self.gripper_actions, indeces, dim=0
        )

        self.stacked_gripper_states = self._subsample(
            self.gripper_states, indeces, dim=0
        )

    @property
    def _n_gripper_states(self):
        """
        Return how often the gripper changes state (avg over all trajectories).
        """
        return (
            np.count_nonzero(
                self.stacked_gripper_states[:, :-1]
                != self.stacked_gripper_states[:, 1:],
                axis=1,
            )
            .mean()
            .astype(int)
            + 1
        )

    @property
    def _world2frames_fixed(self):
        """
        Get world2frames transform for fixed coordinate frames.
        """
        return torch.stack([w2f[:, 0:1, :, :] for w2f in self.world2frames])

    @property
    def _frames2world_fixed(self):
        """
        Get frames2world transform for fixed coordinate frames.
        """
        return torch.stack([f2w[:, 0:1, :, :] for f2w in self.frames2world])

    @property
    def _world2frames_velocities_fixed(self):
        return torch.stack([w2f[:, 0:1, :, :] for w2f in self.world2frames_velocities])

    @property
    def _frames2world_velocities_fixed(self):
        return torch.stack([f2w[:, 0:1, :, :] for f2w in self.frames2world_velocities])

    @property
    def _frame_origins_fixed(self):
        """
        Get the origin of the fixed frames. Ie the frame2world transform.
        As homogenous transform.
        """
        return self._frames2world_fixed
        # w2fs = self._world2frames_fixed
        # shape = w2fs.shape
        # return invert_homogenous_transform(w2fs.reshape(-1, 4, 4)).reshape(
        #     *shape)

    @property
    def _frame_origins_fixed_wquats(self):
        """
        Get the origin of the fixed frames. Ie the frame2world transform.
        As position + quaternion.
        """

        frame_pos = self._frame_origins_fixed[..., 0:3, 3].squeeze(2)
        frame_quats = self._frame_quats2world_fixed.squeeze(2)

        return torch.cat([frame_pos, frame_quats], dim=2)

    @property
    def _frame_quats2world_fixed(self):
        return torch.stack([f2w[:, 0:1, :] for f2w in self.frame_quats])

    @property
    def _frame_quats2world(self):
        return self.frame_quats

    @property
    def _world_quats2frame(self):
        return conjugate_quat(self._frame_quats2world)

    @property
    def _world_quats2frame_fixed(self):
        return conjugate_quat(self._frame_quats2world_fixed)

    @property
    def _frame_quats2world_velocities(self):
        logger.info("Assuming zero frame velocity. Should be fixed.")
        return self._frame_quats2world

    @property
    def _world_quats2frame_velocities(self):
        return conjugate_quat(self._frame_quats2world_velocities)

    @property
    def _frame_quats2world_velocities_fixed(self):
        return torch.stack(
            [f2w[:, 0:1, :] for f2w in self._frame_quats2world_velocities]
        )

    @property
    def _world_quats2frame_velocities_fixed(self):
        return conjugate_quat(self._frame_quats2world_velocities_fixed)

    @lru_cache
    def get_obs_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        as_quaternion: bool = False,
        skip_quat_dim: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Project the EE pose into all coordinate frames.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default False.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion.
        NOTE: the quaternion conversion is not tested for non-stacked/ss trajs.
        Probabaly makes problems.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
            The projected coordinate frames per trajectory and frame.
        """

        @lru_cache
        def _get_obs_per_frame(
            self, subsampled: bool = False, fixed_frames: bool = False
        ) -> torch.Tensor | tuple[torch.Tensor]:
            transforms = self._world2frames_fixed if fixed_frames else self.world2frames

            if subsampled and not fixed_frames:
                transforms = self._subsample(transforms)

            poses = self.stacked_ee_poses if subsampled else self.ee_poses

            return get_obs_per_frame(transforms, poses)

        obs = _get_obs_per_frame(self, subsampled, fixed_frames)

        if as_quaternion:
            # obs = hom_to_shift_quat(obs, skip_quat_dim=skip_quat_dim,
            #                         prefer_positives=True)
            pos = get_b_from_homogenous_transforms(obs)
            rot = self.get_quat_obs_per_frame(
                subsampled=subsampled,
                fixed_frames=fixed_frames,
                skip_quat_dim=skip_quat_dim,
            )

            # HACK: for obs, poses I have inconsistens dim orders.
            # In get_obs_per_frame this is fixed by a final permute.
            # Can't apply the same directly to get_quat_obs_per_frame because
            # it uses the list_or_tensor decorator. Need to properly fix this.
            if type(rot) is list:
                rot = [r.permute(1, 0, 2) for r in rot]
            elif type(rot) is tuple:
                rot = tuple(r.permute(1, 0, 2) for r in rot)
            else:
                rot = rot.permute(0, 2, 1, 3)

            obs = cat(pos, rot, dim=-1)

        return obs

    @lru_cache
    def get_action_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        as_quaternion: bool = False,
        skip_quat_dim: int | None = None,
        as_orientation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Project the EE action (pose delta) into all coordinate frames.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default False.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion.
        as_orientation: bool, optional
            If true, return only the orientation of the action, not the full
            action. For position that's the normalized vector, for rotation
            that's the rotation axis without angle.
        NOTE: the quaternion conversion is not tested for non-stacked/ss trajs.
        Probabaly makes problems.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
            The projected coordinate frames per trajectory and frame.
            Transform is 3+4 if as_quaternion, 3+3 if as_orientation, else 4x4.
        """

        @lru_cache
        def _get_action_per_frame(
            self,
            subsampled: bool = False,
            fixed_frames: bool = False,
            as_orientation: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor]:
            transforms = (
                self._world2frames_velocities_fixed
                if fixed_frames
                else self.world2frames_velocities
            )

            if subsampled and not fixed_frames:
                transforms = self._subsample(transforms)

            actions = self.get_actions_world(subsampled=subsampled)

            if as_orientation:
                directions, _ = self.get_pos_action_factorization(subsampled=subsampled)
                actions = set_b_in_homogenous_transforms(actions, directions)

            return get_obs_per_frame(transforms, actions)

        actions = _get_action_per_frame(self, subsampled, fixed_frames, as_orientation)

        if as_quaternion or as_orientation:
            pos = get_b_from_homogenous_transforms(actions)
            rot = self.get_quat_action_per_frame(
                subsampled=subsampled,
                fixed_frames=fixed_frames,
                skip_quat_dim=skip_quat_dim,
                as_orientation=as_orientation,
            )

            # HACK (see function above)
            if type(rot) in (list, tuple):
                rot = [r.permute(1, 0, 2) for r in rot]
            elif type(rot) is tuple:
                rot = tuple(r.permute(1, 0, 2) for r in rot)
            else:
                rot = rot.permute(0, 2, 1, 3)

            actions = cat(pos, rot, dim=-1)

        return actions

    @lru_cache
    def get_quat_obs_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        skip_quat_dim: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE rotation in all coordinate frames - as a quaternion.
        Bypasses the conversion to homogenous transforms, thus preventing
        possible discontinuities.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
            Analog to get_obs_per_frame, but rotation only.
        """

        transforms = (
            self._world_quats2frame_fixed if fixed_frames else self._world_quats2frame
        )

        if subsampled and not fixed_frames:
            transforms = self._subsample(transforms)

        poses = self.stacked_ee_quats if subsampled else self.ee_quats

        quats = get_quat_per_frame(transforms, poses)

        if skip_quat_dim is not None:
            quats = remove_quaternion_dim(quats, skip_quat_dim)

        return quats

    @lru_cache
    def get_quat_action_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        skip_quat_dim: int | None = None,
        as_orientation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE rotation action in all coordinate frames - as a quaternion.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default False. The ss strategy depends on the args passed to
            subsample_to_common_length.
        fixed_frame : bool, optional
            Wether to use the fixed coordinate frames, by default False.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
            Analog to get_action_per_frame, but rotation only.
        """
        assert not (as_orientation and skip_quat_dim is not None)

        transforms = (
            self._world_quats2frame_velocities_fixed
            if fixed_frames
            else self._world_quats2frame_velocities
        )

        if subsampled and not fixed_frames:
            transforms = self._subsample(transforms)

        if as_orientation:
            axis, _ = self.get_axis_and_angle_actions_world(subsampled=subsampled)
            transformed = get_orientation_per_frame(transforms, axis)
        else:
            actions = self.get_quat_actions_world(subsampled=subsampled)
            transformed = get_quat_per_frame(transforms, actions)

            if skip_quat_dim is not None:
                transformed = remove_quaternion_dim(transformed, skip_quat_dim)

        return transformed

    @lru_cache
    def get_actions_world(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE actions in world frame as homogenous transforms.
        (Raw actions are deltas in EE frame.)
        """
        actions_ee = self.ee_actions
        # NOTE: frames2world_velocities only contains the STATIC initial EE frame,
        # but we need the dynamic EE pose to transform the actions across the trajectory.
        # ee2world = tuple(
        #     traj[self._ee_frame_idx] for traj in self.frames2world_velocities
        # )
        ee2world = self.ee_poses_vel

        actions_world = tuple(trans @ act for trans, act in zip(ee2world, actions_ee))

        if subsampled:
            actions_world = self._subsample(actions_world, dim=0)

        return actions_world

    @lru_cache
    def get_quat_actions_world(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE rotation actions in world frame as quaternions.
        (Raw actions are deltas in EE frame.)
        """
        actions_ee = self.ee_actions_quats
        # NOTE: frame_quats only contains the STATIC initial EE frame, but we need the
        # dynamic EE pose to transform the actions across the trajectory.
        # ee2world = tuple(traj[self._ee_frame_idx] for traj in self.frame_quats)
        ee2world = self.ee_quats

        actions_world = tuple(
            quaternion_multiply(trans, act) for trans, act in zip(ee2world, actions_ee)
        )
        # NOTE: not rotating frames here helps the expected rots.
        # But why? Should transform to orientation before actions_to_world?
        # actions_world = actions_ee
        # How? Other function called get_orient_actions_per_frame.
        # In there, get actions_ee, convert to orientation, then transform with ee2world
        # using rotate_vector_by_quaternion

        if subsampled:
            actions_world = self._subsample(actions_world, dim=0)

        return actions_world

    @lru_cache
    def get_axis_and_angle_actions_world(
        self, subsampled: bool = False, zero_threshold: float = 0.001
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]:
        """
        Get the EE rotation actions in world frame as axis and angle.
        """
        actions_ee = self.ee_actions_quats
        ee2world = self.ee_quats

        axis_ee, angle_ee = quaternion_to_axis_and_angle(
            actions_ee, prefer_small_rots=PREFER_SMALLER_ROTS
        )

        axis_world = tuple(
            rotate_vector_by_quaternion(orient, trans)
            for orient, trans in zip(axis_ee, ee2world)
        )

        zero_idcs = tuple(torch.where(tr <= zero_threshold)[0] for tr in angle_ee)

        neutral_element = torch.Tensor(s2_id)  # .double()

        if axis_world[0].dtype != neutral_element.dtype:
            neutral_element = neutral_element.double()

        for i, idcs in enumerate(zero_idcs):
            axis_world[i][idcs] = neutral_element

        if subsampled:
            axis_world = self._subsample(axis_world, dim=0)
            angle_ee = self._subsample(angle_ee, dim=0)

        return axis_world, angle_ee

    @lru_cache
    def get_pos_action_factorization(
        self, subsampled: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the factorization of the action into direction and magnitude.
        """
        actions = self.get_actions_world(subsampled=subsampled)
        pos = get_b_from_homogenous_transforms(actions)
        dir, mag = translation_to_direction_and_magnitude(pos)

        return dir, mag

    @lru_cache
    def get_action_magnitude(
        self,
        subsampled: bool = False,
        position_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        _, pos_mag = self.get_pos_action_factorization(subsampled=subsampled)

        if position_only:
            if type(pos_mag) is torch.Tensor:
                ret = pos_mag.unsqueeze(-1)
            else:
                ret = tuple(p.unsqueeze(-1) for p in pos_mag)
        else:
            _, angle = self.get_axis_and_angle_actions_world(subsampled=subsampled)
            # quats = self.get_quat_actions_world(subsampled=subsampled)
            # _, angle = quaternion_to_axis_and_angle(
            #     quats, prefer_small_rots=PREFER_SMALLER_ROTS
            # )

            rot_sin = sin(angle)
            rot_cos = cos(angle)

            if type(pos_mag) is torch.Tensor:
                ret = torch.stack([pos_mag, rot_sin, rot_cos], dim=-1)
            else:
                ret = tuple(
                    torch.stack([p, rs, rc], dim=-1)
                    for p, rs, rc in zip(pos_mag, rot_sin, rot_cos)
                )

        return ret

    def get_gripper_action(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        return self.stacked_gripper_actions if subsampled else self.gripper_actions

    def get_gripper_state(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        if subsampled:
            data = self.stacked_gripper_states

            if len(data.shape) == 3:
                logger.info("Averaging gripper states over fingers.")
                data = data.mean(dim=2)
        else:
            data = self.gripper_states

            if len(data[0].shape) == 2:
                data = tuple(d.mean(dim=1) for d in data)

        return data

    @lru_cache
    def get_x_per_frame(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        flat: bool = False,
        pos_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = 0,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the position of the EE in all coordinate frames.
        Same as get_obs_per_frame, but only returns the position, not the full
        homogeneous transform.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default True.
        fixed_frames : bool, optional
            If true, uses the fixed coordinate frames. By default False.
        flat : bool, optional
            If True, flattens the output over the frames. By default False.
        pos_only : bool, optional
            If True, only returns the position, not the full transform.
            By default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default True.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion. By default 0.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
        """

        @lru_cache  # Nested function to cache result independtly of flat-arg.
        def _get_x_per_frame(self, subsampled=True, fixed_frames=False):
            if pos_only:
                obs = self.get_obs_per_frame(
                    subsampled, fixed_frames, False, skip_quat_dim
                )
                obs = get_b_from_homogenous_transforms(obs)
            else:
                obs = self.get_obs_per_frame(
                    subsampled, fixed_frames, as_quaternion, skip_quat_dim
                )

            return obs

        x = _get_x_per_frame(self, subsampled, fixed_frames)

        if flat:
            return x.reshape(self.n_trajs, self.ss_len, -1)
        else:
            return x

    @lru_cache
    def get_dx_per_frame(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        flat: bool = False,
        pos_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = 0,
        as_orientation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the position delta of the EE in all coordinate frames.
        Same as get_action_per_frame, but only returns the position, not the
        full homogeneous transform.

        Parameters
        ----------
        subsampled : bool, optional
            If true, returns the trajectories subsampled to the same length.
            By default True.
        fixed_frames : bool, optional
            If true, uses the fixed coordinate frames. By default False.
        flat : bool, optional
            If True, flattens the output over the frames. By default False.
        pos_only : bool, optional
            If True, only returns the position, not the full transform.
            By default False.
        as_quaternion : bool, optional
            If true, returns the rotation as a quaternion, by default True.
        skip_quat_dim : int, optional
            If not None and as_quaternion is True, pops the given dimension of
            the quaternion. By default 0.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor]
        """

        @lru_cache  # Nested function to cache result independtly of flat-arg.
        def _get_dx_per_frame(self, subsampled=True, fixed_frames=False):
            if pos_only:
                actions = self.get_action_per_frame(
                    subsampled=subsampled,
                    fixed_frames=fixed_frames,
                    as_quaternion=False,
                    skip_quat_dim=skip_quat_dim,
                    as_orientation=as_orientation,
                )
                if as_orientation:  # Return is flat: pos, rot
                    actions = actions[..., 0:3]
                else:  # Return is 4x4 matrix
                    actions = get_b_from_homogenous_transforms(actions)
            else:
                actions = self.get_action_per_frame(
                    subsampled=subsampled,
                    fixed_frames=fixed_frames,
                    as_quaternion=as_quaternion,
                    skip_quat_dim=skip_quat_dim,
                    as_orientation=as_orientation,
                )

            return actions

        dx = _get_dx_per_frame(self, subsampled, fixed_frames)

        if flat:
            return dx.reshape(self.n_trajs, self.ss_len, -1)
        else:
            return dx

    @lru_cache
    def get_per_frame_data(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        flat: bool = False,
        numpy: bool = False,
        pos_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = 0,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        action_as_orientation: bool = False,
        action_with_magnitude: bool = False,
        add_gripper_action: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor] | np.ndarray | tuple[np.ndarray]:
        """
        Get the stacked position and position delta of the EE in all
        coordinate frames.
        """

        if action_with_magnitude or add_gripper_action or add_time_dim:
            assert flat, "Can't stack global vars on non-flat frame data."

        @lru_cache
        def _get_per_frame_data(
            self,
            subsampled=True,
            fixed_frames=False,
            add_action_dim=False,
        ):
            x = self.get_x_per_frame(
                subsampled=subsampled,
                fixed_frames=fixed_frames,
                flat=False,
                pos_only=pos_only,
                as_quaternion=as_quaternion,
                skip_quat_dim=skip_quat_dim,
            )

            if add_action_dim:
                dx = self.get_dx_per_frame(
                    subsampled=subsampled,
                    fixed_frames=fixed_frames,
                    flat=False,
                    pos_only=pos_only,
                    as_quaternion=as_quaternion,
                    skip_quat_dim=skip_quat_dim,
                    as_orientation=action_as_orientation,
                )

                if subsampled:
                    obs = torch.cat((x, dx), dim=3)
                else:
                    obs = tuple(torch.cat((i, j), dim=2) for i, j in zip(x, dx))
            else:
                obs = x

            return obs

        obs = _get_per_frame_data(self, subsampled, fixed_frames, add_action_dim)

        if flat:
            if subsampled:
                obs = obs.reshape(self.n_trajs, self.ss_len, -1)
            else:
                obs = tuple(i.reshape(i.shape[0], -1) for i in obs)

            if add_time_dim:
                obs = add_time_dimension(
                    obs, start=self.relative_start_time, stop=self.relative_stop_time
                )

            if action_with_magnitude:
                dx_mag = self.get_action_magnitude(
                    subsampled=subsampled, position_only=pos_only
                )

                obs = cat(obs, dx_mag, dim=-1)

            if add_gripper_action:
                dx_grasp = unsqueeze(self.get_gripper_action(subsampled=subsampled), -1)

                obs = cat(obs, dx_grasp, dim=-1)

        else:
            if add_time_dim:
                logger.warning("add_time_dim is ignored when not flat.")

        if numpy:
            return to_numpy(obs)
        else:
            return obs

    @lru_cache
    def get_world_data(
        self,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        position_only: bool = False,
        as_quaternion: bool = True,
        skip_quat_dim: int | None = None,
        action_as_orientation: bool = False,
        action_with_magnitude: bool = False,
        add_gripper_action: bool = False,
        numpy: bool = False,
    ) -> tuple[torch.Tensor] | tuple[np.ndarray]:
        """
        Get the global trajectories. Ie the ee pose in the world frame, plus global dims,
        such as time and gripper action.
        For passing to a joint model in world frame, eg. for reconstruction.
        """
        assert not (skip_quat_dim and (not as_quaternion or action_as_orientation))

        obs = self.ee_poses_raw

        if position_only:
            obs = tuple(o[..., 0:3] for o in obs)

        elif not as_quaternion:
            raise NotImplementedError

        if skip_quat_dim is not None:
            obs = tuple(remove_quaternion_dim(o, skip_quat_dim) for o in obs)

        if add_action_dim:
            action_pos = tuple(
                a[..., 0:3, 3] for a in self.get_actions_world(subsampled=False)
            )

            if action_as_orientation:
                action_pos = tuple(
                    translation_to_direction_and_magnitude(a)[0] for a in action_pos
                )

                # import matplotlib.pyplot as plt
                # from riepybdlib.mappings import s2_log_e

                # tan_traj = tuple(s2_log_e(a.numpy()) for a in action_pos)

                # fig, ax = plt.subplots(1, 5)

                # for traj in action_pos:
                #     ax[0].plot(traj[:, 0], color="r")
                #     ax[0].plot(traj[:, 1], color="g")
                #     ax[0].plot(traj[:, 2], color="b")
                # for traj in tan_traj:
                #     ax[1].plot(traj[:, 0], color="r")
                #     ax[1].plot(traj[:, 1], color="g")
                # for traj in action_pos:
                #     ac = np.arccos(traj[:, 2])
                #     ac = np.where(traj[:, 2] < 0, ac - np.pi, ac)
                #     norm = np.linalg.norm(traj[:, :2], axis=1)
                #     xy = traj[:, :2]
                #     log = (xy.T * (ac / norm)).T
                #     ax[2].plot(ac, color="r")
                #     ax[2].plot(norm, color="g")
                #     ax[3].plot(ac / norm, color="k")
                #     ax[4].plot(log[:, 0], color="r")
                #     ax[4].plot(log[:, 1], color="g")

                #     print(xy[:, 0])
                # plt.show()
                # raise KeyboardInterrupt

            if position_only:
                obs = tuple(torch.cat([o, p], dim=1) for o, p in zip(obs, action_pos))
            else:
                if action_as_orientation:
                    action_rot, _ = self.get_axis_and_angle_actions_world(
                        subsampled=False
                    )
                    # action_rot = self.get_quat_actions_world(subsampled=False)
                    # action_rot = tuple(
                    #     quaternion_to_axis_and_angle(a, PREFER_SMALLER_ROTS)[0]
                    #     for a in action_rot
                    # )
                else:
                    action_rot = self.get_quat_actions_world(subsampled=False)

                    if skip_quat_dim is not None:
                        action_rot = tuple(
                            remove_quaternion_dim(a, skip_quat_dim) for a in action_rot
                        )

                obs = tuple(
                    torch.cat([o, p, q], dim=1)
                    for o, p, q in zip(obs, action_pos, action_rot)
                )

        if add_time_dim:
            obs = add_time_dimension(
                obs, start=self.relative_start_time, stop=self.relative_stop_time
            )

        if action_with_magnitude:
            dx_mag = self.get_action_magnitude(
                subsampled=False, position_only=position_only
            )
            obs = cat(obs, dx_mag, dim=-1)

        if add_gripper_action:
            dx_grasp = unsqueeze(self.get_gripper_action(subsampled=False), -1)
            obs = cat(obs, dx_grasp, dim=-1)

        if numpy:
            return to_numpy(obs)
        else:
            return obs

    @lru_cache
    def get_f_hom_per_frame_xdx(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        numpy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor] | np.ndarray or tuple[np.ndarray]:
        """
        Get the homogenous frames2world transforms per trajectory, frame and
        time, stacked for position + velocity.

        Parameters
        ----------
        subsampled : bool, optional
            Subsample in time to common length, by default True
        fixed_frames : bool, optional
            Use fixed coordinate frames per trajectory, by default False
        add_time_dim : bool, optional
            Add a time dimension with identity transform, by default False
        add_action_dim : bool, optional
            Generate transform for action dim as well, by default False
        numpy : bool, optional
            Convert the resulting tensor(s) to numpy, by default False

        Returns
        -------
        torch.Tensor or list[torch.Tensor] or np.ndarray or list[np.ndarray]
            The transforms. Result is a list of tensors/ndarrays if not
            subsampled to common length. Otherwise a single tensor/ndarray.
            Shape: (n_trajs, n_frames, n_observations, 7,7)
        """
        if add_time_dim:
            raise DeprecationWarning(
                "Should not add time dimension in homogenous frame transform. "
                "Handling of global dims (time, action magnitude and gripper) "
                "is now done jointly in the TPGMM implementation."
            )

        @lru_cache
        def _get_f_hom_xdx(
            self,
            subsampled: bool = True,
            fixed_frames: bool = False,
            add_time_dim: bool = False,
            add_action_dim: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor]:
            f_A = self.get_f_A_xdx(
                subsampled, fixed_frames, add_time_dim, add_action_dim, False
            )
            f_b = self.get_f_b_xdx(
                subsampled, fixed_frames, add_time_dim, add_action_dim, False
            )

            n_dim = f_b[0].shape[-1]
            n_frames = self.n_frames

            if type(f_A) is tuple:
                A_flat = [f.reshape(-1, n_dim, n_dim) for f in f_A]
                b_flat = [f.reshape(-1, n_dim) for f in f_b]
                hom = tuple(
                    homogenous_transform_from_rot_shift(A, b).reshape(
                        n_frames, -1, n_dim + 1, n_dim + 1
                    )
                    for A, b in zip(A_flat, b_flat)
                )
                return hom
            else:
                traj_len = 1 if fixed_frames else self.ss_len
                n_trajs = self.n_trajs

                A_flat = f_A.reshape(-1, n_dim, n_dim)
                b_flat = f_b.reshape(-1, n_dim)
                hom = homogenous_transform_from_rot_shift(A_flat, b_flat).reshape(
                    n_trajs, n_frames, traj_len, n_dim + 1, n_dim + 1
                )
                return hom

        f_hom = _get_f_hom_xdx(
            self, subsampled, fixed_frames, add_time_dim, add_action_dim
        )

        if numpy:
            return to_numpy(f_hom)
        else:
            return f_hom

    @lru_cache
    def get_f_A_xdx(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        add_time_dim: bool = False,
        add_action_dim: bool = True,
        numpy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor] | np.ndarray or tuple[np.ndarray]:
        """
        Get the rotation part of the frames2world transform, stacked for
        position + velocity.
        """

        @lru_cache
        def _get_f_A_xdx(
            self,
            subsampled: bool = True,
            fixed_frames: bool = False,
            add_time_dim: bool = False,
            add_action_dim: bool = True,
        ) -> torch.Tensor | tuple[torch.Tensor]:
            f_hom = self._frames2world_fixed if fixed_frames else self.frames2world

            f_A = get_R_from_homogenous_transforms(f_hom)

            if add_action_dim:
                df_hom = (
                    self._frames2world_velocities_fixed
                    if fixed_frames
                    else self.frames2world_velocities
                )

                df_A = get_R_from_homogenous_transforms(df_hom)

                # pos and vel trans should be the same, so just take one + kron
                if type(f_A) is tuple:
                    for f, df in zip(f_A, df_A):
                        assert torch.equal(f, df)
                    f_A = tuple(torch.kron(torch.eye(2), f) for f in f_A)

                    if subsampled:
                        f_A = torch.stack(
                            [f[:, i, :, :] for f, i in zip(f_A, self._ss_idx)]
                        )
                else:
                    assert torch.equal(f_A, df_A)
                    f_A = torch.kron(torch.eye(2), f_A)

                    assert not subsampled, "Subsam not needed for fixed frames"

            if add_time_dim:
                if type(f_A) is tuple:
                    t_As = [
                        torch.ones_like(f[:, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
                        for f in f_A
                    ]
                    f_A = tuple(batched_block_diag(t, f) for t, f in zip(t_As, f_A))
                else:
                    t_A = (
                        torch.ones_like(f_A[:, :, :, 0, 0]).unsqueeze(-1).unsqueeze(-1)
                    )

                    f_A = batched_block_diag(t_A, f_A)

            return f_A

        f_A = _get_f_A_xdx(self, subsampled, fixed_frames, add_time_dim, add_action_dim)

        if numpy:
            return to_numpy(f_A)
        else:
            return f_A

    @lru_cache
    def get_f_b_xdx(
        self,
        subsampled: bool = True,
        fixed_frames: bool = False,
        add_time_dim: bool = False,
        add_action_dim: bool = True,
        numpy: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor] | np.ndarray or tuple[np.ndarray]:
        """
        Get the translation part of the frames2world transform, stacked for
        position + velocity.
        """

        @lru_cache
        def _get_f_b_xdx(
            self,
            subsampled: bool = True,
            fixed_frames: bool = False,
            add_time_dim: bool = False,
            add_action_dim: bool = True,
        ) -> torch.Tensor | tuple[torch.Tensor]:
            f_hom = self._frames2world_fixed if fixed_frames else self.frames2world

            f_b = get_b_from_homogenous_transforms(f_hom)

            if add_action_dim:
                df_hom = (
                    self._frames2world_velocities_fixed
                    if fixed_frames
                    else self.frames2world_velocities
                )

                df_b = get_b_from_homogenous_transforms(df_hom)

                # Velocity transform should be zero. Sanity check and stack.
                # NOTE: for static frames this is unnecessary overhead. But
                # needed for extension to dynamic frames.
                if type(f_b) is tuple:
                    for df in df_b:
                        assert df.sum().data == 0
                    f_b = tuple(torch.cat((f, df), dim=2) for f, df in zip(f_b, df_b))

                    if subsampled:
                        f_b = torch.stack(
                            [f[:, i, :] for f, i in zip(f_b, self._ss_idx)]
                        )
                else:
                    f_b = torch.cat((f_b, df_b), dim=3)

                    assert not subsampled, "Subsam not needed for fixed frames"

            if add_time_dim:
                if type(f_b) is tuple:
                    t_bs = [torch.zeros_like(f[:, :, 0]).unsqueeze(-1) for f in f_b]
                    f_b = tuple(cat((t, f), dim=-1) for t, f in zip(t_bs, f_b))
                else:
                    t_b = torch.zeros_like(f_b[:, :, :, 0]).unsqueeze(-1)

                    f_b = torch.cat((t_b, f_b), dim=-1)

            return f_b

        f_b = _get_f_b_xdx(self, subsampled, fixed_frames, add_time_dim, add_action_dim)

        if numpy:
            return to_numpy(f_b)
        else:
            return f_b

    @lru_cache
    def get_f_quat_per_frame_xdx(
        self, subsampled: bool = True, fixed_frames: bool = False, numpy: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor] | np.ndarray or tuple[np.ndarray]:
        """
        Get the quaternion part of the frames2world transform, stacked for
        position + velocity.

        NOTE: this function bypasses the conversion to homogenous transforms
        thus ensuring that the quaternions are continuous.
        """

        @lru_cache
        def _get_f_quat_xdx(
            self, subsampled: bool = True, fixed_frames: bool = False
        ) -> torch.Tensor | tuple[torch.Tensor]:
            f_quat = (
                self._frame_quats2world_fixed
                if fixed_frames
                else self._frame_quats2world
            )

            df_quat = (
                self._frame_quats2world_velocities_fixed
                if fixed_frames
                else self._frame_quats2world_velocities
            )

            fdf_quat = stack(f_quat, df_quat, dim=-1)

            if fixed_frames:
                assert not subsampled, "Subsamp. not needed for fixed frames"
            elif subsampled:
                self._ss_stack(fdf_quat, dim=1)

            return fdf_quat

        f_quat = _get_f_quat_xdx(self, subsampled, fixed_frames)

        if numpy:
            return to_numpy(f_quat)
        else:
            return f_quat

    def _ss_stack(
        self,
        lot: list[torch.Tensor] | tuple[torch.Tensor],
        ss_dim: int = 1,
        stack_dim: int = 0,
        idx: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Subsample and stack a sequence of tensors.

        Parameters
        ----------
        lot : list[torch.Tensor] | tuple[torch.Tensor]
            The data to subsample and stack.
        ss_dim : int, optional
            The dimension along which to subsample, by default 1
        stack_dim : int, optional
            The dimension on which to stack, by default 0
        idx : list[int], optional
            The subsampling indices, by default uses self._ss_idx

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        if idx is None:
            idx = self._ss_idx

        if type(lot) in (list, tuple):
            # return torch.stack(
            #     [t[:, i, :, :] for t, i in zip(lot, idx)])
            return torch.stack(
                [torch.index_select(t, ss_dim, i) for t, i in zip(lot, idx)], stack_dim
            )
        else:
            raise NotImplementedError(
                f"Subsampling-stacking not implemented for type {type(lot)}"
            )

    @lru_cache
    def get_gmr_data(
        self,
        use_ss: bool = False,
        fix_frames: bool = False,
        position_only: bool = False,
        skip_quaternion_dim: int | None = None,
        add_time_dim: bool = False,
        add_action_dim: bool = False,
        add_gripper_action: bool = False,
        action_as_orientation: bool = False,
        action_with_magnitude: bool = False,
        numpy: bool = True,
    ) -> tuple[NDArrayOrNDArraySeq, NDArrayOrNDArraySeq, NDArrayOrNDArraySeq]:
        """
        Helper function to get the data prepared for GMR.

        Parameters
        ----------
        use_ss : bool
            Whether to use subsampled data, by default False
        fix_frames : bool
            Whether to use fixed frame positions per trajectory, by default
            False
        position_only : bool
            Wether to return only the position part of the data or pos + rot,
            by default False
        skip_quaternion_dim : int or None
            If not None, the dimension of the quaternion to remove, by default
            None
        add_time_dim : bool
            Whether to prepend a time dimension, by default False
        add_action_dim : bool
            Whether to append an action dimension, by default False
        add_gripper_action : bool
            Whether to append a gripper action dimension, by default False

        Returns
        -------
        np.ndarray or list[np.ndarray] x 3
            State + action data in world frame, frame translations, frame quaternions.
        """
        frame_trans = self.get_f_hom_per_frame_xdx(
            subsampled=use_ss,
            fixed_frames=fix_frames,
            add_time_dim=False,
            add_action_dim=add_action_dim,
            numpy=numpy,
        )
        frame_quat = self.get_f_quat_per_frame_xdx(
            subsampled=use_ss, fixed_frames=fix_frames, numpy=numpy
        )

        world_data = self.get_world_data(
            numpy=numpy,
            position_only=position_only,
            skip_quat_dim=skip_quaternion_dim,
            add_time_dim=add_time_dim,
            add_action_dim=add_action_dim,
            add_gripper_action=add_gripper_action,
            action_as_orientation=action_as_orientation,
            action_with_magnitude=action_with_magnitude,
        )

        return world_data, frame_trans, frame_quat

    def key(self):
        return self.__key()

    def __key(self):
        return self.meta_data

    def __hash__(self):
        hash = hashlib.sha1(repr(sorted(self.__key().items())).encode("utf-8"))

        return int(hash.hexdigest(), 16)

    def __eq__(self, other):
        if isinstance(other, Demos):
            return self.__key() == other.__key()
        return NotImplemented

    def segment(
        self,
        min_len: int,
        distance_based: bool,
        gripper_based: bool,
        velocity_based: bool,
        distance_threshold: float,
        repeat_first_step: int,
        repeat_final_step: int,
        fix_frames: bool,
        min_end_distance: int,
        velocity_threshold: float,
        gripper_threshold: float,
        max_idx_dist: int,
    ) -> tuple[Any, ...]:
        if sum((distance_based, gripper_based, velocity_based)) != 1:
            raise ValueError(
                "Can only segment based on exactly one strategy at a time."
            )

        if distance_based and not fix_frames:
            raise NotImplementedError(
                "Distance based segmentation would fail on dynamic frames."
                "TODO: combine with gripper based segmentation?"
            )

        if distance_based:
            trajs = self.get_per_frame_data(
                subsampled=False,
                fixed_frames=fix_frames,
                pos_only=True,
                add_action_dim=False,
                add_time_dim=False,
            )

            indeces_per_traj = self._get_segmentation_indeces_distance_based(
                trajs,
                min_len,
                distance_threshold,
                min_end_distance=min_end_distance,
            )
        elif gripper_based:
            trajs = self.get_gripper_state(subsampled=False)
            indeces_per_traj = self._get_segmentation_indeces_gripper_based(
                trajs,
                min_len,
                min_end_distance=min_end_distance,
                closed_threshold=gripper_threshold,
            )
        elif velocity_based:
            trajs = self.get_action_magnitude(subsampled=False, position_only=False)

            indeces_per_traj = self._get_segmentation_indeces_velocity_based(
                trajs,
                min_len,
                min_end_distance=min_end_distance,
                velocity_threshold=velocity_threshold,
                max_idx_dist=max_idx_dist,
            )
        else:
            raise NotImplementedError

        start_idcs = tuple((0,) + indeces for indeces in indeces_per_traj)
        stop_idcs = tuple(
            indeces + (stop,) for indeces, stop in zip(indeces_per_traj, self.traj_lens)
        )

        no_segments = len(start_idcs[0])

        segments = tuple(
            DemosSegment(
                self,
                tuple(t[i] for t in start_idcs),
                tuple(t[i] for t in stop_idcs),
                repeat_first_step=repeat_first_step,
                repeat_final_step=repeat_final_step,
                segment_no=i,
                segments_total=no_segments,
            )
            for i in range(no_segments)
        )

        return segments

    def _get_segmentation_indeces_velocity_based(
        self,
        trajs: tuple[torch.Tensor, ...],
        min_cluster_len: int,
        translation_only: bool = True,
        velocity_threshold: float = 0.005,
        min_end_distance: int = 10,
        max_idx_dist: int = 4,
        dbg: bool = True,
    ) -> tuple[tuple[int, ...], ...]:
        if translation_only:
            trajs = tuple(t[..., 0] for t in trajs)
        else:
            raise NotImplementedError

        if dbg:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1)
            for t, traj in enumerate(trajs):
                time_dim = np.linspace(0, 1, traj.shape[0])
                ax.plot(time_dim, traj, label=f"traj {t}", linewidth=0.5)
                ax.axhline(y=velocity_threshold, color="r")
            plt.legend()
            plt.show()

        traj_lens = tuple(t.shape[0] for t in trajs)

        stop_indeces = tuple(
            torch.argwhere(abs(t) < velocity_threshold).float().squeeze(1)
            for t in trajs
        )

        # print(stop_indeces)

        # split into segments if distance between indeces is larger than max_idx_dist
        idx_diff = tuple(traj[1:] - traj[:-1] for traj in stop_indeces)

        split_idx = tuple(
            torch.argwhere(diff > max_idx_dist).squeeze(1) + 1 for diff in idx_diff
        )

        stop_indeces_segmented = tuple(
            torch.tensor_split(c, idcs) for c, idcs in zip(stop_indeces, split_idx)
        )

        # for traj in stop_indeces_segmented:
        #     print([torch.mean(c).int() for c in traj])

        # filter out segments that are too short
        stop_indeces_segmented_filtered = tuple(
            tuple(cluster for cluster in clusters if len(cluster) >= min_cluster_len)
            for clusters in stop_indeces_segmented
        )

        # for traj in stop_indeces_segmented:
        #     print([torch.mean(c).int() for c in traj])

        segment_mean = tuple(
            tuple(torch.mean(c, dim=0).int() for c in cluster)
            for cluster in stop_indeces_segmented_filtered
        )

        # filter out clusters that are too close to the start or end of the trajectory
        segment_mean_filtered = [
            tuple(
                cluster
                for cluster in clusters
                if cluster > min_end_distance and cluster < traj_len - min_end_distance
            )
            for clusters, traj_len in zip(segment_mean, traj_lens)
        ]

        min_len = min([len(m) for m in segment_mean_filtered])
        # for i, traj in enumerate(segment_mean_filtered):
        #     print(torch.stack(traj), traj_lens[i])

        # for i, traj in enumerate(segment_mean_filtered):
        #     print(torch.stack(traj)/traj_lens[i])

        if dbg:
            n_trajs = len(trajs)
            fig, ax = plt.subplots(n_trajs, 1)
            for t, traj in enumerate(trajs):
                traj_len = traj.shape[0]
                time_dim = np.linspace(0, traj_len, traj_len)
                ax[t].plot(time_dim, traj, linewidth=0.5, c="gray")
                ax[t].axhline(y=velocity_threshold, color="r")
                for m in segment_mean_filtered[t]:
                    ax[t].axvline(x=m, color="g", linestyle="--")
            fig.set_size_inches(8, 6 * n_trajs)
            plt.show()

        for i, idcs in enumerate(segment_mean_filtered):
            if len(idcs) > min_len:
                logger.warning(
                    "Got different number of segmentation points for different trajectories.\n"
                    "Assuming it's because of settlement time.\n"
                    f"Popping {idcs[min_len:]} from traj {i} of len {traj_lens[i]}"
                )
                idcs = tuple(idcs[:min_len])
                segment_mean_filtered[i] = idcs

        return tuple(segment_mean_filtered)

    def _get_segmentation_indeces_distance_based(
        self,
        trajs: tuple[torch.Tensor, ...],
        min_cluster_len: int,
        distance_threshold: float,
        cluster_eps: float = 1.0,
        min_end_distance: int = 30,
        exlude_ee_frame: bool = True,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Segment the demos based on the distance between end-effector and frame.
        Returns tuple (over trajectories) of tuples of segmention indeces.
        """
        traj_lens = tuple(t.shape[0] for t in trajs)

        n_frames = self.n_frames

        if self.meta_data["add_init_ee_pose_as_frame"] and exlude_ee_frame:
            trajs = tuple(t[:, 1:] for t in trajs)
            n_frames -= 1

        # find all times when some frame  is closer to zero than the threshold
        sub_treshold = tuple(torch.norm(t, dim=2) < distance_threshold for t in trajs)
        indeces = tuple(
            tuple(torch.argwhere(t[:, f]).squeeze(1) for t in sub_treshold)
            for f in range(n_frames)
        )

        # print(indeces)

        indeces_filtered = tuple(
            tuple(
                tuple(
                    traj_idcs[
                        torch.logical_and(
                            traj_idcs > min_end_distance,
                            traj_idcs < traj_lens[t] - min_end_distance,
                        )
                    ]
                    for t, traj_idcs in enumerate(frame)
                )
                for frame in indeces
            )
        )

        # print(indeces_filtered)

        logger.warning("Assuming each frame is close to zero at most once.")
        # If a frame is close to zero at most once, we can just average the sub-
        # threshold indeces to get the center of the zero-segmens.
        logger.warning("Assuming static frames.")
        # Further assumes static frames, as for dynamic frames mean of the zero-
        # segment does not indicate a contact point.
        indeces_mean = tuple(
            tuple(
                torch.mean(traj_idcs.float()).int() if traj_idcs.numel() > 0 else None
                for traj_idcs in frame
            )
            for frame in indeces_filtered
        )

        for frame in indeces_mean:
            if frame[0] is None:
                assert all(traj is None for traj in frame)

        indeces_mean = tuple(frame for frame in indeces_mean if frame[0] is not None)

        # concatenate over frames and stack over trajectories
        # segmentation_indeces = torch.stack(
        #     tuple(
        #         torch.sort(torch.stack(frames, dim=0))[0] for frames in zip(*indeces_mean)
        #     )
        # )
        segmentation_indeces = tuple(
            tuple(sorted(itertools.chain(frames))) for frames in zip(*indeces_mean)
        )

        # print(segmentation_indeces)

        return segmentation_indeces

    def _get_segmentation_indeces_distance_based_old(
        self,
        trajs: tuple[torch.Tensor, ...],
        min_cluster_len: int,
        distance_threshold: float,
        cluster_eps: float = 1.0,
        min_end_distance: int = 30,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Segment the demos based on the distance between end-effector and frame.
        Returns tuple (over trajectories) of tuples of segmention indeces.
        """
        traj_lens = tuple(t.shape[0] for t in trajs)

        # print(traj_lens)

        # find all times when some frame  is closer to zero than the threshold
        sub_treshold = tuple(torch.norm(t, dim=2) < distance_threshold for t in trajs)
        indeces = tuple(
            tuple(torch.argwhere(t[:, f]) for t in sub_treshold)
            for f in range(self.n_frames)
        )
        # cluster the indeces to find the center of the zero-segments. More expensive
        # than just taking the average of consecutive indeces, but more robust in case
        # there's some hole in a zero-segment.
        dbs = DBSCAN(eps=cluster_eps, min_samples=min_cluster_len)
        cluster_labels = tuple(
            tuple(dbs.fit_predict(traj) if len(traj) > 0 else None for traj in frame)
            for frame in indeces
        )
        cluster_memb = tuple(
            tuple(
                (
                    None
                    if traj is None
                    else tuple(traj == l for l in range(traj.max() + 1))
                )
                for traj in frame
            )
            for frame in cluster_labels
        )

        # for frame in cluster_labels:
        #     for traj in frame:
        #         if traj is None:
        #             print(traj)
        #         else:
        #             for l in range(traj.max() + 1):
        #                 print(traj == l)

        frame_max = tuple(
            min([t.max() if t is not None else -1 for t in frame])
            for frame in cluster_labels
        )

        # print(frame_max)
        # print(cluster_labels)

        for f, frame in enumerate(cluster_labels):
            # frame_max = min([t.max() for t in frame if t is not None else -1])
            for t, traj in enumerate(frame):
                # print(f, t)
                # print(traj.max())
                # print(frame_max)
                # print(traj)
                if frame[0] is None:
                    assert traj is None
                else:
                    if not traj.max() == frame_max[f]:
                        logger.warning(
                            "Did not properly implement case where different trajectories "
                            "have different number of clusters. For now assuming the minimal "
                            "number is the correct (consensus) one."
                        )
        no_cluster_per_frame = tuple(
            frame_max[f] + 1 if frame[0] is not None else 0
            for f in range(len(cluster_labels))
        )

        means = tuple(
            tuple(
                tuple(
                    (
                        np.int(
                            np.mean(
                                cluster_memb[f][t][l] * indeces[f][t].squeeze(1).numpy()
                            )
                        )
                        if cluster_memb[f][t] is not None
                        else None
                    )
                    for l in range(no_cluster_per_frame[f])
                )
                for t in range(self.n_trajs)
            )
            for f in range(self.n_frames)
        )

        # filter out cluster means that are too close to start and end of the traj
        means_filtered = tuple(
            tuple(
                tuple(
                    m
                    for m in trajs
                    if m > min_end_distance and m < traj_lens[t] - min_end_distance
                )
                for t, trajs in enumerate(frame)
            )
            for frame in means
        )

        # print(means)
        # print(means_filtered)

        for frame in means_filtered:
            for traj in frame:
                assert len(traj) == len(frame[0]), (
                    "Got different number of filtered cluster means for the same frame"
                    " across trajectories."
                )

        # flatten over frames
        segmentation_indeces = tuple(
            tuple(
                sorted(
                    itertools.chain.from_iterable(
                        means_filtered[f][t] for f in range(self.n_frames)
                    )
                )
            )
            for t in range(self.n_trajs)
        )

        for traj in segmentation_indeces:
            assert len(traj) == len(segmentation_indeces[0])

        return segmentation_indeces

    def _get_segmentation_indeces_gripper_based(
        self,
        trajs: tuple[torch.Tensor, ...],
        min_len: int,
        max_idx_dist: int = 5,
        min_end_distance: int = 40,
        closed_threshold: float = 0.03,
    ) -> tuple[tuple[int, ...], ...]:
        """
        Segment the demos based on the gripper action.
        """
        traj_lens = tuple(t.shape[0] for t in trajs)

        # find indeces when gripper is closed, ie state is smaller than threshold
        closed = tuple(torch.argwhere(t < closed_threshold).squeeze(1) for t in trajs)

        # split into segments if distance between indeces is larger than max_idx_dist
        idx_diff = tuple(traj[1:] - traj[:-1] for traj in closed)
        split_idx = tuple(
            torch.argwhere(diff > max_idx_dist).squeeze(1) + 1 for diff in idx_diff
        )

        closed_segmented = tuple(
            torch.tensor_split(c, idcs) for c, idcs in zip(closed, split_idx)
        )

        # get the indeces of the first and last closed state
        closed_starts = tuple(
            tuple(c[0] for c in closed_segment) for closed_segment in closed_segmented
        )
        closed_ends = tuple(
            tuple(c[-1] for c in closed_segment) for closed_segment in closed_segmented
        )

        # filter out segments that are too short
        mask = tuple(
            tuple(
                end - start > min_len
                # and end < traj_len - min_end_distance
                # and start > min_end_distance
                for start, end in zip(traj_starts, traj_ends)
            )
            for traj_starts, traj_ends, traj_len in zip(
                closed_starts, closed_ends, traj_lens
            )
        )

        closed_starts_filtered = tuple(
            tuple(start for start, m in zip(traj_starts, mask) if m)
            for traj_starts, mask in zip(closed_starts, mask)
        )
        closed_ends_filtered = tuple(
            tuple(end for end, m in zip(traj_ends, mask) if m)
            for traj_ends, mask in zip(closed_ends, mask)
        )

        # concatenate starts and ends over trajectories
        segmentation_indeces = tuple(
            tuple(itertools.chain(traj_starts, traj_ends))
            for traj_starts, traj_ends in zip(
                closed_starts_filtered, closed_ends_filtered
            )
        )

        # filter out segmentation points that are too close to start or end of the traj
        segmentation_indeces = [
            tuple(
                idx
                for idx in traj
                if idx > min_end_distance and idx < traj_len - min_end_distance
            )
            for traj, traj_len in zip(segmentation_indeces, traj_lens)
        ]

        min_len = min(len(traj) for traj in segmentation_indeces)
        for i, idcs in enumerate(segmentation_indeces):
            if len(idcs) > min_len:
                logger.warning(
                    "Got different number of segmentation points for different trajectories.\n"
                    "Assuming it's because of settlement time.\n"
                    f"Popping {idcs[min_len:]} from traj {i} of len {traj_lens[i]}"
                )
                idcs = tuple(idcs[:min_len])
                segmentation_indeces[i] = idcs

        return tuple(segmentation_indeces)


class PartialFrameViewDemos(Demos):
    def __init__(self, full_demos: Demos, frame_indeces: list[int]):
        logger.info("Creating partial frame view of demos.", filter=False)
        self.full_demos = full_demos
        self.meta_data = full_demos.meta_data.copy()
        self.meta_data["FrameIndecies"] = frame_indeces
        self.frame_indecies = frame_indeces

        self.ee_poses = self.full_demos.ee_poses
        self.ee_poses_raw = self.full_demos.ee_poses_raw
        self.ee_poses_vel = self.full_demos.ee_poses_vel
        self.ee_quats = self.full_demos.ee_quats
        self.gripper_actions = self.full_demos.gripper_actions
        self.gripper_states = self.full_demos.gripper_states
        self.ee_actions = self.full_demos.ee_actions
        self.ee_actions_quats = self.full_demos.ee_actions_quats

        self.n_frames = len(frame_indeces)
        self.n_trajs = self.full_demos.n_trajs
        self.traj_lens = self.full_demos.traj_lens
        self.min_traj_len = self.full_demos.min_traj_len
        self.max_traj_len = self.full_demos.max_traj_len
        self.mean_traj_len = self.full_demos.mean_traj_len

        self.subsample_to_common_length()

        self.relative_start_time = self.full_demos.relative_start_time
        self.relative_stop_time = self.full_demos.relative_stop_time
        self.relative_duration = self.full_demos.relative_duration

    @property
    def frame_names(self):
        return tuple(self.full_demos.frame_names[i] for i in self.frame_indecies)

    # NOTE: not needed, bcs get_actions_world and get_quat_actions_world are overwritten
    # @property
    # def _ee_frame_idx(self):  # is always before object frames, so no need to index
    #     return self.full_demos._ee_frame_idx

    @property
    def world2frames(self):
        return self._get_indexed(self.full_demos.world2frames)

    @property
    def world2frames_velocities(self):
        return self._get_indexed(self.full_demos.world2frames_velocities)

    @property
    def frames2world(self):
        return self._get_indexed(self.full_demos.frames2world)

    @property
    def frames2world_velocities(self):
        return self._get_indexed(self.full_demos.frames2world_velocities)

    @property
    def frame_quats(self):
        return self._get_indexed(self.full_demos.frame_quats)

    @property
    def _world2frames_fixed(self):
        return self._get_indexed_stacked(self.full_demos._world2frames_fixed)

    @property
    def _frames2world_fixed(self):
        return self._get_indexed_stacked(self.full_demos._frames2world_fixed)

    @property
    def _world2frames_velocities_fixed(self):
        return self._get_indexed_stacked(self.full_demos._world2frames_velocities_fixed)

    @property
    def _frames2world_velocities_fixed(self):
        return self._get_indexed_stacked(self.full_demos._frames2world_velocities_fixed)

    @property
    def _frame_origins_fixed(self):
        return self._get_indexed_stacked(self.full_demos._frame_origins_fixed)

    @property
    def _frame_origins_fixed_wquats(self):
        return self._get_indexed_stacked(self.full_demos._frame_origins_fixed_wquats)

    @property
    def _frame_quats2world_fixed(self):
        return self._get_indexed_stacked(self.full_demos._frame_quats2world_fixed)

    @property
    def _world_quats2frame(self):
        return self._get_indexed(self.full_demos._world_quats2frame)

    @property
    def _world_quats2frame_fixed(self):
        return self._get_indexed_stacked(self.full_demos._world_quats2frame_fixed)

    @property
    def _frame_quats2world_velocities(self):
        return self._get_indexed(self.full_demos._frame_quats2world_velocities)

    @property
    def _world_quats2frame_velocities(self):
        return self._get_indexed(self.full_demos._world_quats2frame_velocities)

    @property
    def _frame_quats2world_velocities_fixed(self):
        return self._get_indexed_stacked(
            self.full_demos._frame_quats2world_velocities_fixed
        )

    @property
    def _world_quats2frame_velocities_fixed(self):
        return self._get_indexed_stacked(
            self.full_demos._world_quats2frame_velocities_fixed
        )

    def _get_indexed(
        self, data: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Index the full data to get the partial frame view.
        data can be a tensor in which the first dim is the frame dim, or a tuple
        of tensors (over trajectories) with the same property.
        """
        return list_index_first_tensor_dim(data, self.frame_indecies)

    def _get_indexed_stacked(self, data: torch.Tensor):
        """
        Equivalent of _get_indexed for stacked data, eg fixed frames.
        _get_indexed assumes that data is either a tensor in which the first
        dim is the frame dim, or a tuple of tensors (over trajectories) with
        the same property.
        Tensors that are stacked over trajectories have the trajectory dim in
        front, so need to index at second position.
        """
        return data[:, self.frame_indecies]

    @lru_cache
    def get_action_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        as_quaternion: bool = False,
        skip_quat_dim: int | None = None,
        as_orientation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Overwriting this method to allow consistent action direction across segments.
        In DemosSegment, don't have access to previous actions, so need to get the full
        action per frame from the full demos.
        Need to to this here in PartialFrameViewDemos, too, as I first segment and then
        create the partial frame view.
        """

        full_traj = self.full_demos.get_action_per_frame(
            subsampled=subsampled,
            fixed_frames=fixed_frames,
            as_quaternion=as_quaternion,
            skip_quat_dim=skip_quat_dim,
            as_orientation=as_orientation,
        )

        frame_indexed = tuple(self._get_indexed_stacked(t) for t in full_traj)

        if subsampled:
            frame_indexed = torch.stack(frame_indexed, dim=0)

        return frame_indexed

    @lru_cache
    def get_pos_action_factorization(
        self, subsampled: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the factorization of the action into direction and magnitude.
        Overwriting this method for consistent direction across segments.
        """
        return self.full_demos.get_pos_action_factorization(subsampled=subsampled)

    @lru_cache
    def get_actions_world(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE actions in world frame as homogenous transforms.
        (Raw actions are deltas in EE frame.)
        Overwriting this method as the calculation is based on the EE frame, which might
        not be present in the partial frame view.
        """
        return self.full_demos.get_actions_world(subsampled=subsampled)

    @lru_cache
    def get_quat_actions_world(
        self, subsampled: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor]:
        """
        Get the EE rotation actions in world frame as quaternions.
        (Raw actions are deltas in EE frame.)
        """
        return self.full_demos.get_quat_actions_world(subsampled=subsampled)

    @lru_cache
    def get_axis_and_angle_actions_world(
        self, subsampled: bool = False
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]:
        """
        Get the EE rotation actions in world frame as axis and angle.
        """
        return self.full_demos.get_axis_and_angle_actions_world(subsampled=subsampled)


class DemosSegment(Demos):
    def __init__(
        self,
        full_demos: Demos,
        start_idcs: int,
        stop_idcs: int,
        repeat_first_step: int = 0,
        repeat_final_step: int = 0,
        segment_no: int = 0,
        segments_total: int = 1,
    ):
        logger.info("Creating segement of demos.", filter=False)
        self.full_demos = full_demos
        self.meta_data = full_demos.meta_data.copy()
        self.meta_data["start_idcs"] = start_idcs
        self.meta_data["stop_idcs"] = stop_idcs
        self.meta_data["repeat_first_step"] = repeat_first_step
        self.meta_data["repeat_final_step"] = repeat_final_step

        self.start_idcs = start_idcs
        self.stop_idcs = stop_idcs
        self.repeat_first_step = repeat_first_step
        self.repeat_final_step = repeat_final_step

        self.n_frames = self.full_demos.n_frames
        self.n_trajs = self.full_demos.n_trajs
        self.frame_names = self.full_demos.frame_names

        self.world2frames = self._get_indexed(self.full_demos.world2frames, 1)
        self.world2frames_velocities = self._get_indexed(
            self.full_demos.world2frames_velocities, 1
        )
        self.frames2world = self._get_indexed(self.full_demos.frames2world, 1)
        self.frames2world_velocities = self._get_indexed(
            self.full_demos.frames2world_velocities, 1
        )
        self.frame_quats = self._get_indexed(self.full_demos.frame_quats, 1)

        self.ee_poses = self._get_indexed(self.full_demos.ee_poses, 0)
        self.ee_poses_vel = self._get_indexed(self.full_demos.ee_poses_vel, 0)
        self.ee_poses_raw = self._get_indexed(self.full_demos.ee_poses_raw, 0)
        self.ee_quats = self._get_indexed(self.full_demos.ee_quats, 0)
        self.gripper_actions = self._get_indexed(self.full_demos.gripper_actions, 0)
        self.gripper_states = self._get_indexed(self.full_demos.gripper_states, 0)
        self.ee_actions = self._get_indexed(self.full_demos.ee_actions, 0)
        self.ee_actions_quats = self._get_indexed(self.full_demos.ee_actions_quats, 0)

        self.traj_lens = tuple(t.shape[0] for t in self.ee_poses)
        self.min_traj_len = min(self.traj_lens)
        self.max_traj_len = max(self.traj_lens)
        self.mean_traj_len = int(np.mean(self.traj_lens))

        self.subsample_to_common_length()

        # TODO: make this per trajectory? Ie remove the mean?
        # when using max ss, should also take max here.
        padded_start_idc = (
            np.mean(start_idcs) + (repeat_first_step + repeat_first_step) * segment_no
        )
        padded_stop_idcs = np.mean(stop_idcs) + (
            repeat_first_step + repeat_first_step
        ) * (segment_no + 1)
        padded_total_len = (
            self.full_demos.ss_len
            + (repeat_first_step + repeat_final_step) * segments_total
        )
        self.relative_start_time = padded_start_idc / padded_total_len
        self.relative_stop_time = padded_stop_idcs / padded_total_len
        self.relative_duration = self.relative_stop_time - self.relative_start_time

    @property
    def _ee_frame_idx(self):
        # Segment are created from full demos, not PartialFrameViews. Index is the same
        return self.full_demos._ee_frame_idx

    @property
    def _world2frames_fixed(self):  # need to ensure consistency with fixed frames
        return self.full_demos._world2frames_fixed

    @property
    def _frames2world_fixed(self):
        return self.full_demos._frames2world_fixed

    @property
    def _world2frames_velocities_fixed(self):
        return self.full_demos._world2frames_velocities_fixed

    @property
    def _frames2world_velocities_fixed(self):
        return self.full_demos._frames2world_velocities_fixed

    @property
    def _frame_origins_fixed(self):
        return self.full_demos._frame_origins_fixed

    @property
    def _frame_origins_fixed_wquats(self):
        return self.full_demos._frame_origins_fixed_wquats

    @property
    def _frame_quats2world_fixed(self):
        return self.full_demos._frame_quats2world_fixed

    @property
    def _world_quats2frame_fixed(self):
        return self.full_demos._world_quats2frame_fixed

    @property
    def _frame_quats2world_velocities_fixed(self):
        return self.full_demos._frame_quats2world_velocities_fixed

    @property
    def _world_quats2frame_velocities_fixed(self):
        return self.full_demos._world_quats2frame_velocities_fixed

    def _get_indexed(
        self,
        data: torch.Tensor | tuple[torch.Tensor, ...],
        dim: int,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Index the full demos data to get the segment data along the time dimension.
        dim depends on whether the data is per frame or not.
        """
        dims = tuple(dim for _ in range(self.n_trajs))
        sliced_view = slice_tensors(data, dims, self.start_idcs, self.stop_idcs)

        if self.repeat_first_step > 0:
            sliced_view = repeat_step(
                sliced_view, idx=0, repeat=self.repeat_first_step, dim=dim
            )

        if self.repeat_final_step > 0:
            sliced_view = repeat_step(
                sliced_view, idx=-1, repeat=self.repeat_final_step, dim=dim
            )

        return sliced_view

    @lru_cache
    def get_action_per_frame(
        self,
        subsampled: bool = False,
        fixed_frames: bool = False,
        as_quaternion: bool = False,
        skip_quat_dim: int | None = None,
        as_orientation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Overwriting this method to allow consistent action direction across segments.
        In DemosSegment, don't have access to previous actions, so need to get the full
        action per frame from the full demos.
        """

        full_traj = self.full_demos.get_action_per_frame(
            subsampled=False,
            fixed_frames=fixed_frames,
            as_quaternion=as_quaternion,
            skip_quat_dim=skip_quat_dim,
            as_orientation=as_orientation,
        )

        time_indexed = self._get_indexed(full_traj, 0)

        if subsampled:
            time_indexed = self._ss_stack(time_indexed, idx=self._ss_idx, ss_dim=0)

        return time_indexed

    @lru_cache
    def get_pos_action_factorization(
        self, subsampled: bool = False
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...]
    ]:
        """
        Get the factorization of the action into direction and magnitude.
        Overwriting this method for consistent direction across segments.
        """
        orient, mag = self.full_demos.get_pos_action_factorization(subsampled=False)

        orient_time_indexed = self._get_indexed(orient, 0)
        mag_time_indexed = self._get_indexed(mag, 0)

        if subsampled:
            orient_time_indexed = self._ss_stack(
                orient_time_indexed, idx=self._ss_idx, ss_dim=0
            )
            mag_time_indexed = self._ss_stack(
                mag_time_indexed, idx=self._ss_idx, ss_dim=0
            )

        return orient_time_indexed, mag_time_indexed

    @lru_cache
    def get_axis_and_angle_actions_world(
        self, subsampled: bool = False
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], torch.Tensor | tuple[torch.Tensor]]:
        """
        Get the EE rotation actions in world frame as axis and angle.
        Overwriting this method for consistent handling of zero rotations across segments.
        """
        axis, angle = self.full_demos.get_axis_and_angle_actions_world(subsampled=False)

        axis_time_indexed = self._get_indexed(axis, 0)
        angle_time_indexed = self._get_indexed(angle, 0)

        if subsampled:
            axis_time_indexed = self._ss_stack(
                axis_time_indexed, idx=self._ss_idx, ss_dim=0
            )
            angle_time_indexed = self._ss_stack(
                angle_time_indexed, idx=self._ss_idx, ss_dim=0
            )

        return axis_time_indexed, angle_time_indexed


@list_or_tensor
def repeat_step(
    tens: torch.Tensor, idx: int, repeat: int, dim: int = 0
) -> torch.Tensor:
    """
    Repeat an indexed component of a tensor along a given dimension.
    """
    assert repeat > 0
    assert dim < len(tens.shape)
    assert idx < tens.shape[dim]

    if idx == -1:
        idx = tens.shape[dim] - 1

    repeating_step = tens.select(dim, idx).unsqueeze(dim)
    repeats = [1] * len(tens.shape)
    repeats[dim] = repeat

    before = tens.narrow(dim, 0, idx)
    after = tens.narrow(dim, idx + 1, tens.shape[dim] - idx - 1)

    return torch.cat((before, repeating_step.repeat(repeats), after), dim=dim)


@list_or_tensor
def list_index_first_tensor_dim(tens: torch.Tensor, idx: list[int]):
    return tens[idx]


@list_or_tensor_mult_args
def slice_tensors(
    tensors: torch.Tensor, dim: int, start: int, stop: int
) -> torch.Tensor:
    """
    Variant of slice_any_tensor_dim that can expects a Sequence of dims and indeces
    when getting a Sequence of tensors.
    """
    return slice_any_tensor_dim(tensors, dim, start, stop).clone()


def get_frame_transform(
    A: torch.Tensor, b: torch.Tensor, invert: bool = True
) -> torch.Tensor:
    """
    Get the frame transform from the given rotation and shift.
    When invert is True, returns world2frame, else frame2world.
    """
    assert len(A.shape) == 3
    assert A.shape[:-2] == b.shape[:-1]
    n_frames, n_steps, _ = b.shape
    A_flat = A.reshape(-1, 3, 3)
    b_flat = b.reshape(-1, 3)

    frame_transform = get_frame_transform_flat(A_flat, b_flat, invert=invert)

    frame_transform = frame_transform.reshape(n_frames, n_steps, 4, 4)

    return frame_transform


def get_frame_transform_flat(
    A: torch.Tensor, b: torch.Tensor, invert: bool = True
) -> torch.Tensor:
    frame_transform = homogenous_transform_from_rot_shift(A, b)

    if invert:
        frame_transform = invert_homogenous_transform(frame_transform)

    return frame_transform


def get_obs_per_frame(
    frame_transform: TensorOrTensorSeq, ee_poses: TensorOrTensorSeq
) -> TensorOrTensorSeq:
    """
    Transform the observations from world to a given reference frame.
    Both frame_transform and EE_poses are given as homogeneous transforms.
    """
    if type(ee_poses) in (list, tuple):
        x_per_frame = []
        for ft, eep in zip(frame_transform, ee_poses):
            n_steps, _, _ = eep.shape
            n_frames, _, _, _ = ft.shape

            if ft.shape[1] == 1:  # static frame case, repeat
                ft = ft.repeat(1, n_steps, 1, 1)
            # Add frame dimension
            eep = eep.unsqueeze(0).repeat(n_frames, 1, 1, 1)

            x_per_frame.append(
                (ft.reshape(-1, 4, 4) @ eep.reshape(-1, 4, 4))
                .reshape(n_frames, n_steps, 4, 4)
                .permute(1, 0, 2, 3)
            )

        if type(ee_poses) == tuple:
            x_per_frame = tuple(x_per_frame)

        return x_per_frame

    else:
        assert type(ee_poses) is torch.Tensor
        assert type(frame_transform) is torch.Tensor

        n_trajs, n_steps, _, _ = ee_poses.shape
        _, n_frames, _, _, _ = frame_transform.shape

        if frame_transform.shape[2] == 1:  # static frame case, repeat
            frame_transform = frame_transform.repeat(1, 1, n_steps, 1, 1)
        # Add frame dimension
        ee_poses = ee_poses.unsqueeze(1).repeat(1, n_frames, 1, 1, 1)

        x_per_frame = (
            frame_transform.reshape(-1, 4, 4) @ ee_poses.reshape(-1, 4, 4)
        ).reshape(n_trajs, n_frames, n_steps, 4, 4)

        return x_per_frame.permute(0, 2, 1, 3, 4)


def shape_cast_quaternion_transforms(
    frame_quats: torch.Tensor, ee_quats: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast the quaternion frame transforms and ee rotations to the same shape. Ie.
    - add a frame dimension to the ee rotations, and match the number of frames.
    - repeat the frame transforms over time if they are static.
    """
    trans_shape = frame_quats.shape
    ee_shape = ee_quats.shape

    # The trajectory dim is part of both (if we stacked/subsampled) or none.
    # So, we can (almost) ignore it.
    w_traj_dim = True if len(trans_shape) == 4 else False

    if w_traj_dim:
        assert len(ee_shape) == 3
        trans_shape = trans_shape[1:]
        ee_shape = ee_shape[1:]

    n_frames, n_steps_trans, _ = trans_shape
    n_steps_ee, _ = ee_shape

    if n_steps_trans == 1:  # static frame case, repeat
        repeats = (
            (1, 1, n_steps_ee, 1) if w_traj_dim else (1, n_steps_ee, 1)
        )  # repeats depend on trajectory dim
        frame_quats = frame_quats.repeat(*repeats)

    # Add frame dimension
    repeats = (1, n_frames, 1, 1) if w_traj_dim else (n_frames, 1, 1)
    ee_quats = ee_quats.unsqueeze(-3).repeat(*repeats)

    return frame_quats, ee_quats


@list_or_tensor_mult_args
def get_quat_per_frame(
    frame_quats: torch.Tensor, ee_quats: torch.Tensor
) -> torch.Tensor:
    """
    Analog to get_obs_per_frame, but for quaternions.
    """
    frame_quats, ee_quats = shape_cast_quaternion_transforms(frame_quats, ee_quats)

    return quaternion_lot_multiply(frame_quats, ee_quats)


@list_or_tensor_mult_args
def get_orientation_per_frame(
    frame_quats: torch.Tensor, ee_orient: torch.Tensor
) -> torch.Tensor:
    """
    Analog to get_obs_per_frame, but for orientations (S2 vectors).
    """
    frame_quats, ee_orient = shape_cast_quaternion_transforms(frame_quats, ee_orient)

    return rotate_vector_by_quaternion(ee_orient, frame_quats)


def get_idx_by_target_len(traj_len: int, target_len: int) -> torch.Tensor:
    # Copied from tapas_gmm.dataset.trajectory.py, but changed the supersampling strategy
    # for shorter trajs. Analogous to subsampling now, instead of padding with
    # the last frame. And using torch instead of numpy.
    indeces = torch.linspace(start=0, end=traj_len - 1, steps=target_len)
    indeces = torch.round(indeces).int()
    return indeces


@list_or_tensor
def add_time_dimension(
    lot: torch.Tensor, start: int = 0, stop: int = 1
) -> torch.Tensor:
    # Assumes tensors have 2 or 3 dimensions and that time is the third last
    # dimension. Ie (n_traj), n_steps, n_dim.
    # NOTE: dropped the n_frame dimension, as adding time per frame does not
    # make sense.
    n_time_steps = lot.shape[-2]

    time = torch.linspace(start, stop, n_time_steps).unsqueeze(-1)

    if len(lot.shape) == 3:
        time = time.unsqueeze(0).repeat(lot.shape[0], 1, 1)

    return torch.cat([time, lot], dim=-1)
