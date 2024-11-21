from typing import Sequence

import numpy as np
import torch
from loguru import logger

from tapas_gmm.utils.geometry_torch import (
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    quaternion_pose_diff,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from tapas_gmm.utils.observation import SceneObservation


def pad_list(source, target_len=2):
    return source + [None] * (target_len - len(source))


def subsample_image(img):
    return torch.nn.functional.interpolate(
        img, size=(256, 256), mode="bilinear", align_corners=True
    )


def clamp_translation(position, delta, limits):
    goal = position + delta * 0.25

    clipped = np.clip(goal, *limits)

    if goal != clipped:
        logger.info("Limiting translation {} to workspace limits.", goal)

    return clipped


def compute_ee_delta(current: torch.Tensor, next: torch.Tensor) -> torch.Tensor:
    """
    Computes the relative end effector pose change between two batches of poses.
    Returns as position delta and axis-angle rotation delta.
    """
    curr_f_b, curr_quat = current[:, :3], current[:, 3:]
    next_f_b, next_quat = next[:, :3], next[:, 3:]

    curr_f_A = quaternion_to_matrix(curr_quat)
    curr_hom = homogenous_transform_from_rot_shift(curr_f_A, curr_f_b)

    next_f_A = quaternion_to_matrix(next_quat)
    next_hom = homogenous_transform_from_rot_shift(next_f_A, next_f_b)

    world2curr = invert_homogenous_transform(curr_hom)

    delta_hom = world2curr @ next_hom

    delta_pos = delta_hom[:, :3, 3]

    delta_quat = quaternion_pose_diff(curr_quat, next_quat)

    delta_aa = quaternion_to_axis_angle(delta_quat)

    return torch.concatenate((delta_pos, delta_aa), dim=1)


def reconstruct_actions(obs: Sequence[SceneObservation]) -> None:
    """
    When collecting demos via handguiding, the actions are not correctly recorded.
    Reconstructs them inplace from the end effector poses.
    """
    for traj in obs:
        gripper_action = traj.action[:, -1:]  # gripper action is reported correctly
        ee_poses = traj.ee_pose
        next_ee_poses = torch.cat((traj.ee_pose[1:], traj.ee_pose[-1:]), dim=0)
        ee_delta = compute_ee_delta(ee_poses, next_ee_poses)
        traj.action = torch.cat((ee_delta, gripper_action), dim=1)


def reconstruct_actions_obs(obs: SceneObservation) -> torch.Tensor:
    """
    Batch-obs variant of reconstruct_actions, returns instead of inplace.
    """
    assert len(obs.action.shape) == 2, f"Expected 2 dims, got shape {obs.action.shape}."
    gripper_action = obs.action[:, -1:]  # gripper action is reported correctly
    ee_poses = obs.ee_pose
    next_ee_poses = torch.cat((obs.ee_pose[1:], obs.ee_pose[-1:]), dim=0)
    ee_delta = compute_ee_delta(ee_poses, next_ee_poses)
    action = torch.cat((ee_delta, gripper_action), dim=1)

    return action
