# slightly adapted from ManiSkill2 to add observations to replay memory

from abc import ABC

import numpy as np
import sapien.core as sapien
import torch
from mani_skill2.agents.base_controller import CombinedController
from mani_skill2.agents.controllers import PDEEPoseController, PDJointPosController
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.trajectory.replay_trajectory import (
    clip_and_scale_action,
    delta_pose_to_pd_ee_delta,
    qpos_to_pd_joint_delta_pos,
    qpos_to_pd_joint_target_delta_pos,
    qpos_to_pd_joint_vel,
)
from tqdm.auto import tqdm

from tapas_gmm.env.mani_skill import ManiSkillEnv


def from_pd_joint_pos(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
    replay_memory=None,
):
    if "ee" in output_mode:
        return from_pd_joint_pos_to_ee(**locals())

    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        ori_env.step(ori_action)
        flag = True

        for _ in range(2):
            if output_mode == "pd_joint_delta_pos":
                arm_action = qpos_to_pd_joint_delta_pos(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            elif output_mode == "pd_joint_target_delta_pos":
                arm_action = qpos_to_pd_joint_target_delta_pos(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            elif output_mode == "pd_joint_vel":
                arm_action = qpos_to_pd_joint_vel(
                    controller.controllers["arm"], ori_action_dict["arm"]
                )
            else:
                raise NotImplementedError(
                    f"Does not support converting pd_joint_pos to {output_mode}"
                )

            # Assume normalized action
            if np.max(np.abs(arm_action)) > 1 + 1e-3:
                if verbose:
                    tqdm.write(f"Arm action is clipped: {arm_action}")
                flag = False
            arm_action = np.clip(arm_action, -1, 1)
            output_action_dict["arm"] = arm_action

            output_action = controller.from_action_dict(output_action_dict)
            obs, reward, done, info = env.step(output_action)
            obs.action = torch.from_numpy(output_action)
            obs.feedback = torch.Tensor([1])
            if replay_memory is not None:
                replay_memory.add_observation(obs)
            if render:
                env.render()

            if flag:
                break

    return info


def from_pd_joint_pos_to_ee(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
    replay_memory=None,
):
    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    pos_only = not ("pose" in output_mode)
    target_mode = "target" in output_mode

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller

    # NOTE(jigu): We need to track the end-effector pose in the original env,
    # given target joint positions instead of current joint positions.
    # Thus, we need to compute forward kinematics
    pin_model = ori_controller.articulation.create_pinocchio_model()
    ori_arm_controller: PDJointPosController = ori_controller.controllers["arm"]
    arm_controller: PDEEPoseController = controller.controllers["arm"]
    assert arm_controller.config.frame == "ee"
    ee_link: sapien.Link = arm_controller.ee_link

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        # Keep the joint positions with all DoF
        full_qpos = ori_controller.articulation.get_qpos()

        ori_env.step(ori_action)

        # Use target joint positions for arm only
        full_qpos[ori_arm_controller.joint_indices] = ori_arm_controller._target_qpos
        pin_model.compute_forward_kinematics(full_qpos)
        target_ee_pose = pin_model.get_link_pose(arm_controller.ee_link_idx)

        flag = True

        for _ in range(2):
            if target_mode:
                prev_ee_pose_at_base = arm_controller._target_pose
            else:
                base_pose = arm_controller.articulation.pose
                prev_ee_pose_at_base = base_pose.inv() * ee_link.pose

            ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose
            arm_action = delta_pose_to_pd_ee_delta(
                arm_controller, ee_pose_at_ee, pos_only=pos_only
            )

            if (np.abs(arm_action[:3])).max() > 1:  # position clipping
                if verbose:
                    tqdm.write(f"Position action is clipped: {arm_action[:3]}")
                arm_action[:3] = np.clip(arm_action[:3], -1, 1)
                flag = False
            if not pos_only:
                if np.linalg.norm(arm_action[3:]) > 1:  # rotation clipping
                    if verbose:
                        tqdm.write(f"Rotation action is clipped: {arm_action[3:]}")
                    arm_action[3:] = arm_action[3:] / np.linalg.norm(arm_action[3:])
                    flag = False

            output_action_dict["arm"] = arm_action
            output_action = controller.from_action_dict(output_action_dict)

            obs, reward, done, info = env.step(output_action)
            obs.action = torch.from_numpy(output_action)
            obs.feedback = torch.Tensor([1])
            if replay_memory is not None:
                replay_memory.add_observation(obs)
            if render:
                env.render()

            if flag:
                break

    return info


def from_pd_joint_delta_pos(
    output_mode,
    ori_actions,
    ori_env: BaseEnv,
    env: BaseEnv,
    render=False,
    pbar=None,
    verbose=False,
):
    n = len(ori_actions)
    if pbar is not None:
        pbar.reset(total=n)

    ori_controller: CombinedController = ori_env.agent.controller
    controller: CombinedController = env.agent.controller
    ori_arm_controller: PDJointPosController = ori_controller.controllers["arm"]

    assert output_mode == "pd_joint_pos", output_mode
    assert ori_arm_controller.config.normalize_action
    low, high = ori_arm_controller.config.lower, ori_arm_controller.config.upper

    info = {}

    for t in range(n):
        if pbar is not None:
            pbar.update()

        ori_action = ori_actions[t]
        ori_action_dict = ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        prev_arm_qpos = ori_arm_controller.qpos
        delta_qpos = clip_and_scale_action(ori_action_dict["arm"], low, high)
        arm_action = prev_arm_qpos + delta_qpos

        ori_env.step(ori_action)

        output_action_dict["arm"] = arm_action
        output_action = controller.from_action_dict(output_action_dict)
        _, _, _, info = env.step(output_action)

        if render:
            env.render()

    return info


class ActionTranslator(ABC):
    def __init__(self, target_env: ManiSkillEnv, ori_env):
        self.ori_env = ori_env
        self.target_env = target_env
        self.ori_controller: CombinedController = ori_env.agent.controller
        self.controller: CombinedController = target_env.agent.controller

        # NOTE(jigu): We need to track the end-effector pose in the original env,
        # given target joint positions instead of current joint positions.
        # Thus, we need to compute forward kinematics
        self.pin_model = self.ori_controller.articulation.create_pinocchio_model()
        self.ori_arm_controller: PDJointPosController = self.ori_controller.controllers[
            "arm"
        ]
        self.arm_controller: PDEEPoseController = self.controller.controllers["arm"]
        assert self.arm_controller.config.frame == "ee"
        self.ee_link: sapien.Link = self.arm_controller.ee_link

    def translate(
        self,
        ori_action: torch.Tensor,
        gripper_action: torch.Tensor | None,
        target_mode: bool = False,
        pos_only: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


class PDJointPos2EETranslator(ActionTranslator):
    def translate(
        self,
        ori_action: torch.Tensor,
        gripper_action: torch.Tensor | None,
        target_mode: bool = False,
        pos_only: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        # repeat gripper state
        ori_action = torch.cat((ori_action, ori_action[-1:])).numpy()

        ori_action_dict = self.ori_controller.to_action_dict(ori_action)
        output_action_dict = ori_action_dict.copy()  # do not in-place modify

        # Keep the joint positions with all DoF
        full_qpos = self.ori_controller.articulation.get_qpos()

        self.ori_env.step(ori_action)

        # Use target joint positions for arm only
        full_qpos[self.ori_arm_controller.joint_indices] = (
            self.ori_arm_controller._target_qpos
        )
        self.pin_model.compute_forward_kinematics(full_qpos)
        target_ee_pose = self.pin_model.get_link_pose(self.arm_controller.ee_link_idx)

        flag = True

        for _ in range(2):
            if target_mode:
                prev_ee_pose_at_base = self.arm_controller._target_pose
            else:
                base_pose = self.arm_controller.articulation.pose
                prev_ee_pose_at_base = base_pose.inv() * self.ee_link.pose

            ee_pose_at_ee = prev_ee_pose_at_base.inv() * target_ee_pose

            arm_action = delta_pose_to_pd_ee_delta(
                self.arm_controller, ee_pose_at_ee, pos_only=pos_only
            )

            if (np.abs(arm_action[:3])).max() > 1:  # position clipping
                if verbose:
                    tqdm.write(f"Position action is clipped: {arm_action[:3]}")
                arm_action[:3] = np.clip(arm_action[:3], -1, 1)
                flag = False
            if not pos_only:
                if np.linalg.norm(arm_action[3:]) > 1:  # rotation clipping
                    if verbose:
                        tqdm.write(f"Rotation action is clipped: {arm_action[3:]}")
                    arm_action[3:] = arm_action[3:] / np.linalg.norm(arm_action[3:])
                    flag = False

            output_action_dict["arm"] = arm_action
            output_action = self.controller.from_action_dict(output_action_dict)

            if flag:
                break

        output_action = torch.from_numpy(output_action)

        # add correct gripper action, as motion planner does not include it
        if gripper_action is not None:
            output_action = torch.cat((output_action[:-1], gripper_action))

        # self.target_env.step(output_action)

        return output_action
