from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from omegaconf import MISSING

from tapas_gmm.utils.geometry_np import (
    axis_angle_to_quaternion,
    euler_angles_to_axis_angle,
    quaternion_to_axis_angle,
)
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.robot_trajectory import RobotTrajectory


def squash(array, order=20):
    # map to [-1, 1], but more linear than tanh
    return np.sign(array) * np.power(
        np.tanh(np.power(np.power(array, 2), order / 2)), 1 / order
    )


class GripperPlot:
    def __init__(self, headless):
        self.headless = headless

        if headless:
            return

        self.displayed_gripper = 0.9

        self.fig = plt.figure()

        ax = self.fig.add_subplot(111)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)

        horizontal_patch = plt.Rectangle((-1, 0), 2, 0.6)
        self.left_patch = plt.Rectangle((-0.9, -1), 0.4, 1, color="black")
        self.right_patch = plt.Rectangle((0.5, -1), 0.4, 1, color="black")

        ax.add_patch(horizontal_patch)
        ax.add_patch(self.left_patch)
        ax.add_patch(self.right_patch)

        self.fig.canvas.draw()

        plt.show(block=False)

        plt.pause(0.1)

        for _ in range(2):
            self.set_data(0)
            plt.pause(0.1)
            self.set_data(1)
            plt.pause(0.1)

    def set_data(self, new_state: float) -> None:
        """
        Set the gripper plot to the given gripper state.

        Parameters
        ----------
        new_state : float
            The new gripper state, either 0.9 (open) or -0.9 (closed).
        """
        if self.headless or self.displayed_gripper == new_state:
            return

        if new_state == 0.9:
            self.displayed_gripper = 0.9
            self.left_patch.set_xy((-0.9, -1))
            self.right_patch.set_xy((0.5, -1))

        elif new_state == -0.9:
            self.displayed_gripper = -0.9
            self.left_patch.set_xy((-0.4, -1))
            self.right_patch.set_xy((0, -1))

        self.fig.canvas.draw()

        plt.pause(0.01)

        return

    def reset(self) -> None:
        self.set_data(1)


@dataclass(kw_only=True)
class BaseEnvironmentConfig:
    task: str

    cameras: tuple[str, ...]
    # dict-value is numerical 7tuple, but OmegaConf cant handel that annotation
    camera_pose: dict[str, Any]
    image_size: tuple[int, int]

    static: bool
    headless: bool

    scale_action: bool
    delay_gripper: bool

    gripper_plot: bool

    env_type: str = MISSING  # Set in subclasses
    postprocess_actions: bool = True


class BaseEnvironment(ABC):
    def __init__(self, config: BaseEnvironmentConfig, **kwargs) -> None:
        self.config = config

        self.do_postprocess_actions = config.postprocess_actions
        self.do_scale_action = config.scale_action
        self.do_delay_gripper = config.delay_gripper

        image_size = config.image_size

        self.image_height, self.image_width = image_size

        self.gripper_plot = GripperPlot(not config.gripper_plot)
        self.gripper_open = 0.9

        self.queue_length = 4
        self.gripper_deque = deque([0.9] * self.queue_length, maxlen=self.queue_length)

        # Scale actions from [-1,1] to the actual action space, ie transtions
        # in meters etc.
        self._delta_pos_scale = 0.01
        self._delta_angle_scale = 0.04

    def reset(self) -> None:
        """
        Reset the environment to a new episode. In the BaseEnvironment, this
        only resets the gripper plot.
        """
        if self.gripper_plot:
            self.gripper_plot.reset()

        self.gripper_open = 0.9
        self.gripper_deque = deque([0.9] * self.queue_length, maxlen=self.queue_length)

    def step(self, action: np.ndarray) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Simple wrapper around _step, that provides the kwargs for
        postprocessing from self.config.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.

        Returns
        -------
        tuple[SceneObservation, float, bool, dict]
            The observation, reward, done flag and info dict.
        """

        return self._step(
            action,
            postprocess=self.do_postprocess_actions,
            delay_gripper=self.do_delay_gripper,
            scale_action=self.do_scale_action,
        )

    @abstractmethod
    def _step(
        self,
        action: np.ndarray,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
    ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        """
        raise NotImplementedError

    def render(self) -> None:
        """
        Explicit render function for simulated environments. Currently only
        used when replaying the downloaded demos from ManiSkill.
        In the BaseEnvironment, this does nothing.
        """
        return

    @abstractmethod
    def close(self):
        """
        Gracefully close the environment.
        """
        raise NotImplementedError

    def postprocess_action(
        self,
        action: np.ndarray,
        prediction_is_quat: bool = True,
        prediction_is_euler: bool = False,
        scale_action: bool = False,
        delay_gripper: bool = False,
        trans_scale: float | None = None,
        rot_scale: float | None = None,
    ) -> np.ndarray:
        """
        Postprocess the action predicted by the policy for the action space of
        the environment.

        Parameters
        ----------
        action : np.ndarray[(7,), np.float32]
            Original action predicted by the policy
            Concatenation of delta_position, delta rotation, gripper action.
            Delta rotation can be axis angle (NN) or Quaternion (GMM).
        scale_action : bool, optional
            Whether to scale the position and rotation action, by default False
        delay_gripper : bool, optional
            Whether to delay the gripper, by default False
        trans_scale : float | None, optional
            The scaling for the translation action,
            by default self._delta_pos_scale
        rot_scale : float | None, optional
            The scaling for the rotation (applied to the Euler angles),
            by default self._delta_angle_scale

        Returns
        -------
        np.ndarray
            _description_
        """
        if trans_scale is None:
            trans_scale = self._delta_pos_scale
        if rot_scale is None:
            rot_scale = self._delta_angle_scale

        rot_dim = 4 if prediction_is_quat else 3

        delta_position, delta_rot, gripper = np.split(action, [3, 3 + rot_dim])

        if prediction_is_quat:
            delta_rot_axis_angle = quaternion_to_axis_angle(delta_rot)
        elif prediction_is_euler:
            delta_rot_axis_angle = euler_angles_to_axis_angle(delta_rot)
        else:
            delta_rot_axis_angle = delta_rot

        # print(prediction_is_quat, prediction_is_euler, delta_rot_axis_angle)

        if scale_action:
            delta_position = delta_position * trans_scale
            delta_rot_axis_angle = delta_rot_axis_angle * rot_scale

        # print("scaled", delta_position, delta_rot_axis_angle)

        delta_rot_quat = axis_angle_to_quaternion(delta_rot_axis_angle)

        delta_rot_env = self.postprocess_quat_action(delta_rot_quat)

        if delay_gripper:
            gripper = [self.delay_gripper(gripper)]

        return np.concatenate((delta_position, delta_rot_env, gripper))

    def postprocess_quat_to_quat(self, quat: np.ndarray) -> np.ndarray:
        """
        Postprocess quat action to match environment quat convention.
        RLBench uses real last, everything else uses real first.
        """
        return quat

    def delay_gripper(self, gripper_action: float) -> float:
        """
        Delay gripper action, ie. only open/close gripper if the gripper action
        is constant over the last few steps (self.queue_length).
        Useful to smooth noisy gripper actions predicted by neural networks.

        Parameters
        ----------
        gripper_action : float
            The current gripper action.

        Returns
        -------
        float
            The smoothed/delayed gripper action.
        """
        if gripper_action >= 0.0:
            gripper_action = 0.9
        elif gripper_action < 0.0:
            gripper_action = -0.9

        self.gripper_plot.set_data(gripper_action)
        self.gripper_deque.append(gripper_action)

        if all(x == 0.9 for x in self.gripper_deque):
            self.gripper_open = 1

        elif all(x == -0.9 for x in self.gripper_deque):
            self.gripper_open = 0

        return self.gripper_open

    def update_visualization(self, info: dict) -> None:
        """
        Update additional visualizations hosted by the environment.
        Currently only used by franka env. And even that is a bit of a hack.

        TODO Should move this outside of the environment class or use it more
        consistently.
        """
        pass

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Postprocess the rotation action predicted by the policy for the action
        space of the environment.

        Parameters
        ----------
        quaternion : np.ndarray
            Quaternion action predicted by the policy (real first).

        Returns
        -------
        np.ndarray
            The rotation action in the action space of the environment.
        """
        raise NotImplementedError

    def get_inverse_kinematics(self, target_pose: np.ndarray) -> np.ndarray:
        """
        Get the inverse kinematics of the robot for a given target pose.

        Overwrite this method in the environment subclass to ensure that quaternion conventions are respected and that the proper ik model is used.

        Parameters
        ----------
        target_pose : np.ndarray
            The target pose.

        Returns
        -------
        np.ndarray
            The inverse kinematics for the target pose.
        """
        raise NotImplementedError

    def _get_state(self) -> Any:
        logger.info("No state to save for this environment.")

    def _set_state(self, state: Any) -> None:
        logger.info("No state to restore for this environment.")

    def _get_action_mode(self) -> Any:
        logger.info("No need to save action mode for this environment.")

    def _set_action_mode(self, action_mode: Any) -> None:
        logger.info("No need to restore action mode for this environment.")

    def reset_joint_pose(self) -> None:
        raise NotImplementedError("Need to implement in child class.")

    def recover_from_errors(self) -> None:
        """
        Error recovery for real robot.
        """
        pass

    def publish_frames(self, frame_trans: np.ndarray, frame_quats: np.ndarray) -> None:
        """
        Publishes the given frames to the tf tree for visualization in rviz.
        Used for visualizing keypoint frames.
        """
        pass

    def publish_path(self, trajectory: RobotTrajectory) -> None:
        """
        Publishes the given trajectory to the path topic for visualization in rviz.
        """
        pass

    def propose_update_visualization(self, info: dict) -> None:
        """
        Propose an update to the visualization. Currently only used by franka env.
        """
        pass


class RestoreEnvState:
    def __init__(self, env):
        self.env = env

    def __enter__(self):
        self.state = self.env._get_state()

    def __exit__(self, type, value, traceback):
        self.env._set_state(self.state)


class RestoreActionMode:
    def __init__(self, env):
        self.env = env

    def __enter__(self):
        self.action_mode = self.env._get_action_mode()

    def __exit__(self, type, value, traceback):
        self.env._set_action_mode(self.action_mode)
