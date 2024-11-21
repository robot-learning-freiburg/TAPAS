import os
from dataclasses import dataclass
from functools import wraps
from typing import Any

import mani_skill2
import numpy as np
import torch
from loguru import logger
from pyrep.const import RenderMode
from pyrep.errors import ConfigurationPathError, IKError
from rlbench.action_modes.action_mode import ActionMode, MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    EndEffectorPoseViaIK,
    EndEffectorPoseViaPlanning,
)
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.exceptions import InvalidActionError
from rlbench.backend.observation import Observation as RLBenchObservation
from rlbench.demo import Demo
from rlbench.environment import Environment as RLBenchInternalEnvironment
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.task_environment import TaskEnvironment
from rlbench.tasks import (
    CloseJar,
    CloseMicrowave,
    InsertOntoSquarePeg,
    LightBulbIn,
    MeatOnGrill,
    OpenDrawer,
    PhoneOnBase,
    PickAndLift,
    PlaceCups,
    PlaceShapeInShapeSorter,
    PushButton,
    PushButtons,
    PutGroceriesInCupboard,
    PutItemInDrawer,
    PutMoneyInSafe,
    PutRubbishInBin,
    ReachAndDrag,
    SlideBlockToTarget,
    StackBlocks,
    StackCups,
    StackWine,
    SweepToDustpan,
    TakeLidOffSaucepan,
    TurnTap,
)

from tapas_gmm.env import Environment
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.utils.geometry_np import (
    conjugate_quat,
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    quat_real_first_to_real_last,
    quat_real_last_to_real_first,
    quaternion_diff,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_pose_diff,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]


task_switch = {
    "CloseMicrowave": CloseMicrowave,
    "TakeLidOffSaucepan": TakeLidOffSaucepan,
    "PhoneOnBase": PhoneOnBase,
    "PutRubbishInBin": PutRubbishInBin,
    "StackWine": StackWine,
    "PickAndLift": PickAndLift,
    "PushButton": PushButton,
    "OpenDrawer": OpenDrawer,
    "TurnTap": TurnTap,
    "PushButton": PushButton,
    "PushButtons": PushButtons,
    "SweepToDustpan": SweepToDustpan,
    "SlideBlockToTarget": SlideBlockToTarget,
    "InsertOntoSquarePeg": InsertOntoSquarePeg,
    "MeatOnGrill": MeatOnGrill,
    "PlaceShapeInShapeSorter": PlaceShapeInShapeSorter,
    "PutGroceriesInCupboard": PutGroceriesInCupboard,
    "PutMoneyInSafe": PutMoneyInSafe,
    "CloseJar": CloseJar,
    "ReachAndDrag": ReachAndDrag,
    "LightBulbIn": LightBulbIn,
    "StackCups": StackCups,
    "PlaceCups": PlaceCups,
    "PutItemInDrawer": PutItemInDrawer,
    "StackBlocks": StackBlocks,
}


world_pos_action_mode = MoveArmThenGripper(
    arm_action_mode=EndEffectorPoseViaIK(
        absolute_mode=False,  # False
        frame="world",  # end effector
    ),
    gripper_action_mode=Discrete(),
)


@dataclass(kw_only=True)
class RLBenchEnvironmentConfig(BaseEnvironmentConfig):
    action_mode: Any = None
    env_type: Environment = Environment.RLBENCH

    # RLBench changed. AMs are no longer Enum and Omega can't store classes as vals
    planning_action_mode: bool = False
    absolute_action_mode: bool = False
    action_frame: str = "end effector"

    postprocess_actions: bool = True
    background: str | None = None
    model_ids: tuple[str, ...] | None = None


class RLBenchEnvironment(BaseEnvironment):
    def __init__(self, config: RLBenchEnvironmentConfig, **kwargs):
        super().__init__(config)

        self.cameras = config.cameras

        assert set(self.cameras).issubset(
            {"left_shoulder", "right_shoulder", "wrist", "overhead", "front"}
        )

        left_shoulder_on = "left_shoulder" in self.cameras
        right_shoulder_on = "right_shoulder" in self.cameras
        wrist_on = "wrist" in self.cameras
        overhead_on = "overhead" in self.cameras
        front_on = "front" in self.cameras

        render_mode = RenderMode.OPENGL
        image_size = (self.image_height, self.image_width)

        obs_config = ObservationConfig(
            left_shoulder_camera=CameraConfig(
                rgb=left_shoulder_on,
                depth=left_shoulder_on,
                mask=left_shoulder_on,
                render_mode=render_mode,
                depth_in_meters=True,
                image_size=image_size,
                point_cloud=False,
            ),
            right_shoulder_camera=CameraConfig(
                rgb=right_shoulder_on,
                depth=right_shoulder_on,
                mask=right_shoulder_on,
                render_mode=render_mode,
                depth_in_meters=True,
                image_size=image_size,
                point_cloud=False,
            ),
            front_camera=CameraConfig(
                rgb=front_on,
                depth=front_on,
                mask=front_on,
                render_mode=render_mode,
                depth_in_meters=True,
                image_size=image_size,
                point_cloud=False,
            ),
            wrist_camera=CameraConfig(
                rgb=wrist_on,
                depth=wrist_on,
                mask=wrist_on,
                render_mode=render_mode,
                depth_in_meters=True,
                image_size=image_size,
                point_cloud=False,
            ),
            overhead_camera=CameraConfig(
                rgb=overhead_on,
                depth=overhead_on,
                mask=overhead_on,
                render_mode=render_mode,
                depth_in_meters=True,
                image_size=image_size,
                point_cloud=False,
            ),
            joint_positions=True,
            joint_velocities=True,
            joint_forces=False,
            gripper_pose=True,
            gripper_matrix=True,
            task_low_dim_state=True,
        )

        self.planning_action_mode = config.planning_action_mode

        self.launch_simulation_env(config, obs_config)

        self.setup_camera_controls(config)

    @property
    def _move_group(self) -> str:
        """
        For using the mplib Planner, eg for TOPP(RA) in gmm policy.
        """
        return "panda_hand_tcp"

    @property
    def _urdf_path(self) -> str:
        """
        For using the mplib Planner, eg for TOPP(RA) in gmm policy.
        """
        return f"{mani_skill2.PACKAGE_ASSET_DIR}/descriptions/panda_v2.urdf"

    @property
    def _srdf_path(self) -> str:
        return f"{mani_skill2.PACKAGE_ASSET_DIR}/descriptions/panda_v2.srdf"

    def launch_simulation_env(
        self, config: RLBenchEnvironmentConfig, obs_config: ObservationConfig
    ) -> None:
        # sphere policy uses custom action mode, ABS_EE_POSE_PLAN_WORLD_FRAME
        # for others: everything like in parent class
        if config.action_mode is None:
            config.action_mode = (
                # instead of TOPPRA  EndEffectorPoseViaIK
                EndEffectorPoseViaPlanning
                if self.planning_action_mode
                else EndEffectorPoseViaIK
            )

        if (
            config.action_mode is EndEffectorPoseViaIK
            and not config.postprocess_actions
        ):
            logger.warning(
                "Using default action mode without action "
                "postprocessing. Is that intended?"
            )
        action_mode = MoveArmThenGripper(
            arm_action_mode=config.action_mode(
                absolute_mode=self.config.absolute_action_mode,
                frame=self.config.action_frame,
            ),
            gripper_action_mode=Discrete(),
        )

        self.env = RLBenchInternalEnvironment(
            action_mode=action_mode,
            obs_config=obs_config,
            static_positions=config.static,
            headless=config.headless,
        )

        self.env.launch()

        self.task_env: TaskEnvironment = self.env.get_task(task_switch[config.task])

    def close(self):
        self.env.shutdown()

    def _get_robot_base_pose(self) -> np.ndarray:
        raw = self.env._robot.arm.get_pose()

        pos = raw[:3]
        quat = raw[3:]
        logger.error("Dbg quat overwrite in _get_robot_base_pose")

        quat = quat_real_last_to_real_first(quat)

        return np.concatenate([pos, quat])

    def setup_camera_controls(self, config: RLBenchEnvironmentConfig):
        self.camera_pose = config.camera_pose

        camera_map = {
            "left_shoulder": self.env._scene._cam_over_shoulder_left,
            "right_shoulder": self.env._scene._cam_over_shoulder_right,
            "wrist": self.env._scene._cam_wrist,
            "overhead": self.env._scene._cam_overhead,
            "front": self.env._scene._cam_front,
        }

        self.camera_map = {k: v for k, v in camera_map.items() if k in self.cameras}

    def reset(self):
        super().reset()

        descriptions, obs = self.task_env.reset()

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.process_observation(obs)

        return obs

    def reset_to_demo(self, demo: Demo):
        super().reset()

        descriptions, obs = self.task_env.reset_to_demo(demo)

        if self.camera_pose:
            self.set_camera_pose(self.camera_pose)

        obs = self.process_observation(obs)

        return obs

    def _step(
        self,
        action: np.ndarray,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
    ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Catches invalid actions and executes a zero action instead.

        Parameters
        ----------
        action : np.ndarray
            The raw action predicted by a policy.
        postprocess : bool, optional
            Whether to postprocess the action at all, by default True
        delay_gripper : bool, optional
            Whether to delay the gripper action. Usually needed for ML
            policies, by default True
        scale_action : bool, optional
            Whether to scale the action. Usually needed for ML policies,
            by default True

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        RuntimeError
            If raised by the environment.
        """
        prediction_is_quat = action.shape[0] == 8

        if postprocess:
            action_delayed = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
            )
        else:
            action_delayed = action

        # NOTE: Quaternion in RLBench is real-last.
        gripper = 0.0 if np.isnan(action_delayed[-1]) else action_delayed[-1]
        zero_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, gripper]

        if np.isnan(action_delayed).any():
            logger.warning("NaN action, skipping")
            action_delayed = zero_action

        # action_delayed[:3] *= 0.05

        logger.info(f"Action {action_delayed}")

        try:
            with np.errstate(invalid="raise"):
                next_obs, reward, done = self.task_env.step(action_delayed)
        except (
            IKError,
            InvalidActionError,
            FloatingPointError,
            ConfigurationPathError,
        ):
            logger.info("Skipping invalid action {}.".format(action_delayed))

            try:
                with np.errstate(invalid="raise"):
                    next_obs, reward, done = self.task_env.step(zero_action)
            except (
                IKError,
                InvalidActionError,
                FloatingPointError,
                ConfigurationPathError,
            ):
                logger.info("Can't execute noop either. Just haning in there...")
                next_obs, reward, done = None, 0, True
        except RuntimeError as e:
            logger.error(f"Error in stepping action: {action_delayed}")
            logger.error(f"Raw action: {action}")
            raise e

        obs = None if next_obs is None else self.process_observation(next_obs)

        info = {}

        return obs, reward, done, info

    def get_camera_pose(self) -> dict[str, np.ndarray]:
        return {name: cam.get_pose() for name, cam in self.camera_map.items()}

    def set_camera_pose(self, pos_dict: dict[str, np.ndarray]) -> None:
        for camera_name, pos in pos_dict.items():
            if camera_name in self.camera_map:
                camera = self.camera_map[camera_name]
                camera.set_pose(pos)

    def _get_obj_poses(self) -> np.ndarray:
        """
        Low dim state of the task can contain more than just object poses, eg.
        force sensor readings, joint positions, etc., which makes it hard to parse.

        This is a fallback method to get the object poses from the task state.
        """

        state = []
        for obj, objtype in self.task_env._task._initial_objs_in_scene:
            if not obj.still_exists():
                # It has been deleted
                empty_len = 7
                # if objtype == ObjectType.JOINT:
                #     empty_len += 1
                # elif objtype == ObjectType.FORCE_SENSOR:
                #     empty_len += 6
                state.extend(np.zeros((empty_len,)).tolist())
            else:
                state.extend(np.array(obj.get_pose()))
                # if obj.get_type() == ObjectType.JOINT:
                #     state.extend([Joint(obj.get_handle()).get_joint_position()])
                # elif obj.get_type() == ObjectType.FORCE_SENSOR:
                #     forces, torques = ForceSensor(obj.get_handle()).read()
                #     state.extend(forces + torques)

        return np.array(state).flatten()

    def process_observation(self, obs: RLBenchObservation) -> SceneObservation:
        """
        Convert the observation from the environment to a SceneObservation.

        Parameters
        ----------
        obs : RLBenchObservation
            Observation as RLBench's Observation class.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        camera_obs = {}

        for cam in self.cameras:
            rgb = getattr(obs, cam + "_rgb").transpose((2, 0, 1)) / 255
            depth = getattr(obs, cam + "_depth")
            mask = getattr(obs, cam + "_mask").astype(int)
            extr = obs.misc[cam + "_camera_extrinsics"]
            intr = obs.misc[cam + "_camera_intrinsics"].astype(float)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(depth),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(extr),
                    "intr": torch.Tensor(intr),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.cameras)} | camera_obs
        )

        joint_pos = torch.Tensor(obs.joint_positions)
        joint_vel = torch.Tensor(obs.joint_velocities)

        ee_pose = torch.Tensor(
            np.concatenate(
                [
                    obs.gripper_pose[:3],
                    quat_real_last_to_real_first(obs.gripper_pose[3:]),
                ]
            )
        )
        logger.info(f"EE Pose {ee_pose}")
        gripper_open = torch.Tensor([obs.gripper_open])

        flat_object_poses = obs.task_low_dim_state

        n_objs = int(len(flat_object_poses) // 7)  # poses are 7 dim and stacked

        if len(flat_object_poses) % 7 != 0:
            logger.info("Can't parse low dim state, using fallback method.")
            flat_object_poses = self._get_obj_poses()
            n_objs = int(len(flat_object_poses) // 7)

        object_poses = tuple(
            np.concatenate((pose[:3], quat_real_last_to_real_first(pose[3:])))
            for pose in np.split(flat_object_poses, n_objs)
        )

        object_poses = dict_to_tensordict(
            {f"obj{i:03d}": torch.Tensor(pose) for i, pose in enumerate(object_poses)}
        )

        obs = SceneObservation(
            action=None,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            object_poses=object_poses,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=gripper_open,
            batch_size=empty_batchsize,
        )

        return obs

    @staticmethod
    def _get_action(
        current_obs: RLBenchObservation, next_obs: RLBenchObservation
    ) -> np.ndarray:
        gripper_action = np.array(
            [2 * next_obs.gripper_open - 1]  # map from [0, 1] to [-1, 1]
        )

        curr_b = current_obs.gripper_pose[:3]
        curr_q = quat_real_last_to_real_first(current_obs.gripper_pose[3:])
        curr_A = quaternion_to_matrix(curr_q)

        next_b = next_obs.gripper_pose[:3]
        next_q = quat_real_last_to_real_first(next_obs.gripper_pose[3:])
        next_A = quaternion_to_matrix(next_q)
        next_hom = homogenous_transform_from_rot_shift(next_A, next_b)

        # Transform from world into EE frame. In EE frame target pose and delta pose
        # are the same thing.
        world2ee = invert_homogenous_transform(
            homogenous_transform_from_rot_shift(curr_A, curr_b)
        )
        rot_delta = quaternion_to_axis_angle(quaternion_pose_diff(curr_q, next_q))

        pred_local = world2ee @ next_hom
        pos_delta = pred_local[:3, 3]

        return np.concatenate([pos_delta, rot_delta, gripper_action])

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        return quat_real_first_to_real_last(quaternion)

    def get_inverse_kinematics(
        self, target_pose: np.ndarray, reference_qpos: np.ndarray, max_configs: int = 20
    ) -> np.ndarray:
        arm = self.env._robot.arm  # .copy()
        arm.set_joint_positions(reference_qpos[:7], disable_dynamics=True)
        arm.set_joint_target_velocities([0] * len(arm.joints))

        return arm.solve_ik_via_sampling(
            position=target_pose[:3],
            quaternion=quat_real_first_to_real_last(target_pose[3:7]),
            relative_to=None,
            ignore_collisions=True,
            max_configs=max_configs,  # samples this many configs, then ranks them
        )[
            0
        ]  # return the closest one

        # return arm.solve_ik_via_jacobian(
        #     position=target_pose[:3],
        #     quaternion=quat_real_first_to_real_last(target_pose[3:7]),
        #     relative_to=None,
        # )

    def set_world_action_mode(self):
        self._set_action_mode(world_pos_action_mode)

    def get_forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        # RLBench needs the action mode to be set to world frame, otherwise the returned
        # gripper pose is in the end effector frame.
        # Thus, set it hear and put action mode context manager around the call.
        self.set_world_action_mode()

        self.env._robot.arm.set_joint_positions(qpos[:7], disable_dynamics=True)

        pose = self.env._robot.arm.get_tip().get_pose()

        return np.concatenate([pose[:3], quat_real_last_to_real_first(pose[3:])])

    def _rlbench_task_reset(self):
        self.task_env.reset()

    # def _get_variation_index(self) -> int:
    #     return self.task._variation_number

    # def _set_variation_index(self, index: int) -> None:
    #     self.task.set_variation(index)

    def _get_action_mode(self) -> ActionMode:
        return self.env._action_mode

    def _set_action_mode(self, action_mode: ActionMode) -> None:
        self.env._action_mode = action_mode

    def _get_state(self) -> tuple[bytes, int]:
        return self.task_env._task.get_state()

    def _set_state(self, state: tuple[bytes, int]):
        """
        The task state seems to only include scene objects, not the robot pose.

        Thus copied the scene reset from RLBench and replaced robot state restoration.
        https://github.com/stepjam/RLBench/blob/7c3f425f4a0b6b5ce001ba7246354eb3c70555be/rlbench/backend/scene.py#L150
        """
        self.env._robot.gripper.release()

        arm, gripper = self.env._scene._initial_robot_state
        self.env._scene.pyrep.set_configuration_tree(arm)
        self.env._scene.pyrep.set_configuration_tree(gripper)
        self.env._scene.robot.arm.set_joint_positions(
            self.env._scene._start_arm_joint_pos, disable_dynamics=True
        )
        self.env._scene.robot.arm.set_joint_target_velocities(
            [0] * len(self.env._scene.robot.arm.joints)
        )
        self.env._scene.robot.gripper.set_joint_positions(
            self.env._scene._starting_gripper_joint_pos, disable_dynamics=True
        )
        self.env._scene.robot.gripper.set_joint_target_velocities(
            [0] * len(self.env._scene.robot.gripper.joints)
        )

        if self.task_env is not None and self.env._scene._has_init_task:
            self.task_env._task.cleanup_()
            self.task_env._task.restore_state(state)  # Setting the desired state
        self.task_env._task.set_initial_objects_in_scene()
