import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import actionlib
import cv2
import nav_msgs.msg
import numpy as np
import roboticstoolbox as rtb
import rospy
import tf
import torch
from franka_gripper.msg import GraspAction, GraspGoal
from franka_msgs.msg import ErrorRecoveryActionGoal  # , FrankaState
from geometry_msgs.msg import PoseStamped, Vector3
from loguru import logger
from rl_franka import RLControllerManager
from rl_franka.panda import Panda
from rl_franka.panda_controller_manager import PandaControllerManager
from robot_io.cams.threaded_camera import ThreadedCamera
from sensor_msgs.msg import JointState

from tapas_gmm.env import Environment
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.utils.franka import clamp_translation
from tapas_gmm.utils.franka_joint_commander import ThreadedJointTrajectoryFollower
from tapas_gmm.utils.geometry_np import (
    homogenous_transform_from_rot_shift,
    matrix_to_quaternion,
    quat_real_first_to_real_last,
    quat_real_last_to_real_first,
    quaternion_is_unit,
    quaternion_to_matrix,
)
from tapas_gmm.utils.logging import indent_logs
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)
from tapas_gmm.utils.robot_trajectory import RobotTrajectory, TrajectoryPoint
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.viz.keypoint_selector import COLOR_RED, draw_reticle
from tapas_gmm.viz.operations import int_to_float_range, np_channel_back2front

# TODO: figure out how to define this in package-level init and import it here
PACKAGE_DIR = pathlib.Path(__file__).parent.parent.resolve()
PACKAGE_ASSET_DIR = PACKAGE_DIR / "assets"


class IK_Solvers(Enum):
    LM = "Levenberg-Marquardt"
    GN = "Gauss-Newton"
    NR = "Newton-Raphson"


@dataclass
class IKConfig:
    solver: IK_Solvers = IK_Solvers.GN
    max_iterations: int = 100
    max_searches: int = 10
    tolerance: float = 1e-6


@dataclass
class RealSenseConfig:
    _target_: str = "robot_io.cams.realsense.realsense.Realsense"
    _recursive_: bool = False
    name: str = "realsense"
    fps: int = 30
    img_type: str = "rgb_depth"
    resolution: tuple[int, int] = (640, 480)
    # crop_coords: tuple[int, int, int, int] = (40, 440, 120, 520)
    params: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 3.0,
            "contrast": 50.0,
            "exposure": 33.0,
            "gain": 16.0,
            "gamma": 300.0,
            "hue": 0.0,
            "saturation": 64.0,
            "sharpness": 50.0,
            "white_balance": 3400.0,
            "enable_auto_exposure": 1.0,
            "enable_auto_white_balance": 1.0,
        }
    )
    serial_number: str | None = None


@dataclass(kw_only=True)
class FrankaEnvironmentConfig(BaseEnvironmentConfig):
    teleop: bool
    eval: bool

    tele_grasp: bool = True

    image_crop: tuple[int, int, int, int] | None = None

    camera_stream_window_name = "Wrist RGB"

    joint_names: tuple[str, ...] = (
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    )
    neutral_joints: tuple[float, ...] = (
        -0.02233664,
        0.05485502,
        0.03904168,
        -1.6664815,
        -0.01360612,
        1.77192928,
        0.85765993,
    )

    position_limits: tuple = (  # tuple[tuple[float, float], ...]
        (-0.60, 0.75),  # x
        (-0.60, 0.60),  # y
        (-0.40, 0.90),
    )  # z

    gripper_threshold: float = 0.5  # 0.9

    cartesian_controller: str = "cartesian_impedance_controller"
    joint_controller: str = "joint_position_controller"
    joint_trajectory_controller: str = "position_joint_trajectory_controller"

    physical_cameras: dict[str, Any] = field(  # dict[str, tuple[str, str]]
        default_factory=lambda: {
            "wrist": ("/camera_wrist_depth_optical_frame", "128422270329"),
            "overhead": ("camera_tilt_depth_optical_frame", "135222066060"),
        }
    )
    realsense: RealSenseConfig = RealSenseConfig()

    ik: IKConfig = IKConfig()

    env_type: Environment = Environment.PANDA
    postprocess_actions: bool = True

    background: str | None = None
    model_ids: tuple[str, ...] | None = None
    camera_pose: Any = None  # not needed for franka, overwrite super()

    action_is_absolute: bool = False

    joint_action: bool = False

    dynamic_camera_visualization: bool = True


def euler_to_quaternion(euler_angle):
    # NOTE: quaternion is real-last! Only use the function in franka.py
    roll, pitch, yaw = euler_angle
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    quaternion = np.array([qx, qy, qz, qw])

    return quaternion


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    # NOTE: quaternion is real-last! Only use the function in franka.py
    qx, qy, qz, qw = quaternion
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(2 * (qw * qy - qz * qx))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 - qz**2))
    return np.array([roll, pitch, yaw])


class FrankaEnv(BaseEnvironment):
    def __init__(
        self,
        config: FrankaEnvironmentConfig,
        viz_encoder_callback: Callable | None = None,
    ):
        super().__init__(config)

        self.camera_names = config.cameras

        self.teleop = config.teleop
        self.tele_grasp = config.tele_grasp
        self.eval = config.eval

        self.neutral_joints = np.asarray(config.neutral_joints)
        self.cartesian_controller = config.cartesian_controller
        self.joint_controller = config.joint_controller
        self.joint_trajectory_controller = config.joint_trajectory_controller
        self.position_limits = config.position_limits

        logger.info("Initializing ROS...")

        rospy.init_node("franka_teleop_node_gpu")

        self.link_name = "panda_link0"

        self.robot_pose = PoseStamped()
        self.rot_euler = Vector3()

        self.robot = Panda()
        self.wait_for_initial_pose()

        # self.franka_state_sub = rospy.Subscriber(
        #     "/franka_state_controller/franka_states", FrankaState,
        #     self.update_robot_state)

        self.tf_publisher = tf.TransformBroadcaster()
        self.path_publisher = rospy.Publisher("/path", nav_msgs.msg.Path, queue_size=10)

        self.trajectory_follower = None

        self._set_up_robot_control(config)

        self.joint_state = JointState()
        self.joint_state.name = config.joint_names
        self.joint_state.position = config.neutral_joints

        self._set_up_cameras(config)

        self.trans = tf.TransformListener()

        self.win_rgb_name = config.camera_stream_window_name
        self.camera_rgb_window = cv2.namedWindow(self.win_rgb_name, cv2.WINDOW_AUTOSIZE)

        self._viz_encoder_callback = viz_encoder_callback

        self.rtb_model = rtb.models.URDF.Panda()

        self.reset()

        logger.info("Done!")

        self._currently_following_trajectory = False

        self._traj_follower_thread = None

    def _set_up_cameras(self, config: FrankaEnvironmentConfig) -> None:
        logger.info("Setting up cameras...")

        self.cameras = []
        self.camera_frames = []

        with indent_logs():
            realsense_conf = config.realsense
            for cam in self.camera_names:
                frame, sn = config.physical_cameras[cam]

                realsense_conf.serial_number = sn

                self.cameras.append(ThreadedCamera(realsense_conf))
                self.camera_frames.append(frame)

                logger.info("Found cam {} with serial number {}", cam, sn)

        assert len(self.cameras) == len(self.camera_names), "Some camera was not found."

        if not self.cameras:
            logger.info("Found no camera.")

        assert config.image_size == config.realsense.resolution[::-1]
        self.image_size = config.image_size
        self.image_crop = config.image_crop

        self.intrinsics = [torch.Tensor(c.get_camera_matrix()) for c in self.cameras]

        if (img_crop := config.image_crop) is not None:
            x_offset = img_crop[0]
            y_offset = img_crop[2]

            for i in range(len(self.intrinsics)):
                self.intrinsics[i][0][2] = self.intrinsics[i][0][2] - x_offset
                self.intrinsics[i][1][2] = self.intrinsics[i][1][2] - y_offset

    def _set_up_robot_control(self, config: FrankaEnvironmentConfig) -> None:
        if self.teleop:
            logger.info("Setting up teleop...")
            self.pose_publisher = rospy.Publisher(
                "/controllers/cartesian_impedance_controller/equilibrium_pose",
                PoseStamped,
                queue_size=10,
            )
            self.joint_publisher = rospy.Publisher(
                "/controllers/joint_position_controller/command",
                # "/franka_state_controller/joint_states_desired",
                JointState,
                queue_size=10,
            )

            rospy.Timer(rospy.Duration(0.005), self.publisher_callback)

            logger.info(" .. controllers set up.")

            # NOTE: if the code stops here after controller setup and before grasp setup,
            # then the ROS_IP is not set correctly. Check with `echo $ROS_IP` and
            # ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'

            self.grasp_client = actionlib.SimpleActionClient(
                "/franka_gripper/grasp", GraspAction
            )
            self.grasp_client.wait_for_server()
            self.grasp_state = None

            logger.info(" .. grasp client set up.")

            self.error_recovery_pub = rospy.Publisher(
                "/franka_control/error_recovery/goal",
                ErrorRecoveryActionGoal,
                queue_size=1,
            )

            self.recover_from_errors()

            logger.info(" .. error recovery set up.")

            self.panda_controller_manager = PandaControllerManager()
            # self.panda_controller_manager.set_joint_stiffness_high()
            self.panda_controller_manager.set_joint_stiffness_low()
            # self.panda_controller_manager.set_cartesian_stiffness_high()
            # self.panda_controller_manager.set_cartesian_stiffness(
            #     [100, 100, 100, 100, 100, 100]
            # )

            self.rl_controller_manager = RLControllerManager()
            self.rl_controller_manager.activate_controller(config.cartesian_controller)

            logger.info(" .. controller managers set up.")

            # self.trajectory_follower = JointTrajectoryFollower(gripper_cb=self.set_gripper_pose)

            # logger.info(" .. trajectory follower set up.")

        elif self.tele_grasp:
            self.grasp_client = actionlib.SimpleActionClient(
                "/franka_gripper/grasp", GraspAction
            )
            self.grasp_client.wait_for_server()
            self.grasp_state = None

    @property
    def _move_group(self) -> str:
        """
        For using the mplib Planner, eg for TOPP(RA) in gmm policy.
        """
        return "panda_hand_tcp"  # "panda_hand"  # "@panda_link8"

    @property
    def _urdf_path(self) -> str:
        """
        For using the mplib Planner, eg for TOPP(RA) in gmm policy.
        """
        return str(PACKAGE_ASSET_DIR / "mplib/panda/panda.urdf")

    @property
    def _srdf_path(self) -> str:
        return str(PACKAGE_ASSET_DIR / "mplib/panda/panda.srdf")

    def close(self) -> None:
        rospy.signal_shutdown("Keyboard interrupt.")

    def wait_for_initial_pose(self) -> None:
        while self.robot.state.O_T_EE is None:
            pass

        O_T_EE = self.robot.state.O_T_EE
        initial_quaternion = tf.transformations.quaternion_from_matrix(O_T_EE)
        assert quaternion_is_unit(initial_quaternion)
        # initial_quaternion /= np.linalg.norm(initial_quaternion)

        (
            self.robot_pose.pose.orientation.x,
            self.robot_pose.pose.orientation.y,
            self.robot_pose.pose.orientation.z,
            self.robot_pose.pose.orientation.w,
        ) = initial_quaternion
        (
            self.robot_pose.pose.position.x,
            self.robot_pose.pose.position.y,
            self.robot_pose.pose.position.z,
        ) = O_T_EE[:3, 3]

        self.rot_euler.x, self.rot_euler.y, self.rot_euler.z = quaternion_to_euler(
            initial_quaternion
        )

        # self.initial_pos = O_T_EE[:3, 3]
        # self.initial_quaternion = initial_quaternion  # NOTE: real-last quaternion
        # self.initial_q = self.robot.state.q

    @logger.contextualize(filter=False)
    def return_to_neutral_pose(self) -> None:
        self.recover_from_errors()

        logger.info("Returning to neutral pose ...")

        self.robot.authorize_reset()
        self.robot.move_joint_position(self.neutral_joints, 0.15, 0.02)
        self.wait_for_initial_pose()

        with indent_logs():
            logger.info("Done.")

        self.robot.cm.activate_controller(self.cartesian_controller)

    def recover_from_errors(self) -> None:
        """
        Error recovery for real robot.
        """
        self.robot.cm.recover_error_state()

    def reset(self) -> SceneObservation:
        super().reset()

        if self.trajectory_follower is not None:
            self.trajectory_follower.cancel_all_goals()

        if self.teleop:
            self.set_gripper_pose(1)
            self.return_to_neutral_pose()
            self.set_gripper_pose(1)
            self.return_to_neutral_pose()
            # self.set_gripper_pose(1)
        elif self.tele_grasp:
            self.set_gripper_pose(1)

        obs = self.get_obs(update_visualization=True)

        return obs

    def publisher_callback(self, msg) -> None:
        self.robot_pose.header.frame_id = self.link_name
        self.robot_pose.header.stamp = rospy.Time(0)

        self.pose_publisher.publish(self.robot_pose)

    # def update_robot_state(self, state):
    #     self.robot_state = state

    def set_gripper_pose(self, action) -> None:
        width_max = 0.2
        width_min = 0.0
        force = 5  # max: 70N
        speed = 0.1
        epsilon_inner = 0.6
        epsilon_outer = 0.6

        open_grip = action > self.config.gripper_threshold

        if self.grasp_state is None or self.grasp_state != action:
            self.grasp_state = action

            grasp_action = GraspGoal()
            grasp_action.speed = speed
            grasp_action.force = force
            grasp_action.epsilon.inner = epsilon_inner
            grasp_action.epsilon.outer = epsilon_outer

            grasp_action.width = width_max if open_grip else width_min

            self.grasp_client.send_goal(grasp_action)

    def _step(
        self,
        action: np.ndarray | RobotTrajectory,
        postprocess: bool = True,
        delay_gripper: bool = True,
        scale_action: bool = True,
        invert_action: tuple[bool, bool, bool] = (True, False, True),
    ) -> tuple[SceneObservation, float, bool, dict]:
        """
        Postprocess the action and execute it in the environment.
        Clamps translations to the workspace limits.

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
        invert_action: tuple[bool], optional
            Whether to invert the translation in the x, y, z direction.

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.
        """
        reward, done = 1, False

        info = {}

        if (
            action is None
        ):  # Should only pass None if the robot is currently following a trajectory
            assert self._currently_following_trajectory
            done = not self.trajectory_follower.is_running
        elif type(action) is RobotTrajectory and action.has_qposes:
            self._follow_trajectory(action)
        elif type(action) is RobotTrajectory and action.has_ee_poses:
            for a in action:
                self._step_single_action(
                    np.concatenate((a.ee, a.gripper)),
                    postprocess,
                    delay_gripper,
                    scale_action,
                    invert_action,
                )

        else:
            self._step_single_action(
                action, postprocess, delay_gripper, scale_action, invert_action
            )

        obs = self.get_obs(
            update_visualization=self.config.dynamic_camera_visualization
        )

        return obs, reward, done, info

    def _step_single_action(
        self, action, postprocess, delay_gripper, scale_action, invert_action
    ):
        ee_action, gripper_action = action[:-1], action[-1]

        if self.config.joint_action:
            self._step_joint_action(ee_action)

        elif self.config.action_is_absolute:
            self._step_absolute_ee_action(ee_action)

        else:
            # does postprocessing and might delay the gripper action
            gripper_action = self._step_relative_ee_action(
                action, postprocess, delay_gripper, scale_action, invert_action
            )

        if self.teleop or self.tele_grasp:
            self.set_gripper_pose(gripper_action)

    def _step_relative_ee_action(
        self,
        action: np.ndarray,
        postprocess: bool,
        delay_gripper: bool,
        scale_action: bool,
        invert_action: tuple[bool, bool, bool],
    ) -> np.ndarray:
        prediction_is_quat = action.shape[0] == 8

        if postprocess:
            action = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
                prediction_is_euler=not prediction_is_quat,
            )
        else:
            action = action

        delta_position, delta_rot_euler, gripper = np.split(action, [3, 6])

        for i in range(3):
            if invert_action[i]:
                delta_position[i] = -delta_position[i]

        self.robot_pose.pose.position.x = clamp_translation(
            self.robot_pose.pose.position.x,
            delta_position[0],
            self.position_limits[0],
        )

        self.robot_pose.pose.position.y = clamp_translation(
            self.robot_pose.pose.position.y,
            delta_position[1],
            self.position_limits[1],
        )

        self.robot_pose.pose.position.z = clamp_translation(
            self.robot_pose.pose.position.z,
            delta_position[2],
            self.position_limits[2],
        )

        # delta_angle_euler = quaternion_to_euler(delta_angle_quat)
        self.rot_euler.x += delta_rot_euler[0]
        self.rot_euler.y += delta_rot_euler[1]
        self.rot_euler.z += delta_rot_euler[2]

        if self.teleop:
            self._set_rotation(
                euler_to_quaternion(
                    [self.rot_euler.x, self.rot_euler.y, self.rot_euler.z]
                )
            )

        return gripper

    def _step_absolute_ee_action(self, ee_action: np.ndarray) -> None:
        if ee_action.shape == (7,):
            goal_pos_world, goal_quat_world = np.split(ee_action, [3])
            control_rotation = True
        elif ee_action.shape == (3,):  # no rotation
            goal_pos_world = ee_action
            control_rotation = False
            goal_quat_world = [0, 0, 0, 1]  # dummy value
        else:
            raise ValueError(f"Unexpected action shape {ee_action.shape}.")

        panda_base_pose = self.get_panda_base_pose()

        goal_pose_matrix = tf.transformations.quaternion_matrix(goal_quat_world)
        goal_pose_matrix[:3, 3] = goal_pos_world

        # Transform the goal pose from world frame to panda base frame
        goal_pose_matrix = np.linalg.inv(panda_base_pose) @ goal_pose_matrix

        goal_pos = goal_pose_matrix[:3, 3]
        goal_quat = tf.transformations.quaternion_from_matrix(goal_pose_matrix)

        self.robot_pose.pose.position.x = clamp_translation(
            goal_pos[0], 0, self.position_limits[0]
        )
        self.robot_pose.pose.position.y = clamp_translation(
            goal_pos[1], 0, self.position_limits[1]
        )
        self.robot_pose.pose.position.z = clamp_translation(
            goal_pos[2], 0, self.position_limits[2]
        )

        if self.teleop and control_rotation:
            self._set_rotation(quat_real_first_to_real_last(goal_quat))

    def _step_joint_action(
        self, qposes: np.ndarray, vel_scale=0.15, tolerance=1e-2
    ) -> None:
        assert qposes.shape == (7,), f"Unexpected action shape {qposes.shape}."

        self.robot.move_joint_position(qposes, vel_scale=vel_scale, tolerance=tolerance)

    def _follow_trajectory(self, trajectory: RobotTrajectory) -> None:
        self._currently_following_trajectory = True

        self.rl_controller_manager.refresh_controller_state()
        self.rl_controller_manager.activate_controller(self.joint_trajectory_controller)
        self.trajectory_follower = ThreadedJointTrajectoryFollower(
            gripper_cb=self.set_gripper_pose
        )

        self.trajectory_follower.follow_time_based(trajectory)

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        """
        Postprocess the quaternion action.
        NOTE: quaternion is real first! Real last is only used internally in the
        franka environment. All interfaces with the rest of the codebase should
        use real-first quaternions.
        """
        return quaternion_to_euler(quat_real_first_to_real_last(quaternion))

    def _set_position(self, position) -> None:
        (
            self.robot_pose.pose.position.x,
            self.robot_pose.pose.position.y,
            self.robot_pose.pose.position.z,
        ) = position

    def _set_rotation(self, quaternion) -> None:
        (
            self.robot_pose.pose.orientation.x,
            self.robot_pose.pose.orientation.y,
            self.robot_pose.pose.orientation.z,
            self.robot_pose.pose.orientation.w,
        ) = quaternion

    def get_obs(self, update_visualization: bool = False) -> SceneObservation:
        frames = [cam.get_image() for cam in self.cameras]
        img_frames = [f[0] for f in frames]
        depth_frames = [f[1] for f in frames]

        # NOTE: franka state's O_T_EE seems to be relative to the arm's base, not
        # the FMM base link, which I use as the world frame.
        # gripper and wrist pose are coordinates + quaternions in world frame
        # wrist_position = np.array(self.robot.state.O_T_EE[:3, 3])
        # wrist_quaternion = quat_real_last_to_real_first(
        #     tf.transformations.quaternion_from_matrix(self.robot.state.O_T_EE)
        # )
        # wrist_pose = torch.Tensor(np.concatenate((wrist_position, wrist_quaternion)))

        tcp_pose = torch.Tensor(self.get_tcp_pose())

        # ee_pose = torch.Tensor(self.get_franka_ee_pose())
        # logger.info("ee_pose: {}", ee_pose)
        # logger.info("tcp_pose: {}", tcp_pose)

        joint_pos = torch.Tensor(self.robot.state.q)
        joint_vel = torch.Tensor(self.robot.state.d_q)

        gripper_width = torch.Tensor([self.robot.gripper_pos])

        # extrinsics are in homegenous matrix format
        extrinsics = [self.get_camera_pose(f) for f in self.camera_frames]

        cam_img = img_frames[0][:, :, ::-1].copy()  # make contiguous

        if self.image_crop is not None:
            logger.warning("ImageCrop in franka env is untested.")
            image_h, image_w = cam_img.shape[:2]
            crop_l, crop_r, crop_t, crop_b = self.image_crop
            display_image = cv2.line(
                cam_img, (crop_l, 0), (crop_l, image_h), (0, 0, 255), 2
            )
            display_image = cv2.line(
                display_image,
                (image_w - crop_r, 0),
                (image_w - crop_r, image_h),
                (0, 0, 255),
                2,
            )
            display_image = cv2.line(
                display_image, (0, crop_t), (image_w, crop_t), (0, 0, 255), 2
            )
            display_image = cv2.line(
                display_image,
                (0, image_h - crop_b),
                (image_w, image_h - crop_b),
                (0, 0, 255),
                2,
            )

        else:
            display_image = cam_img

        img_frames = [int_to_float_range(np_channel_back2front(f)) for f in img_frames]

        if self.image_crop is not None:
            logger.warning("ImageCrop in franka env is untested.")
            l, r, t, b = self.image_crop
            img_frames = [
                i[:, t : i.shape[-2] - b, l : i.shape[-1] - r] for i in img_frames
            ]
            depth_frames = [
                i[t : i.shape[-2] - b, l : i.shape[-1] - r] for i in depth_frames
            ]

        camera_obs = {}
        for i, cam in enumerate(self.camera_names):
            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(img_frames[i]),
                    "depth": torch.Tensor(depth_frames[i]),
                    "extr": torch.Tensor(extrinsics[i]),
                    "intr": torch.Tensor(self.intrinsics[i]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.camera_names)} | camera_obs
        )

        obs = SceneObservation(
            cameras=multicam_obs,
            ee_pose=tcp_pose,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=gripper_width,
            batch_size=empty_batchsize,
        )

        self._current_cam_img = cam_img
        # self.rgb_w_kp = self._current_cam_img.copy()

        if update_visualization:
            if self._viz_encoder_callback is None:
                viz_info = {}
            else:
                _, enc_info = self._viz_encoder_callback(obs.to(device).unsqueeze(0))
                viz_info = {
                    "vis_encoding": [enc_info["kp_raw_2d"][0].squeeze(0).cpu()],
                    "heatmap": [enc_info["post"][0].squeeze(0).cpu()],
                }
            self.update_visualization(viz_info)

        return obs

    def propose_update_visualization(self, info: dict) -> None:
        if self.config.dynamic_camera_visualization:
            self.update_visualization(info)

    def update_visualization(self, info: dict) -> None:
        self.rgb_w_kp = self._current_cam_img.copy()

        if "vis_encoding" in info and info["vis_encoding"] is not None:
            # Update the visualization of the keypoints
            # Keeps the last image if no info passed
            kp_tens = info["vis_encoding"][0]  # First one should be wrist camera

            u, v = kp_tens.chunk(2)
            u = (u / 2 + 0.5) * self.image_size[1]
            v = (v / 2 + 0.5) * self.image_size[0]

            if self.image_crop is not None:
                u += self.image_crop[0]
                v += self.image_crop[2]

        if "heatmap" in info and info["heatmap"] is not None:
            heatmap = info["heatmap"][0]
            heatmap = (
                torch.nn.functional.interpolate(
                    heatmap.unsqueeze(0),
                    size=self.rgb_w_kp.shape[:2],
                    mode="bilinear",
                    align_corners=True,
                )
                .squeeze(0)
                .sum(dim=0)
                .cpu()
                .numpy()
            )
            heatmap *= 255
            heatmap = heatmap.astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            self.rgb_w_kp = cv2.addWeighted(self.rgb_w_kp, 0.3, heatmap_color, 0.7, 0)

            for i, j in zip(u, v):
                draw_reticle(self.rgb_w_kp, int(i), int(j), COLOR_RED)

        cv2.imshow(self.win_rgb_name, self.rgb_w_kp)

        cv2.waitKey(1)

    def get_camera_pose(self, topic_name: str, frame: str = "/base_link") -> np.ndarray:
        cam_position, cam_quaternion = self.trans.lookupTransform(
            frame, topic_name, rospy.Time(0)
        )
        cam_rot_matrix = tf.transformations.quaternion_matrix(cam_quaternion)[:3, :3]
        return homogenous_transform_from_rot_shift(cam_rot_matrix, cam_position)

    def get_tcp_pose(
        self, frame: str = "/base_link", prefer_pos_real_quat: bool = False
    ) -> np.ndarray:
        ee_position, ee_quaternion = self.trans.lookupTransform(
            frame, "/panda_hand_tcp", rospy.Time(0)
        )
        ee_position = np.array(ee_position)
        ee_quaternion = quat_real_last_to_real_first(np.array(ee_quaternion))

        if prefer_pos_real_quat:
            if ee_quaternion[0] < 0:
                ee_quaternion = -ee_quaternion

        return np.concatenate((ee_position, ee_quaternion))

    def get_franka_ee_pose(self, frame: str = "/base_link") -> np.ndarray:
        ee_position, ee_quaternion = self.trans.lookupTransform(
            frame, "/panda_EE", rospy.Time(0)
        )
        ee_position = np.array(ee_position)
        ee_quaternion = quat_real_last_to_real_first(np.array(ee_quaternion))

        return np.concatenate((ee_position, ee_quaternion))

    def get_panda_base_pose(self, frame: str = "/base_link") -> np.ndarray:
        base_position, base_quaternion = self.trans.lookupTransform(
            frame, "/panda_link0", rospy.Time(0)
        )
        base_matrix = tf.transformations.quaternion_matrix(base_quaternion)

        base_matrix[:3, 3] = base_position

        return np.array(base_matrix)

    def get_inverse_kinematics(
        self,
        target_pose: np.ndarray,
        reference_qpos: np.ndarray,
    ) -> np.ndarray:
        if target_pose.shape == (7,):
            target_hom = homogenous_transform_from_rot_shift(
                quaternion_to_matrix(target_pose[3:]), target_pose[:3]
            )
            mask = np.ones(6)
        elif target_pose.shape == (3,):  # no rotation specified
            gripper_rot = np.diag([1, -1, -1])
            target_hom = homogenous_transform_from_rot_shift(gripper_rot, target_pose)
            mask = np.ones(6)
        else:
            raise ValueError(f"Unexpected target pose shape {target_pose.shape}.")

        robot_base_pose = self.get_panda_base_pose()

        target_robo_frame = np.linalg.inv(robot_base_pose) @ target_hom

        ik_funcs = {
            IK_Solvers.LM: self.rtb_model.ik_LM,
            IK_Solvers.GN: self.rtb_model.ik_GN,
            IK_Solvers.NR: self.rtb_model.ik_NR,
        }

        qpos, success, iters, searches, residual = ik_funcs[self.config.ik.solver](
            target_robo_frame,
            q0=reference_qpos,
            ilimit=self.config.ik.max_iterations,
            slimit=self.config.ik.max_searches,
            tol=self.config.ik.tolerance,
            mask=mask,
        )

        if not success:
            raise ValueError(f"Failed to find IK solution, residual: {residual}")

        return qpos

    def get_forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        pose_local = self.rtb_model.fkine(qpos).data[0]

        robot_base_pose = self.get_panda_base_pose()

        pose_world = robot_base_pose @ pose_local

        quat_world = matrix_to_quaternion(pose_world[:3, :3])

        return np.concatenate((pose_world[:3, 3], quat_world))

    def publish_frames(self, frame_trans: np.ndarray, frame_quats: np.ndarray) -> None:
        """
        Publishes the given frames to the tf tree for visualization in rviz.
        Used for visualizing keypoint frames.
        """
        for i, (trans, quat) in enumerate(zip(frame_trans, frame_quats)):
            self.tf_publisher.sendTransform(
                trans[:3, 3],
                quat_real_first_to_real_last(quat),
                rospy.Time.now(),
                f"kp-{i}",
                "/base_link",
            )

    def publish_path(self, trajectory: RobotTrajectory) -> None:
        """
        Publishes the given trajectory to the path topic for visualization in rviz.
        """
        path_msg = nav_msgs.msg.Path()
        path_msg.header.frame_id = "/base_link"

        path = trajectory.ee

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = point[:3]
            if point.shape[0] == 8:
                (
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ) = quat_real_first_to_real_last(point[3:7])
            else:
                (
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ) = (0, 0, 0, 1)

            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
