import time

import numpy as np
import rlbench
import sapien.core as sapien
from loguru import logger
from sapien.utils.viewer import Viewer


class SapienScene:
    def __init__(self, robot_pose: np.ndarray):
        self._setup_scene()
        self._load_robot(robot_pose)
        self._pin_model = self.robot.create_pinocchio_model()

    @property
    def _urdf_path(self) -> str:
        import rlbench

        logger.info(rlbench.__file__)
        # return f"{mani_skill2.PACKAGE_ASSET_DIR}/descriptions/panda_v2.urdf"
        return "/home/hartzj/CodeRepos/RLBench/urdfs/panda/panda.urdf"

    @property
    def _ee_link_idx(self) -> int:
        return 10

    @property
    def _q_mask(self) -> np.ndarray:
        return np.array([True] * 7 + [False] * 2)

    @property
    def _move_group(self) -> str:
        return self.robot.get_links()[self._ee_link_idx].get_name()

    def _setup_scene(self, **kwargs):
        # declare sapien sim
        self.engine = sapien.Engine()
        # declare sapien renderer
        self.renderer = sapien.SapienRenderer(offscreen_only=False)
        # give renderer to sapien sim
        self.engine.set_renderer(self.renderer)

        # declare sapien scene
        scene_config = sapien.SceneConfig()
        self.scene = self.engine.create_scene(scene_config)
        # set simulation timestep
        self.scene.set_timestep(kwargs.get("timestep", 1 / 240))
        # add ground to scene
        self.scene.add_ground(kwargs.get("ground_height", 0))
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            kwargs.get("static_friction", 1),
            kwargs.get("dynamic_friction", 1),
            kwargs.get("restitution", 0),
        )
        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(kwargs.get("ambient_light", [0.5, 0.5, 0.5]))
        # default enable shadow unless specified otherwise
        shadow = kwargs.get("shadow", True)
        # default spotlight angle and intensity
        direction_lights = kwargs.get(
            "direction_lights", [[[0, 1, -1], [0.5, 0.5, 0.5]]]
        )
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        point_lights = kwargs.get(
            "point_lights",
            [[[1, 2, 2], [1, 1, 1]], [[1, -2, 2], [1, 1, 1]], [[-1, 0, 1], [1, 1, 1]]],
        )
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

        # initialize viewer with camera position and orientation
        self.viewer = Viewer(self.renderer)
        self.viewer.set_scene(self.scene)
        self.viewer.set_camera_xyz(
            x=kwargs.get("camera_xyz_x", 1.2),
            y=kwargs.get("camera_xyz_y", 0.25),
            z=kwargs.get("camera_xyz_z", 0.4),
        )
        self.viewer.set_camera_rpy(
            r=kwargs.get("camera_rpy_r", 0),
            p=kwargs.get("camera_rpy_p", -0.4),
            y=kwargs.get("camera_rpy_y", 2.7),
        )
        logger.info("Created scene.")

    def _load_robot(self, robot_pose: np.ndarray, **kwargs):
        """
        Adapted from https://motion-planning-lib.readthedocs.io/latest/tutorials/plan_a_path.html
        Creates a robot model for kinematic computations, when using RLBench.
        """
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = False
        self.robot: sapien.Articulation = loader.load(
            kwargs.get("urdf_path", self._urdf_path)
        )
        self.robot.set_root_pose(
            sapien.Pose(
                kwargs.get("robot_origin_xyz", [0, 0, 0]),
                kwargs.get("robot_origin_quat", [1, 0, 0, 0]),
            )
        )

        self.robot_pose = robot_pose

        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness=kwargs.get("joint_stiffness", 1000),
                damping=kwargs.get("joint_damping", 200),
            )

    def get_inverse_kinematics(
        self,
        target_pose: np.ndarray,
        reference_qpos: np.ndarray,
        max_iterations: int = 100,
    ) -> np.ndarray:
        self.viewer.render()

        import time

        # time.sleep(5)
        self.robot.set_qpos(reference_qpos)
        logger.info(
            f"Set initial qpos. EE at {self.get_forward_kinematics(reference_qpos)}."
        )
        self.viewer.render()
        time.sleep(10)

        # use self.robot_pose to transform target_pose to local frame for pin model
        # robot_pos, robot_quat = self.robot_pose[:3], self.robot_pose[3:]
        # target_pos, target_quat = target_pose[:3], target_pose[3:]
        # robot_matrix = homogenous_transform_from_rot_shift(
        #     quaternion_to_matrix(robot_quat), robot_pos
        # )
        # target_matrix = homogenous_transform_from_rot_shift(
        #     quaternion_to_matrix(target_quat), target_pos
        # )
        # target_pos_local = (invert_homogenous_transform(robot_matrix) @ target_matrix)[
        #     :3, 3
        # ]
        # target_quat_local = quaternion_pose_diff(robot_quat, target_quat)
        # target_pose = np.concatenate([target_pos_local, target_quat_local])

        logger.info(f"Target pose: {target_pose}.")

        qpos, success, error = self._pin_model.compute_inverse_kinematics(
            self._ee_link_idx,
            sapien.Pose(target_pose[:3], target_pose[3:7]),
            initial_qpos=reference_qpos,
            active_qmask=self._q_mask,
            max_iterations=max_iterations,
        )

        print(target_pose, reference_qpos)
        print(qpos, success, error)

        if not success:
            raise ValueError(f"Failed to find IK solution: {error}")

        return qpos

    def get_forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        self._pin_model.compute_forward_kinematics(qpos)

        pose = self._pin_model.get_link_pose(self._ee_link_idx)

        return np.concatenate([pose.p, pose.q])
