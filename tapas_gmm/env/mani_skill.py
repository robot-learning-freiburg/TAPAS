from dataclasses import dataclass
from types import MappingProxyType

import cv2
import gymnasium as gym
import mani_skill2.agents.controllers
import mani_skill2.envs  # noqa: F401
import numpy as np
import sapien.core
import torch
from loguru import logger

from tapas_gmm.env import Environment
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.utils.geometry_np import (
    invert_homogenous_transform,
    quaternion_to_axis_angle,
)
from tapas_gmm.utils.misc import invert_dict
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)

ACTION_MODE = "pd_ee_delta_pose"
OBS_MODE = "state_dict+image"


default_cameras = ("hand_camera", "base_camera")

cam_name_tranlation = MappingProxyType(
    {
        "hand_camera": "wrist",
        "base_camera": "base",
        "overhead": "overhead",
        "overhead_camera_0": "overhead_0",
        "overhead_camera_1": "overhead_1",
        "overhead_camera_2": "overhead_2",
    }
)

inv_cam_name_tranlation = invert_dict(cam_name_tranlation)


@dataclass(kw_only=True)
class ManiSkillEnvironmentConfig(BaseEnvironmentConfig):
    render_sapien: bool
    background: str | None
    model_ids: tuple[str, ...] | None
    real_depth: bool
    seed: int | None

    fixed_target_link_idx: str | None = None

    env_type: Environment = Environment.MANISKILL
    postprocess_actions: bool = True
    invert_xy: bool = True

    obs_mode: str = OBS_MODE
    action_mode: str = ACTION_MODE


class ManiSkillEnv(BaseEnvironment):
    def __init__(self, config: ManiSkillEnvironmentConfig, **kwargs) -> None:
        super().__init__(config)

        # NOTE: removed ms_config dict. Just put additional kwargs into the
        # config dict and treat them here and in launch_simulation_env.

        # ManiSkill controllers have action space normalized to [-1,1].
        # Max speed is a bit fast for teleop, so scale down.
        self._delta_pos_scale = 0.25
        self._delta_angle_scale = 0.5

        self.cameras = config.cameras
        self.cameras_ms = [inv_cam_name_tranlation[c] for c in self.cameras]

        image_size = (self.image_height, self.image_width)

        self.camera_cfgs = {
            "width": image_size[1],
            "height": image_size[0],
            "use_stereo_depth": config.real_depth,
            "add_segmentation": True,
            # NOTE: these are examples of how to pass camera params.
            # Should specify these in the config file.
            # "overhead": {  # can pass specific params per cam as well
            #     'p': [0.2, 0, 0.2],
            #     # Quaternions are [w, x, y, z]
            #     'q': [7.7486e-07, -0.194001, 7.7486e-07, 0.981001]
            # },
            # "base_camera": {
            #     'p': [0.2, 0, 0.2],
            #     'q': [0, 0.194, 0, -0.981]  # Quaternions are [w, x, y, z]
            # }
        }

        self.extra_cams = []

        for c, pq in config.camera_pose.items():
            ms_name = inv_cam_name_tranlation[c]
            if self.camera_cfgs.get(ms_name) is None:
                self.camera_cfgs[ms_name] = {}
            if ms_name not in default_cameras:
                self.extra_cams.append(ms_name)
            self.camera_cfgs[ms_name]["p"] = pq[:3]
            self.camera_cfgs[ms_name]["q"] = pq[3:]

        self.task_name = config.task
        self.headless = config.headless

        self.gym_env = None

        self.render_sapien = config.render_sapien
        self.bg_name = config.background
        self.model_ids = config.model_ids

        self.obs_mode = config.obs_mode
        self.action_mode = config.action_mode

        self.invert_xy = config.invert_xy

        self.seed = config.seed

        if config.static:
            raise NotImplementedError

        # NOTE: would like to make the horizon configurable, but didn't figure
        # it out how to make this work with the Maniskill env registry. TODO
        # self.horizon = -1

        # if self.model_ids is None:
        #     self.model_ids = []

        if not self.render_sapien and not self.headless:
            self.cam_win_title = "Observation"
            self.camera_rgb_window = cv2.namedWindow(
                self.cam_win_title, cv2.WINDOW_AUTOSIZE
            )

        self._patch_register_cameras()
        self.launch_simulation_env(config)

        self._pin_model = self._create_pin_model()

    def _create_pin_model(self) -> sapien.core.PinocchioModel:
        return self.agent.controller.articulation.create_pinocchio_model()

    @property
    def _arm_controller(self) -> mani_skill2.agents.controllers.PDJointPosController:
        return self.agent.controller.controllers["arm"]

    @property
    def _ee_link_idx(self) -> int:
        return self._arm_controller.ee_link_idx  # type: ignore

    @property
    def _q_mask(self) -> np.ndarray:
        return self._arm_controller.qmask  # type: ignore

    @property
    def _move_group(self) -> str:
        return self.robot.get_links()[self._ee_link_idx].get_name()

    @property
    def camera_names(self):
        return tuple(
            cam_name_tranlation[c] for c in self.gym_env.env._camera_cfgs.keys()
        )

    @property
    def _urdf_path(self) -> str:
        return self.agent._get_urdf_path()

    @property
    def _srdf_path(self) -> str:
        return self.agent._get_srdf_path()

    @property
    def agent(self):
        return self.gym_env.agent

    @property
    def robot(self):
        return self.agent.robot

    def get_solution_sequence(self):
        return self.gym_env.env.get_solution_sequence()

    def _patch_register_cameras(self):
        from mani_skill2.sensors.camera import CameraConfig
        from mani_skill2.utils.sapien_utils import look_at

        # from sapien.core import Pose as SapienPose

        envs = [
            mani_skill2.envs.pick_and_place.pick_clutter.PickClutterEnv,
            mani_skill2.envs.pick_and_place.pick_cube.PickCubeEnv,
            mani_skill2.envs.pick_and_place.pick_cube.LiftCubeEnv,
            mani_skill2.envs.pick_and_place.pick_clutter.PickClutterYCBEnv,
            mani_skill2.envs.pick_and_place.stack_cube.StackCubeEnv,
            mani_skill2.envs.pick_and_place.pick_single.PickSingleEGADEnv,
            mani_skill2.envs.pick_and_place.pick_single.PickSingleYCBEnv,
            # mani_skill2.envs.assembly.assembling_kits.AssemblingKitsEnv,
            # TODO: for some reason, these two break upon patching
            # mani_skill2.envs.assembly.peg_insertion_side.PegInsertionSideEnv,
            # mani_skill2.envs.assembly.plug_charger.PlugChargerEnv
        ]

        if self.task_name in ["PegInsertionSide-v0", "PlugCharger-v0"]:
            logger.opt(ansi=True).warning(
                f"Skipping camera patching for {self.task_name}. "
                "<red>This disables camera customization, including the "
                "overhead camera.</red> See code for details."
            )
            if "overhead" in self.camera_cfgs:
                self.camera_cfgs.pop("overhead")

        def _register_cameras(self):
            cfgs = _orig_register_cameras(self)
            if type(cfgs) is CameraConfig:
                cfgs = [cfgs]
            pose = look_at([0, 0, 0], [0, 0, 0])
            for c in self._extra_camera_names:
                if c == "base_camera":
                    continue
                else:
                    logger.info(f"Registering camera {c}")
                    cfgs.append(
                        CameraConfig(c, pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
                    )
            return cfgs

        for env in envs:
            _orig_register_cameras = env._register_cameras

            env._extra_camera_names = self.extra_cams
            env._register_cameras = _register_cameras

    def launch_simulation_env(self, config):
        env_name = self.task_name

        kwargs = {
            "obs_mode": self.obs_mode,
            "control_mode": self.action_mode,
            "camera_cfgs": self.camera_cfgs,
            "shader_dir": "rt" if config.real_depth else "ibl",
            # "render_camera_cfgs": dict(width=640, height=480)
            "bg_name": self.bg_name,
            "model_ids": self.model_ids,
            # "max_episode_steps": self.horizon,
            "fixed_target_link_idx": self.config.fixed_target_link_idx,
            "reward_mode": "sparse",
        }

        if kwargs["model_ids"] is None:
            kwargs.pop("model_ids")  # model_ids only needed for some tasks
        if kwargs["fixed_target_link_idx"] is None:
            kwargs.pop("fixed_target_link_idx")  # only needed for some tasks

        # NOTE: full list of arguments
        # obs_mode = None,
        # control_mode = None,
        # sim_freq: int = 500,
        # control_freq: int = 20, That's what I use already.
        # renderer: str = "sapien",
        # renderer_kwargs: dict = None,
        # shader_dir: str = "ibl",
        # render_config: dict = None,
        # enable_shadow: bool = False,
        # camera_cfgs: dict = None,
        # render_camera_cfgs: dict = None,
        # bg_name: str = None,

        self.gym_kwargs = kwargs

        self.gym_env = gym.make(env_name, **kwargs)

        if self.seed is not None:
            self.gym_env.seed(self.seed)

    def make_twin(self, control_mode: str | None = None, obs_mode: str | None = None):
        kwargs = self.gym_kwargs.copy()
        if control_mode is not None:
            kwargs["control_mode"] = control_mode
        if obs_mode is not None:
            kwargs["obs_mode"] = obs_mode

        return gym.make(self.task_name, **kwargs)

    def render(self):
        if not self.headless:
            if self.render_sapien:
                self.gym_env.render_human()
            else:
                obs = self.gym_env.render_cameras()
                cv2.imshow(self.cam_win_title, obs)
                cv2.waitKey(1)

    def reset(self, **kwargs):
        super().reset()

        obs, _ = self.gym_env.reset(**kwargs)

        obs = self.process_observation(obs)

        self._pin_model = self._create_pin_model()

        return obs

    def reset_to_demo(self, demo):
        reset_kwargs = demo["reset_kwargs"]
        seed = reset_kwargs.pop("seed")
        return self.reset(seed=seed, options=reset_kwargs)

    def get_seed(self):
        return self.gym_env.get_episode_seed()

    def get_state(self):
        return self.gym_env.get_state()

    def set_state(self, state):
        self.gym_env.set_state(state)

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
        invert_xy : bool, optional
            Whether to invert x and y translation. Makes it easier to teleop
            in ManiSkill because of the base camera setup, by default True

        Returns
        -------
        SceneObservation, float, bool, dict
            The observation, reward, done flag and info dict.

        Raises
        ------
        Exception
            Do not yet know how ManiSkill handles invalid actions, so raise
            an exception if it occurs in stepping the action.
        """
        prediction_is_quat = action.shape[0] == 8

        if postprocess:
            action = self.postprocess_action(
                action,
                scale_action=scale_action,
                delay_gripper=delay_gripper,
                prediction_is_quat=prediction_is_quat,
            )
        else:
            action = action

        if self.invert_xy:
            # Invert x, y movement and rotation, but not gripper and z.
            action[:2] = -action[:2]
            action[3:-2] = -action[3:-2]

        # zero_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, action[-1]])
        zero_action = np.zeros_like(action)

        if np.isnan(action).any():
            logger.warning("NaN action, skipping")
            action = zero_action

        # NOTE: if stepping fails might be bcs postprocess_action is deactivated
        # should be used now as it also converts quats predicted by the GMM to

        try:
            next_obs, reward, done, _, info = self.gym_env.step(action)
        except Exception as e:
            logger.info("Skipping invalid action {}.".format(action))

            logger.warning("Don't yet know how ManiSkill handles invalid actions")
            raise e

            next_obs, reward, done, _, info = self.gym_env.step(zero_action)

        obs = self.process_observation(next_obs)

        self.render()

        return obs, reward, done, info

    def close(self):
        self.gym_env.close()

    def process_observation(self, obs: dict) -> SceneObservation:
        """
        Convert the observation dict from ManiSkill to a SceneObservation.

        Parameters
        ----------
        obs : dict
            The observation dict from ManiSkill.

        Returns
        -------
        SceneObservation
            The observation in common format as a TensorClass.
        """
        cam_obs = obs["image"]
        cam_names = cam_obs.keys()

        translated_names = [cam_name_tranlation[c] for c in cam_names]
        assert set(self.cameras).issubset(set(translated_names))

        cam_rgb = {
            cam_name_tranlation[c]: cam_obs[c]["Color"][:, :, :3].transpose((2, 0, 1))
            for c in cam_names
        }

        # Negative depth is channel 2 in the position tensor.
        # See https://insiders.vscode.dev/github/vonHartz/ManiSkill2/blob/main/mani_skill2/sensors/depth_camera.py#L100-L101
        cam_depth = {
            cam_name_tranlation[c]: -cam_obs[c]["Position"][:, :, 2] for c in cam_names
        }

        # NOTE channel 0 is mesh-wise, channel 1 is actor-wise, see
        # https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#visualize-segmentation
        cam_mask = {
            cam_name_tranlation[c]: cam_obs[c]["Segmentation"][:, :, 0]
            for c in cam_names
        }

        # Invert extrinsics for consistency with RLBench, Franka. cam2world vs world2cam.
        cam_ext = {
            cam_name_tranlation[c]: invert_homogenous_transform(
                obs["camera_param"][c]["extrinsic_cv"]
            )
            for c in cam_names
        }

        cam_int = {
            cam_name_tranlation[c]: obs["camera_param"][c]["intrinsic_cv"]
            for c in cam_names
        }

        ee_pose = torch.Tensor(obs["extra"]["tcp_pose"])
        object_poses = dict_to_tensordict(
            {
                k: torch.Tensor(v)
                for k, v in obs["extra"].items()
                if k.endswith("pose") and k != "tcp_pose"
            }
        )

        joint_pos = torch.Tensor(obs["agent"]["qpos"])
        joint_vel = torch.Tensor(obs["agent"]["qvel"])

        if joint_pos.shape == torch.Size([7]):
            # For tasks with excavator attached, there's no additional joints
            finger_pose = torch.empty(0)
            finger_vel = torch.empty(0)
        else:
            # NOTE: the last two dims are the individual fingers, but they are
            # forced to be identical.
            # NOTE: switched from using split([7, 2]) (ie enforce 7 joints) to
            # assuming that the last two joints are the fingers and the rest are
            # the arm joints, as mobile manipulation envs seem to have 8 joints.
            joint_pos, finger_pose = joint_pos[:-2], joint_pos[-2:]
            joint_vel, finger_vel = joint_vel[:-2], joint_vel[-2:]

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(self.cameras)}
            | {
                c: SingleCamObservation(
                    **{
                        "rgb": torch.Tensor(cam_rgb[c]),
                        "depth": torch.Tensor(cam_depth[c]),
                        "mask": torch.Tensor(cam_mask[c].astype(np.uint8)).to(
                            torch.uint8
                        ),
                        "extr": torch.Tensor(cam_ext[c]),
                        "intr": torch.Tensor(cam_int[c]),
                    },
                    batch_size=empty_batchsize,
                )
                for c in self.cameras
            }
        )

        obs = SceneObservation(
            cameras=multicam_obs,
            ee_pose=ee_pose,
            object_poses=object_poses,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            gripper_state=finger_pose,
            batch_size=empty_batchsize,
        )

        return obs

    def get_replayed_obs(self):
        # To be used from extract_demo.py
        obs = self.gym_env._episode_data[0]["o"]
        print(obs)
        done = self.gym_env._episode_data[0]["d"]
        reward = self.gym_env._episode_data[0]["r"]
        info = self.gym_env._episode_data[0]["info"]

    def postprocess_quat_action(self, quaternion: np.ndarray) -> np.ndarray:
        return quaternion_to_axis_angle(quaternion)

    def get_inverse_kinematics(
        self,
        target_pose: np.ndarray,
        reference_qpos: np.ndarray,
        max_iterations: int = 100,
    ) -> np.ndarray:
        qpos, success, error = self._pin_model.compute_inverse_kinematics(
            self._ee_link_idx,
            sapien.core.Pose(target_pose[:3], target_pose[3:7]),
            initial_qpos=reference_qpos,
            active_qmask=self._q_mask,
            max_iterations=max_iterations,
        )

        if not success:
            raise ValueError(f"Failed to find IK solution: {error}")

        return qpos

    def get_forward_kinematics(self, qpos: np.ndarray) -> np.ndarray:
        self._pin_model.compute_forward_kinematics(qpos)

        pose = self._pin_model.get_link_pose(self._ee_link_idx)

        return np.concatenate([pose.p, pose.q])

    def reset_joint_pose(
        self,
        joint_pos=[
            -8.2433e-03,
            4.3171e-01,
            -2.0684e-03,
            -1.9697e00,
            -7.5249e-04,
            2.3248e00,
            8.0096e-01,
            0.04,
            0.04,
        ],
    ) -> None:
        self.robot.set_qpos(joint_pos)
