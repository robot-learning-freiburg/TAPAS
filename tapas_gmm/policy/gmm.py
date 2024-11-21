from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from riepybdlib.mappings import s1_log_e
from tqdm.auto import tqdm

from tapas_gmm.dataset.demos import get_frame_transform_flat, get_frames_from_obs
from tapas_gmm.encoder.encoder import ObservationEncoder, ObservationEncoderConfig
from tapas_gmm.env.environment import (
    BaseEnvironment,
    RestoreActionMode,
    RestoreEnvState,
)
from tapas_gmm.policy.models.tpgmm import TPGMM, AutoTPGMM, AutoTPGMMConfig, TPGMMConfig

# from tapas_gmm.policy.motion_planner import MotionPlannerPolicy
from tapas_gmm.policy.policy import Policy, PolicyConfig
from tapas_gmm.utils.geometry_np import (
    axis_and_angle_to_quaternion,
    compute_angle_between_quaternions,
    conjugate_quat,
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    overlapping_split,
    quaternion_diff,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_pose_diff,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    rotate_vector_by_quaternion,
)
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.robot_trajectory import RobotTrajectory, TrajectoryPoint
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.topp import TOPP  # , SapienScene
from tapas_gmm.viz.gmm import plot_traj_topp

zero_pos = np.array([0, 0, 0])
zero_quat = np.array([1, 0, 0, 0])

close_gripper = torch.tensor([-0.9])
open_gripper = torch.tensor([0.9])


@dataclass
class GMMPolicyConfig(PolicyConfig):
    model: Any  # AutoTPGMMConfig | TPGMMConfig

    force_overwrite_checkpoint_config: bool = False

    time_based: bool | None = None

    predict_dx_in_xdx_models: bool = False

    batch_predict_in_t_models: bool = True
    batch_t_max: float = 1
    topp_in_t_models: bool = True
    return_full_batch: bool = False

    time_scale: float = 1  # 0.25
    pos_lag_thresh: float | None = 0.02
    quat_lag_thresh: float | None = 0.1
    pos_change_thresh: float | None = 0.002
    quat_change_thresh: float | None = 0.002

    topp_supersampling: float = 0.15

    dbg_prediction: bool = False

    binary_gripper_action: bool = False
    binary_gripper_closed_threshold: float = 0

    encoder: Any = None
    obs_encoder: ObservationEncoderConfig = ObservationEncoderConfig()
    encoder_path: str = "demos_vit_keypoints_encoder"

    postprocess_prediction: bool = True

    invert_prediction_batch: bool = False

    # visual_embedding_dim: int | None
    # proprio_dim: int
    # action_dim: int
    # lstm_layers: int

    # use_ee_pose: bool
    # add_gripper_state: bool

    # training: PolicyTrainingConfig | None


class GMMPolicy(Policy):
    def __init__(
        self, config: GMMPolicyConfig, skip_module_init: bool = False, **kwargs
    ):
        super().__init__(config, skip_module_init, **kwargs)

        self.config = config

        Model = AutoTPGMM if isinstance(config.model, AutoTPGMMConfig) else TPGMM

        self.model = Model(config.model)

        self._time_based = config.time_based
        self._time_scale = config.time_scale
        self._pos_lag_thresh = config.pos_lag_thresh
        self._quat_lag_thresh = config.quat_lag_thresh
        self._pos_change_thresh = config.pos_change_thresh or -1
        self._quat_change_thresh = config.quat_change_thresh or -1
        self._predict_dx_in_xdx_models = config.predict_dx_in_xdx_models
        self._force_overwrite_checkpoint_config = (
            config.force_overwrite_checkpoint_config
        )

        self._model_contains_time_dim = None
        self._model_contains_action_dim = None
        self._model_factorizes_action = None

        self._local_marginals = None

        self._add_init_ee_pose_as_frame = None
        self._add_world_frame = None

        self._last_prediction = None
        self._last_pose = None

        self._env = None
        self._pin_model = None

        self._t_curr = None
        self._prediction_batch: RobotTrajectory | None = None

        self.obs_encoder = ObservationEncoder(
            self.config.obs_encoder, self.config.encoder_path
        )

    @property
    def _model_is_txdx(self) -> bool:
        return self._model_contains_time_dim and self._model_contains_action_dim

    @property
    def _prediction_is_delta_pose(self) -> bool:
        return (
            self._model_is_txdx
            and (self._predict_dx_in_xdx_models or not self._time_based)
        ) or not self._model_contains_time_dim

    @property
    def _fix_frames(self) -> bool:
        return self.model._fix_frames

    @property
    def frames_from_keypoints(self) -> bool:
        return self.model._demos.meta_data["frames_from_keypoints"]

    @property
    def kp_indeces(self) -> list[int] | None:
        return self.model._demos.meta_data["kp_indeces"]

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self._t_curr = np.array([0.0])

        self.model.reset_episode()

        self._env = env

        if self.config.batch_predict_in_t_models:
            self._prediction_batch = None

        self._last_prediction = None
        self._last_pose = None

    def get_frames(self, obs: SceneObservation) -> tuple[np.ndarray, np.ndarray]:
        return get_frames_from_obs(
            obs=obs,
            frames_from_keypoints=self.frames_from_keypoints,
            add_init_ee_pose_as_frame=self._add_init_ee_pose_as_frame,
            add_world_frame=self._add_world_frame,
            indeces=self.kp_indeces,
            add_action_dim=self._model_contains_action_dim,
        )

    # TODO: clean up and refactor (disentangle the nested cases)
    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray | RobotTrajectory, dict]:
        info = {}

        frame_trans, frame_quats, viz_encoding = self._get_frame_trans(obs)

        info["viz_encoding"] = viz_encoding

        if self.config.batch_predict_in_t_models and self._time_based:
            if self._prediction_batch is None:
                self._prediction_batch = self._create_prediction_batch(
                    obs=obs, frame_trans=frame_trans, frame_quats=frame_quats
                )
                self._env.publish_path(self._prediction_batch)

                if self.config.return_full_batch:
                    info["done"] = True

                    if self.config.binary_gripper_action:
                        self._prediction_batch.gripper = self._binary_gripper_action(
                            self._prediction_batch.gripper
                        )

                    return self._prediction_batch, info

            if self._prediction_batch.is_finished:
                # action = self._make_noop_plan(obs.cpu(), duration=1)[0]
                # info = {"done": True}
                prediction = (
                    self._last_prediction
                    if self._last_prediction is not None
                    else self._make_noop_plan(obs.cpu(), duration=1)[0]
                )
                info["done"] = self._last_prediction is None
            else:
                prediction = self._prediction_batch.step()
                info["done"] = False
            action = (
                self._postprocess_prediction(obs.ee_pose.numpy(), prediction.ee)
                if self.config.postprocess_prediction
                else prediction
            )

            self._last_prediction = prediction

        else:
            _, action, _ = self._predict_and_step(
                obs=obs,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                postprocess=self.config.postprocess_prediction,
            )
            info["done"] = False

        if self.config.binary_gripper_action:
            action[-1] = self._binary_gripper_action(action[-1])

        return action, info

    def _get_frame_trans(self, obs):
        if self.model._online_first_step or not self._fix_frames:
            if self.config.obs_encoder.image_encoder is not None:
                kp, enc_info = self.obs_encoder.get_image_encoding(
                    obs.to(device).unsqueeze(0)
                )
                obs.kp = kp.cpu().squeeze(0)

                viz_encoding = [enc_info["kp_raw_2d"][0].squeeze(0).cpu()]
            else:
                viz_encoding = None

            frame_trans, frame_quats = self.get_frames(obs)
            self._env.publish_frames(frame_trans, frame_quats)
        else:
            viz_encoding = None

            frame_trans, frame_quats = None, None

        return frame_trans, frame_quats, viz_encoding

    def _binary_gripper_action(
        self,
        gripper_state: float | np.ndarray,
    ) -> float | np.ndarray:
        if isinstance(gripper_state, np.ndarray):
            return np.where(
                gripper_state < self.config.binary_gripper_closed_threshold, -1, 1
            )
        else:
            return (
                -1 if gripper_state < self.config.binary_gripper_closed_threshold else 1
            )

    # NOTE: copied here to avoid import from MotionPlannerPolicy, to avoid ManiSkill2 dependency
    # TODO: refactor
    @staticmethod
    def _normalize_gripper_state(
        gripper_state: torch.Tensor, finger_width: float = 0.04
    ) -> torch.Tensor:
        # average the two fingers - should be identical anyway
        gripper_state_1d = gripper_state.mean().unsqueeze(0)

        # map from [0, finger_width] to [-1, 1]
        return gripper_state_1d * 2 / finger_width - 1

    @staticmethod
    def _binary_gripper_state(
        gripper_state: torch.Tensor, closed_threshold: float = 0.85
    ) -> torch.Tensor:
        normalized_state = GMMPolicyConfig._normalize_gripper_state(gripper_state)

        return torch.where(
            normalized_state < closed_threshold, close_gripper, open_gripper
        )

    def _make_noop_plan(self, obs: SceneObservation, duration: int = 5):
        """
        Returns a plan for doing nothing for a given number of timesteps.

        Actions are given as EE Delta Pose and normalized gripper state.

        Parameters
        ----------
        obs : SceneObservation
            Current observation.
        duration : int, optional
            Number of timesteps for which to do nothing, by default 5

        Returns
        -------
        list[torch.Tensor]
            Plan for doing nothing as EE Delta Pose and gripper action.
        """
        current_gripper = GMMPolicy._binary_gripper_state(obs.gripper_state)

        return [np.concatenate((zero_pos, zero_quat, current_gripper))] * duration

    def _create_prediction_batch(
        self,
        obs: SceneObservation,
        frame_trans: np.ndarray | None,
        frame_quats: np.ndarray | None,
    ) -> RobotTrajectory:
        """
        For time-based models only. Predict the full trajectory at once.
        Allows to apply TOPP(RA).
        """

        assert self._time_based and self._fix_frames

        inputs = []
        prediction_raw = []
        prediction_tan = []
        first_step = True

        while self._t_curr <= self.config.batch_t_max:
            # TODO: needs to return a TrajectoryPoint too, so that in RobotTrajectory.from_np
            # we can assign the gripper state properly as well.
            inp, pred, extra = self._predict_and_step(
                obs=obs,
                frame_trans=frame_trans if first_step else None,
                frame_quats=frame_quats if first_step else None,
                always_step=True,
                postprocess=False,
            )
            prediction_raw.append(pred)
            prediction_tan.append(extra["mu_tangent"])
            inputs.append(inp)

            first_step = False

        logger.info(f"Predicted {len(prediction_raw)} steps in batch.")

        if self.config.invert_prediction_batch:
            prediction_raw = prediction_raw[::-1]

        stacked_pred = np.stack(prediction_raw)
        raw_traj = RobotTrajectory.from_np(
            ee=stacked_pred[:, :7], gripper=stacked_pred[:, 7:]
        )

        # if self.config.invert_prediction_batch:
        #     raw_traj = raw_traj.invert()

        if self.config.dbg_prediction:
            # TODO: tangent proj rotation for plot!
            full_rec = np.concatenate(
                (np.stack(inputs), np.stack(prediction_tan)), axis=1
            )
            frame_origs = np.concatenate((frame_trans[:, :3, 3], frame_quats), axis=1)
            self.model.plot_reconstructions(
                marginals=[self.model._online_trans_margs_joint],
                joint_models=[self.model._online_hmm_cascade],
                reconstructions=[full_rec],
                frame_orig_wquats=np.expand_dims(frame_origs, 0),
                original_trajectories=None,
                plot_trajectories=False,
                plot_reconstructions=True,
                plot_gaussians=True,
                time_based=True,
                equal_aspect=False,
                per_segment=False,
            )

        if self.config.topp_in_t_models:
            assert (
                not self._model_contains_action_dim
            ), "TOPP not implemented for delta models or TXDX models."
            # NOTE: for delta models would need to first reconstruct the full trajectory
            # for XDX models: prediction is stacked state and action and only teased
            # apart in the postprocessing step and then do the reconstruction if we
            # were to use the DX part.
            initial_qpos = obs.joint_pos.numpy()
            # Add gripper state HACK
            initial_qpos = np.concatenate((initial_qpos, np.array([0.04, 0.04])))

            # print(initial_qpos, self._env._arm_controller.articulation.get_qpos())
            # if hasattr(self._env, "_arm_controller"):  # sanity check
            #     assert np.allclose(
            #         initial_qpos, self._env._arm_controller.articulation.get_qpos()
            #     )
            logger.info(f"Initial ee pose: {obs.ee_pose.numpy()}")

            # TODO: make use of RobotTrajectory class to pass the trajectory
            topp_traj, topp_info = self._topp(
                ee_poses_wgripper=np.stack(prediction_raw),
                initial_qpos=initial_qpos,
                supersampling=self.config.topp_supersampling,
            )

            if self.config.dbg_prediction:
                plot_traj_topp(
                    np.stack(prediction_raw),
                    topp_traj.ee,
                    frame_origs,
                    topp_info["split_idcs"],
                )

            if not topp_traj:
                logger.warning("TOPP failed. Returning raw trajectory.")
                return raw_traj

            # return ee_poses_topp
            # return [x for x in topp_info["qposes_topp"]]
            return topp_traj
        else:
            return raw_traj

    def _predict_and_step(
        self,
        obs: SceneObservation,
        frame_trans: np.ndarray | None,
        frame_quats: np.ndarray | None,
        always_step: bool = False,
        postprocess: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Predict the next action and step the time if needed.

        Returns: input data, action, extra dict
        """

        ee_pose = obs.ee_pose.numpy()
        input_data = np.clip(self._t_curr, 0, 1) if self._time_based else ee_pose

        prediction, extra = self.model.online_predict(
            input_data=input_data,
            time_based=self._time_based,
            frame_trans=frame_trans,
            frame_quats=frame_quats,
            local_marginals=self._local_marginals,
        )

        if postprocess:
            action = self._postprocess_prediction(ee_pose, prediction)
        else:
            action = prediction

        self._manage_time_step(ee_pose, always_step=always_step)

        self._last_prediction = prediction
        self._last_pose = ee_pose

        return input_data, action, extra

    def _manage_time_step(self, ee_pose: np.ndarray, always_step: bool = False) -> None:
        """
        Helper function for time-driven models to step the time if needed.
        """
        if self._should_time_step(ee_pose, always_step):
            self._t_curr = self.model._online_step_time(
                self._t_curr, time_scale=self._time_scale
            )

    def _should_time_step(self, ee_pose: np.ndarray, always_step: bool = False):
        if self._last_prediction is not None and not always_step:
            ee_f_b, ee_quat = ee_pose[:3], ee_pose[3:]
            pos_lag = ee_f_b - self._last_prediction[:3]
            pos_change = self._last_pose[:3] - ee_f_b

            if self._model_contains_rotation:
                quat_lag = compute_angle_between_quaternions(
                    ee_quat, self._last_prediction[3:7]
                )
                quat_change = compute_angle_between_quaternions(
                    self._last_pose[3:], ee_quat
                )
            else:
                quat_lag = None
                quat_change = None

            logger.info(
                f"Pos lag: {pos_lag}, quat lag: {quat_lag}, "
                + f"pos change {pos_change}, quat change {quat_change}",
                filter=False,
            )
        else:
            pos_lag = None

        action_succes = (
            not always_step
            and self._last_prediction is not None
            and (
                self._pos_lag_thresh is None
                or np.linalg.norm(pos_lag) < self._pos_lag_thresh
            )
            and (quat_lag is None or (quat_lag < self._quat_change_thresh))
        )

        ee_stuck = (
            not always_step
            and self._last_pose is not None
            and (
                self._pos_change_thresh is None
                or np.linalg.norm(pos_change) < self._pos_change_thresh
            )
            and (quat_change is None or (quat_change < self._quat_change_thresh))
        )

        do_step = (
            always_step
            or self._prediction_is_delta_pose
            or self._pos_lag_thresh is None
            or action_succes
            or ee_stuck
        )

        return do_step

    def _postprocess_prediction(
        self,
        ee_pose: np.ndarray,
        prediction: np.ndarray,
    ) -> np.ndarray:
        """
        Actions might be poses, pose delta, or factorized delta.
        Convert to pose delta.
        Should skip this step in batch mode.
        """
        # Split gripper and EE part
        if self._model_contains_gripper_action:
            gripper_action = prediction[-1:]
            ee_prediction = prediction[:-1]
        else:
            gripper_action = -np.ones((1))
            ee_prediction = prediction

        state_dim = 7 if self._model_contains_rotation else 3
        action_dim = (
            9
            if self._model_factorizes_action
            else 7 if self._model_contains_rotation else 3
        )

        # Split EE part into state and action if needed
        if self._model_is_txdx and self._time_based:
            # TXDX model, ie contains time, state and action. Can use either x or dx.
            assert ee_prediction.shape == (state_dim + action_dim,)

            ee_prediction = ee_prediction[
                (
                    slice(state_dim, None)
                    if self._predict_dx_in_xdx_models
                    else slice(0, state_dim)
                )
            ]

        else:
            ee_dim = action_dim if self._prediction_is_delta_pose else state_dim
            assert ee_prediction.shape == (ee_dim,)

        if self._prediction_is_delta_pose:
            if self._model_factorizes_action:
                # Prediction is factorized -> reassemble delta pose
                pos_dir, rot_dir, pos_mag, rot_mag = np.split(ee_prediction, (3, 6, 7))

                # Transform directions from world to EE frame
                ee2world = homogenous_transform_from_rot_shift(
                    quaternion_to_matrix(ee_pose[3:]), np.zeros(3)
                )
                world2ee = invert_homogenous_transform(ee2world)
                world2ee_quat = conjugate_quat(ee_pose[3:])

                action_hom = homogenous_transform_from_rot_shift(np.eye(3), pos_dir)

                pos_dir_ee = (world2ee @ action_hom)[:3, 3]
                rot_dir_ee = rotate_vector_by_quaternion(rot_dir, world2ee_quat)

                pos_delta = pos_dir_ee * pos_mag

                rot_angle = s1_log_e(rot_mag)
                rot_delta = axis_and_angle_to_quaternion(rot_dir_ee, rot_angle)

            else:  # prediction is delta pose -> use directly
                pos_delta = ee_prediction[:3]
                rot_delta = (
                    ee_prediction[3:7] if self._model_contains_rotation else zero_quat
                )

                # Transform Delta Pose from world to EE frame
                pos_delta, rot_delta = self._delta_frame_transform(
                    ee_pose, pos_delta, rot_delta
                )

        else:  # prediction is absolute pose -> get finite difference
            pos_delta, rot_delta = self._pose_to_pose_delta(ee_pose, ee_prediction)

        if not self._model_contains_rotation:
            rot_delta = zero_quat

        ee_action = np.concatenate([pos_delta, rot_delta])

        return np.concatenate([ee_action, gripper_action])

    def _delta_frame_transform(
        self, ee_pose: np.ndarray, pos_delta: np.ndarray, rot_delta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform delta pose from world to EE frame.
        """
        action_R = quaternion_to_matrix(rot_delta)
        act2world = homogenous_transform_from_rot_shift(action_R, pos_delta)
        # world2act = invert_homogenous_transform(act2world)

        ee2world = homogenous_transform_from_rot_shift(
            quaternion_to_matrix(ee_pose[3:]), np.zeros(3)
        )
        world2ee = invert_homogenous_transform(ee2world)

        act2ee = world2ee @ act2world

        pos_local = act2ee[:3, 3]

        rot_local = quaternion_pose_diff(ee_pose[3:], rot_delta)
        # rot_local = quaternion_from_matrix(act2ee[:3, :3])

        return pos_local, rot_local

    def _pose_to_pose_delta(
        self, ee_pose: np.ndarray, ee_prediction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given the current pose and predicted next pose, compute the delta pose.
        Returns delta position and quaternion.
        """
        pred_b = ee_prediction[:3]
        if self._model_contains_rotation:
            pred_q = ee_prediction[3:7]
        else:
            pred_q = zero_quat
        pred_A = quaternion_to_matrix(pred_q)
        pred_hom = homogenous_transform_from_rot_shift(pred_A, pred_b)

        # Transform from world into EE frame. In EE frame target pose and delta pose
        # are the same thing.
        ee_f_b, ee_quat = ee_pose[:3], ee_pose[3:]
        ee_f_A = quaternion_to_matrix(ee_quat)

        world2ee = invert_homogenous_transform(
            homogenous_transform_from_rot_shift(ee_f_A, ee_f_b)
        )
        rot_delta = quaternion_pose_diff(ee_quat, pred_q)

        pred_local = world2ee @ pred_hom
        pos_delta = pred_local[:3, 3]
        # rot_delta = quaternion_from_matrix(pred_local[:3, :3])

        return pos_delta, rot_delta

    def _pose_delta_to_pose(
        self, ee_pose: np.ndarray, ee_delta_pose: np.ndarray
    ) -> np.ndarray:
        """
        Given the current pose and delta pose, compute the next pose.
        """
        ee_f_b, ee_quat = ee_pose[:3], ee_pose[3:]
        ee_f_A = quaternion_to_matrix(ee_quat)

        ee2world = homogenous_transform_from_rot_shift(ee_f_A, ee_f_b)
        pred_local = homogenous_transform_from_rot_shift(
            quaternion_to_matrix(ee_delta_pose[3:7]), ee_delta_pose[:3]
        )

        pred_world = ee2world @ pred_local
        pred_f_b = pred_world[:3, 3]
        pred_f_A = quaternion_from_matrix(pred_world[:3, :3])
        # pred_f_A = quaternion_multiply(ee_quat, ee_delta_pose[3:7])

        return np.concatenate([pred_f_b, pred_f_A])

    def _topp(
        self,
        ee_poses_wgripper: np.ndarray,
        initial_qpos: np.ndarray,
        supersampling: float = 0.15,
        repeat_segment_final_pose: int = 10,
        # gripper_delta_thresh: float = 0.5,
    ) -> tuple[RobotTrajectory, dict]:
        """
        Use TOPP(RA) to smooth the trajectory.
        """
        if self._model_contains_gripper_action:
            ee_poses = ee_poses_wgripper[:, :-1]
            gripper_states = ee_poses_wgripper[:, -1:]
        else:
            ee_poses = ee_poses_wgripper
            gripper_states = np.zeros((ee_poses.shape[0], 1))

        pos_only = ee_poses.shape[1] == 3

        try:
            qposes = self._eeposes_to_qposes(initial_qpos, ee_poses)
        except ValueError as e:
            logger.warning("Failed to find IK solution for trajectory. Skipping.")
            return [], {}

        # import matplotlib.pyplot as plt
        # for i in range(7):
        #     plt.plot(qposes[:, i])
        # plt.show()

        # Get indeces where gripper state changes -> does not work well for small dt
        # split_idcs = (
        #     np.argwhere(
        #         np.abs(np.diff(gripper_states.squeeze(1), axis=0))
        #         > gripper_delta_thresh
        #     ).squeeze(1)
        #     + 1
        # )

        # NOTE: using the activation of the segment gmms to estimate segment borders
        # would be most elegant, but using the relative duration of the segments
        # is a decent approximation.
        # split_idcs = np.cumsum([int(s.relative_duration * len(qposes)) for s in self.model._demos_segments][:-1])
        split_idcs = np.cumsum(
            [
                int(s.mean_traj_len / self._time_scale)
                for s in self.model._demos_segments
            ]
        )[:-1]

        # TODO TODO TODO: does not quite work yet.

        info = {"split_idcs": split_idcs}

        # # Get indeces where gripper state crosses zero, ie changes sign
        # split_idcs = np.where(np.diff(np.sign(gripper_states.squeeze(1))))[0]
        # Split trajectory into overlapping segments (split idx is in both)
        qposes_segments = overlapping_split(qposes, split_idcs)
        gripper_segments = overlapping_split(gripper_states, split_idcs)

        planner = TOPP(
            urdf_path=self._env._urdf_path,
            srdf_path=self._env._srdf_path,
            move_group=self._env._move_group,
        )

        traj_segments = []

        for poses, gripper in tqdm(
            zip(qposes_segments, gripper_segments), desc="Segment TOPP"
        ):
            seg = planner.plan_segment(
                poses=poses,
                gripper=gripper,
                supersampling=supersampling,
                repeat_segment_final_pose=repeat_segment_final_pose,
                time_scale=self._time_scale,
            )
            traj_segments.append(seg)

        # trajectory = traj_segments[0]

        trajectory = RobotTrajectory.concatenate(traj_segments)

        # Convert qposes back to end-effector poses and add gripper state
        trajectory.ee = self._qposes_to_eeposes(trajectory.q, trajectory.gripper)

        if pos_only:
            trajectory.ee = np.stack(
                [np.concatenate((a[:3], np.zeros(4), a[3:])) for a in trajectory.ee]
            )

        return trajectory, info

    def _qposes_to_eeposes(
        self, qposes_topp: np.ndarray, gripper_topp: np.ndarray
    ) -> np.ndarray:
        actions_topp = []

        # RLBench does not support virtual computation of the robot state
        # So need to step in the scene and restore the state
        with RestoreEnvState(self._env), RestoreActionMode(self._env):
            for qpos, gr in tqdm(
                zip(qposes_topp, gripper_topp),
                desc="Forward kinematics",
                total=qposes_topp.shape[0],
            ):
                # Add dummy gripper state for forward kinematics
                pose = self._env.get_forward_kinematics(
                    np.concatenate((qpos, np.zeros(2)))
                )

                actions_topp.append(np.concatenate([pose, gr]))

        return np.stack(actions_topp)

    def _eeposes_to_qposes(
        self, initial_qpos: np.ndarray, ee_poses: np.ndarray
    ) -> np.ndarray:
        """
        Get joint positions for a given end-effector trajectory.
        """
        qposes = []

        # robot_pose = self._env._get_robot_base_pose()
        # logger.info(f"robot_pose: {robot_pose}")
        # sapien_scene = SapienScene(robot_pose)

        # See _qposes_to_eeposes for info on context manager
        with RestoreEnvState(self._env):
            for target_pose in tqdm(ee_poses, desc="Inverse kinematics"):
                ref_q = initial_qpos if len(qposes) == 0 else qposes[-1]

                try:
                    qpos = self._env.get_inverse_kinematics(target_pose, ref_q)
                    # qpos = sapien_scene.get_inverse_kinematics(target_pose, ref_q)
                    assert len(qpos.shape) == 1
                    # logger.info(f"Found IK solution for target pose {target_pose}")
                except Exception as e:
                    logger.warning(
                        f"Failed to find IK solution for target pose {target_pose}"
                    )
                    raise e

                qposes.append(qpos)

        qposes_stack = np.stack(qposes)[:, :7]

        return qposes_stack

    def from_disk(self, chekpoint_path: str) -> None:
        self.model.from_disk(
            chekpoint_path,
            force_config_overwrite=self._force_overwrite_checkpoint_config,
        )

        self._add_init_ee_pose_as_frame = self.model._demos.meta_data[
            "add_init_ee_pose_as_frame"
        ]
        self._add_world_frame = self.model._demos.meta_data["add_world_frame"]

        self._model_contains_rotation = self.model.add_rotation_component
        self._model_contains_time_dim = self.model.add_time_component
        self._model_contains_action_dim = self.model.add_action_component
        self._model_contains_gripper_action = self.model.add_gripper_action
        self._model_factorizes_action = self.model.action_as_orientation
        if self._model_factorizes_action:
            assert self._model_contains_action_dim
            assert self.model.action_with_magnitude, "Need magnitude for factorization."

        if self._time_based is None:
            self._time_based = self._model_contains_time_dim
            logger.info(
                "Detected time-based model: {}. Using time-driven policy. "
                "Set time_based in config to overwrite.",
                self._time_based,
            )

        # assert self._model_contains_time_dim == self._time_based, (
        #     "Model contains time dimension, but policy is not time-based, or vice "
        #     "versa. TODO: implement case where model is time-based, but policy is "
        #     "not."
        # )

        if self._time_based and self._model_contains_action_dim:
            logger.info(
                "Got TXDX model, flag to be time-driven and flag to "
                + "prediction action directly (TDX)."
                if self._predict_dx_in_xdx_models
                else "predict action from state (TX)."
            )

        logger.info("Creating local marginals")
        self._local_marginals = self.model.get_frame_marginals(
            time_based=self._time_based
        )
