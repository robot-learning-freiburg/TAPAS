# import mani_skill2

import numpy as np
from loguru import logger

from tapas_gmm.utils.robot_trajectory import RobotTrajectory

# from tapas_gmm.utils.geometry_np import (
#     conjugate_quat,
#     homogenous_transform_from_rot_shift,
#     invert_homogenous_transform,
#     quaternion_pose_diff,
#     quaternion_to_matrix,
# )

# from tapas_gmm.utils.debug import save_q_traj_dbg

try:
    import mplib
except ImportError:
    logger.warning("mplib not found, TOPP not available.")
    mplib = None


class TOPP:
    def __init__(self, urdf_path: str, srdf_path: str, move_group: str):
        assert mplib is not None

        self.planner = mplib.Planner(  # see motion_planner.py for details
            urdf=urdf_path,
            srdf=srdf_path,
            move_group=move_group,
        )

    def plan_segment(
        self,
        poses: np.ndarray,
        gripper: np.ndarray,
        supersampling: float,
        repeat_segment_final_pose: int,
        td: float = 0.05,
        min_ss_length: int = 3,
        threshold: float = 0.0001,
        time_scale: float = 1.0,
        keep_first: int = 10,
    ) -> RobotTrajectory:
        """
        Plan a single segment of the trajectory using TOPP.
        Subsamples the gripper traj to match the length of the poses.
        """
        # Subsample trajectory for TOPP to have a tractable length
        t_orig = poses.shape[0]
        ss_len = max(min_ss_length, int(t_orig * supersampling))
        q_ss_idcs = np.linspace(keep_first, t_orig - 1, ss_len).astype(int)
        q_ss_idcs = np.concatenate(
            (np.linspace(0, keep_first - 1, keep_first).astype(int), q_ss_idcs)
        )
        poses_ss = poses[q_ss_idcs]

        # print("ss", poses_ss.shape)
        # save_q_traj_dbg(poses_ss, "/home/hartzj/MT-GMM/test/data/poses_ss_coffee")

        # Filter out poses that are too close to each other for TOPPRA
        mask = np.ones(poses_ss.shape[0], dtype=bool)
        last_val = poses_ss[0]
        for i in range(1, poses_ss.shape[0] - 1):
            if np.linalg.norm(poses_ss[i] - last_val) < threshold:
                mask[i] = False
            else:
                last_val = poses_ss[i]
        poses_filtered = poses_ss[mask]

        # poses_filtered = poses_ss

        duration = t_orig * td / time_scale

        try:
            ttimes, tpos, tvel, tacc, tduration = self.planner.TOPP_SD(
                poses_filtered, duration, td, verbose=False
            )
        except (AttributeError, RuntimeError):
            logger.warning("TOPP failed for segment. Returning original.", filter=False)
            return poses, gripper

        # Supersample gripper actions to match the length of the trajectory
        g_ss_idcs = np.linspace(0, t_orig - 1, tpos.shape[0]).astype(int)
        gripper_ss = gripper[g_ss_idcs]
        # Repeat the final pose n times to ensure the gripper state is reached
        # Do so after the sample matching so the final gripper action is repeated
        tpos = np.concatenate(
            (tpos, np.repeat(tpos[-1:], repeat_segment_final_pose, axis=0))
        )
        gripper_ss = np.concatenate(
            (
                gripper_ss,
                np.repeat(gripper_ss[-1:], repeat_segment_final_pose, axis=0),
            )
        )

        traj = RobotTrajectory.from_np(
            t=ttimes, q=tpos, qd=tvel, qdd=tacc, gripper=gripper_ss, duration=tduration
        )

        return traj
