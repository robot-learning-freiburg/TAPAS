import mplib
import numpy as np
from loguru import logger


class MotionPlanner:
    def __init__(self, config, env):
        self.env = env
        self.agent = env.agent
        self.robot = env.robot

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        urdf_path = self.agent._get_urdf_path()
        srdf_path = self.agent._get_srdf_path()

        self.move_group_idx = 10
        move_group_name = self.robot.get_links()[self.move_group_idx].get_name()

        self.planner = mplib.Planner(
            urdf=urdf_path,
            srdf=srdf_path,
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=move_group_name,
            joint_vel_limits=np.ones(7),
            joint_acc_limits=np.ones(7),
        )

        self.active_joints = self.robot.get_active_joints()

    @staticmethod
    def _check_success(result):
        return result["status"] == "Success"

    def plan_to_goal(self, current_pose, goal_pose, time_step=1 / 250, with_screw=True):
        success = False
        plan = {}

        if with_screw:
            plan = self.planner.plan_screw(goal_pose, current_pose, time_step=time_step)
            success = self._check_success(plan)

            if not success:
                logger.warning("Planning with screw failed. Trying again without.")

        if not with_screw or not success:
            plan = self.planner.plan_qpos_to_pose(
                goal_pose, current_pose, time_step=time_step
            )
            success = self._check_success(plan)

        if not success:
            logger.error("Planning failed.")

            plan["position"] = []
            plan["velocity"] = []

        # Unpack ndarrays into lists to allow pop
        plan["position"] = [p for p in plan["position"]]
        plan["velocity"] = [v for v in plan["velocity"]]
        plan["acceleration"] = [a for a in plan["acceleration"]]
        plan["time"] = [t for t in plan["time"]]

        return plan
