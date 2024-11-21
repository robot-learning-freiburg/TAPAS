import threading
from multiprocessing import RLock
from typing import Callable

import numpy as np
import rospy
from loguru import logger
from sensor_msgs.msg import JointState as JointStateMsg
from tqdm.auto import tqdm
from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint as JointTrajectoryPointMsg

from tapas_gmm.utils.robot_trajectory import RobotTrajectory

integer_infinity = np.iinfo(np.int32).max


class JointPoitionCommander(object):
    def __init__(self, command_topic, joints) -> None:
        self._joints = joints
        self._goal = None
        self._last_q = None
        self._lock = RLock()

        self.pub_command = rospy.Publisher(
            command_topic, JointStateMsg, queue_size=1, tcp_nodelay=True
        )

        self.sub_js = rospy.Subscriber(
            "/joint_states",
            JointStateMsg,
            callback=self._cb_js,
            queue_size=1,
            tcp_nodelay=True,
        )

    def _cb_js(self, msg):
        js = dict(zip(msg.name, msg.position))

        temp = np.zeros(len(self._joints))
        for x, j in enumerate(self._joints):
            temp[x] = js[j]

        with self._lock:
            self._last_q = temp

    def set_goal(self, goal):
        if len(goal) != len(self._joints):
            raise ValueError(
                f"Goal is of wrong size: Expected {len(self._joints)} got {len(goal)}"
            )

        self._goal = goal
        self.pub_command.publish(
            JointStateMsg(name=self._joints, position=list(self._goal))
        )

    @property
    def joints(self):
        return self._joints

    @property
    def goal(self):
        return self._goal

    @property
    def q(self):
        return self._last_q

    @property
    def delta(self):
        with self._lock:
            return self._goal - self._last_q

    def reached_goal(self, tolerance=1e-3):
        if self._goal is not None and self._last_q is None:
            return False
        return np.abs(self.delta).max() < tolerance


class JointTrajectoryFollower:
    def __init__(self, gripper_cb: Callable) -> None:
        self.pub = rospy.Publisher(
            "/controllers/position_joint_trajectory_controller/command",
            JointTrajectoryMsg,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.commander = JointPoitionCommander(
            "/controllers/joint_position_controller/command",
            [f"panda_joint{x + 1}" for x in range(7)],
        )

        logger.info("Waiting for joint states...")
        while self.commander.q is None:
            rospy.sleep(0.1)

        self.gripper_cb = gripper_cb

        logger.info(" .. done.")

    def follow(self, trajectory: RobotTrajectory, accuracy: float = 0.01) -> None:
        # get indeces where gripper changes sign
        gripper_indices = np.where(np.diff(np.sign(trajectory.gripper.squeeze(1))))[0]

        segments = trajectory.split(gripper_indices)

        for s, segment in tqdm(
            enumerate(segments), total=len(segments), desc="Executing segments"
        ):
            self._follow_segment(segment, accuracy)
            if s < len(segments) - 1:
                self._execute_gripper_action(segments[s + 1].gripper[0])

    def follow_time_based(
        self, trajectory: RobotTrajectory, accuracy: float = 0.01
    ) -> None:
        # get indeces where gripper changes sign
        gripper_indices = np.where(np.diff(np.sign(trajectory.gripper.squeeze(1))))[0]

        gripper_times = trajectory.t[gripper_indices]
        gripper_commands = trajectory.gripper[gripper_indices]

        c_q = self.commander.q
        initial_delta = trajectory.q[0] - c_q
        duration = np.abs((initial_delta / 0.1)).max()

        initial_goal = JointTrajectoryMsg(
            joint_names=self.commander.joints,
            points=[
                JointTrajectoryPointMsg(positions=c_q, velocities=np.zeros_like(c_q)),
                JointTrajectoryPointMsg(
                    positions=trajectory.q[0],
                    velocities=np.zeros_like(c_q),
                    time_from_start=rospy.Duration(duration),
                ),
            ],
        )
        initial_goal.header.stamp = rospy.Time.now()

        self.pub.publish(initial_goal)
        rospy.sleep(duration)

        full_traj = JointTrajectoryMsg(
            joint_names=self.commander.joints,
            points=[
                JointTrajectoryPointMsg(positions=q, time_from_start=rospy.Duration(t))
                for t, q, qd in zip(trajectory.t, trajectory.q, trajectory.qd)
            ],
        )

        while not rospy.is_shutdown() and (
            self.commander.q is None
            or np.abs(trajectory.q[0] - self.commander.q).max() > accuracy
        ):
            rospy.sleep(0.02)

        start_time = rospy.Time.now()

        full_traj.header.stamp = start_time
        self.pub.publish(full_traj)

        gripper_idx = 0
        next_gripper_time = gripper_times[gripper_idx]
        next_gripper_action = gripper_commands[gripper_idx]

        while (
            not rospy.is_shutdown()
            and np.abs(trajectory.q[-1] - self.commander.q).max() > accuracy
        ):
            if rospy.Time.now() - start_time >= rospy.Duration(next_gripper_time):
                self._execute_gripper_action(next_gripper_action)
                gripper_idx += 1
                if gripper_idx < len(gripper_times):
                    next_gripper_time = gripper_times[gripper_idx]
                    next_gripper_action = gripper_commands[gripper_idx]
                else:
                    next_gripper_time = integer_infinity
            rospy.sleep(0.05)

    def _execute_gripper_action(self, action: float) -> None:
        self.gripper_cb(1 - action * 2)

    def _follow_segment(self, trajectory: RobotTrajectory, accuracy: float) -> None:
        c_q = self.commander.q
        initial_delta = trajectory.q[0] - c_q
        duration = np.abs((initial_delta / 0.8)).max()

        initial_goal = JointTrajectoryMsg(
            joint_names=self.commander.joints,
            points=[
                JointTrajectoryPointMsg(positions=c_q, velocities=np.zeros_like(c_q)),
                JointTrajectoryPointMsg(
                    positions=trajectory.q[0],
                    velocities=np.zeros_like(c_q),
                    time_from_start=rospy.Duration(duration),
                ),
            ],
        )
        initial_goal.header.stamp = rospy.Time.now()

        self.pub.publish(initial_goal)
        rospy.sleep(duration)

        full_traj = JointTrajectoryMsg(
            joint_names=self.commander.joints,
            points=[
                JointTrajectoryPointMsg(positions=q, time_from_start=rospy.Duration(t))
                for t, q, qd in zip(trajectory.t, trajectory.q, trajectory.qd)
            ],
        )

        while not rospy.is_shutdown() and (
            self.commander.q is None
            or np.abs(trajectory.q[0] - self.commander.q).max() > accuracy
        ):
            rospy.sleep(0.02)

        full_traj.header.stamp = rospy.Time.now()
        self.pub.publish(full_traj)

        while (
            not rospy.is_shutdown()
            and np.abs(trajectory.q[-1] - self.commander.q).max() > accuracy
        ):
            rospy.sleep(0.02)

    def cancel_all_goals(self):
        self.pub.publish(JointTrajectoryMsg())
        self.commander.set_goal(self.commander.q)


class ThreadedJointTrajectoryFollower(JointTrajectoryFollower):
    def __init__(self, gripper_cb: Callable) -> None:
        super().__init__(gripper_cb)

        self._reset_thread()

    def _reset_thread(self):
        self._thread = None

    def follow_time_based(self, trajectory: RobotTrajectory, accuracy: float = 0.01):
        if self._thread is not None:
            self._thread.join()

        self._thread = threading.Thread(
            target=super().follow_time_based, args=(trajectory, accuracy)
        )
        self._thread.start()

    def cancel_all_goals(self):
        super().cancel_all_goals()
        self._reset_thread()

    @property
    def is_running(self):
        """
        Returns whether the thread is still running.
        If not, the trajectory is finished.
        """
        return self._thread is not None and self._thread.is_alive()
