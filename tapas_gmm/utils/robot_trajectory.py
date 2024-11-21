import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

# NOTE: would be super nice to have something like TensorDict/TensorClass for numpy arrays
# as well to clean up the TrajectoryPoint and RobotTrajectory classes


@dataclass
class TrajectoryPoint:
    t: float | None = None
    q: np.ndarray | None = None
    qd: np.ndarray | None = None
    qdd: np.ndarray | None = None
    gripper: np.ndarray | None = None
    ee: np.ndarray | None = None


@dataclass
class RobotTrajectory:
    points: list[TrajectoryPoint]
    duration: float
    _iter_idx: int = 0

    @property
    def has_qposes(self) -> bool:
        return all(p.q is not None for p in self.points)

    @property
    def has_ee_poses(self) -> bool:
        return all(p.ee is not None for p in self.points)

    @property
    def t(self) -> np.ndarray:
        return np.array([p.t for p in self.points])

    @t.setter
    def t(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.t = value[i]

    @property
    def q(self) -> np.ndarray:
        return np.array([p.q for p in self.points])

    @q.setter
    def q(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.q = value[i]

    @property
    def qd(self) -> np.ndarray:
        return np.array([p.qd for p in self.points])

    @qd.setter
    def qd(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.qd = value[i]

    @property
    def qdd(self) -> np.ndarray:
        return np.array([p.qdd for p in self.points])

    @qdd.setter
    def qdd(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.qdd = value[i]

    @property
    def gripper(self) -> np.ndarray:
        return np.array([p.gripper for p in self.points])

    @gripper.setter
    def gripper(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.gripper = value[i]

    @property
    def ee(self) -> np.ndarray:
        return np.array([p.ee for p in self.points])

    @ee.setter
    def ee(self, value: np.ndarray):
        assert len(value) == len(self.points)
        for i, p in enumerate(self.points):
            p.ee = value[i]

    @classmethod
    def from_np(cls, duration=None, dt=0.05, **kwargs):
        for key in kwargs.keys():
            if key not in TrajectoryPoint.__annotations__:
                raise ValueError(f"Invalid key {key} for TrajectoryPoint.")
            if kwargs[key] is not None and not isinstance(kwargs[key], np.ndarray):
                raise ValueError(
                    f"Value for key {key} must be np.ndarray, but is {type(kwargs[key])}."
                )

        n_traj_points = len(kwargs[next(iter(kwargs.keys()))])

        points = [
            TrajectoryPoint(**{k: v[i] for k, v in kwargs.items()})
            for i in range(n_traj_points)
        ]

        duration = n_traj_points * dt if duration is None else duration

        return RobotTrajectory(points=points, duration=duration)

    # def _concatenate_single(self, other: Self) -> "RobotTrajectory":
    #     t_offset = self.points[-1].t

    #     other_points = other.points.copy()

    #     for p in other_points:
    #         p.t += t_offset

    #     return RobotTrajectory(
    #         self.points + other_points,
    #         self.duration + other.duration,
    #     )

    @classmethod
    def concatenate(cls, trajs: Iterable[Self]) -> "RobotTrajectory":
        offsets = np.cumsum([0] + [t.points[-1].t + 0.05 for t in trajs])[:-1]

        points = [t.points.copy() for t in trajs]

        for t, (traj_points, off) in enumerate(zip(points, offsets)):
            for p in range(len(traj_points)):
                points[t][p].t += off

        return RobotTrajectory(
            points=list(itertools.chain(*points)),
            duration=sum(t.duration for t in trajs),
        )

    def split(self, indeces: Iterable[int]) -> list["RobotTrajectory"]:
        indeces = [0] + list(indeces) + [len(self.points)]
        segments = [
            RobotTrajectory(
                points=self.points[i:j],
                duration=self.points[j - 1].t - self.points[i].t,
            )
            for i, j in zip(indeces[:-1], indeces[1:])
        ]

        for j in range(1, len(segments)):
            for p in range(len(segments[j].points)):
                segments[j].points[p].t -= segments[j].points[0].t

        return segments

    # def concatenate(self, other: Self | Iterable[Self]) -> "RobotTrajectory":
    #     if isinstance(other, RobotTrajectory):
    #         return self._concatenate_single(other)
    #     elif isinstance(other, Iterable):
    #         return self._concatenate_iterable(other)
    #     else:
    #         raise ValueError(f"Cannot concatenate {type(other)} with Trajectory.")

    def step(self) -> TrajectoryPoint:
        if self._iter_idx >= len(self.points):
            raise StopIteration

        point = self.points[self._iter_idx]
        self._iter_idx += 1

        return point

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> TrajectoryPoint:
        return self.points[idx]

    def __iter__(self):
        return iter(self.points)

    @property
    def remaining_len(self) -> int:
        return len(self.points) - self._iter_idx

    @property
    def is_finished(self) -> bool:
        return self.remaining_len <= 0

    def invert(self) -> "RobotTrajectory":
        return RobotTrajectory(points=self.points[::-1], duration=self.duration)
