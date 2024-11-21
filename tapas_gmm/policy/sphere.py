import numpy as np

from tapas_gmm.utils.observation import SceneObservation


class SpherePolicy:
    def __init__(self, config, **kwargs):
        self.pose_counter = 0
        self.scan_poses = np.array(
            [
                [
                    2.78524697e-01,
                    -8.16994347e-03,
                    1.47182894e00,
                    5.04820946e-06,
                    9.92677510e-01,
                    -8.55109920e-06,
                    1.20795168e-01,
                    1.0,
                ],
                [
                    -0.13868015,
                    -0.018499,
                    1.30796123,
                    -0.00450313,
                    0.93944383,
                    0.01647907,
                    0.34227696,
                    1.0,
                ],
                [
                    0.23379445,
                    -0.25981659,
                    1.46826792,
                    0.02166777,
                    0.98085356,
                    0.08603449,
                    0.17336345,
                    1.0,
                ],
                [
                    0.09003212,
                    -0.51703942,
                    1.34673989,
                    0.02836432,
                    0.9223175,
                    0.22182107,
                    0.31515279,
                    1.0,
                ],
                [
                    0.23925975,
                    0.34963048,
                    1.40325511,
                    0.12973945,
                    0.92903697,
                    -0.22003566,
                    0.2676608,
                    1.0,
                ],
            ]
        )

        self.path_planner_func = kwargs["path_planner_func"]
        self.current_path = None

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        # scan is finished, return None st the training loop knows to exit
        if (
            self.pose_counter == self.scan_poses.shape[0] - 1
            and len(self.current_path) == 0
        ):
            pos = None
            self.pose_counter = 0
        else:
            # no plan is made or the current plan is done, plan to next pose
            if self.current_path is None or len(self.current_path) == 0:
                next_pos = self.scan_poses[self.pose_counter + 1]

                self.current_path = self.path_planner_func(
                    next_pos[:3], quaternion=next_pos[3:7]
                )

                self.pose_counter += 1

            len_conf = len(self.current_path._arm.joints)

            pos, self.current_path._path_points = (
                self.current_path._path_points[:len_conf],
                self.current_path._path_points[len_conf:],
            )

            pos = np.append(pos, [1.0])

        return pos, {}
