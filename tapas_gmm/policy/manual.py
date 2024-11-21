import numpy as np

from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.utils.human_feedback import correct_action
from tapas_gmm.utils.observation import SceneObservation


class ManualPolicy:
    def __init__(self, config, env, keyboard_obs, **kwargs):
        self.keyboard_obs = keyboard_obs

        self.gripper_open = 0.9

    def from_disk(self, file_name):
        pass  # nothing to load

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.gripper_open])
        if self.keyboard_obs.has_joints_cor() or self.keyboard_obs.has_gripper_update():
            action = correct_action(self.keyboard_obs, action)
            self.gripper_open = action[-1]
        return action, {}

    def reset_episode(self, env: BaseEnvironment | None = None):
        # TODO: add this to all other policies as well and use it to store
        # the LSTM state as well?
        self.gripper_open = 0.9
