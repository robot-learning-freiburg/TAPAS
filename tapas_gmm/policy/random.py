import random
from dataclasses import dataclass

import numpy as np

from tapas_gmm.policy.policy import PolicyConfig
from tapas_gmm.utils.observation import SceneObservation


@dataclass
class RandomPolicyConfig(PolicyConfig):
    action_dim: int


class RandomPolicy:
    def __init__(self, config: RandomPolicyConfig, **kwargs):
        self.action_dim = config.action_dim

    def from_disk(self, file_name):
        pass  # nothing to load

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        action = np.asarray([random.random() for _ in range(self.action_dim)])
        return action, {}
