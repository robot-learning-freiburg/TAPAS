import abc
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributions.normal import Normal

from tapas_gmm.dataset.bc import BCDataset
from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.utils.logging import log_constructor
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.select_gpu import device  # , normalize_quaternion


@dataclass
class PolicyConfig:
    suffix: str | None


class Policy(nn.Module):
    @abc.abstractmethod  # Policy object misses encoder.
    @log_constructor
    def __init__(
        self,
        config: PolicyConfig,
        skip_module_init: bool = False,
        encoder_checkpoint: str | None = None,
        **kwargs
    ):
        if not skip_module_init:
            super().__init__()
        self.config = config
        if encoder_checkpoint:
            self._load_encoder_checkpoint(encoder_checkpoint)

    def from_disk(self, chekpoint_path: str) -> None:
        self.load_state_dict(torch.load(chekpoint_path, map_location=device))

    def to_disk(self, checkpoint_path: str) -> None:
        logger.info("Saving policy at {}", checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    def _load_encoder_checkpoint(self, ckpt: str) -> None:
        logger.info("Adding encoder checkpoint to snapshot.")
        if self.config.encoder_name == "keypoints_gt":
            self.encoder.from_disk(ckpt, force_read=True)
        else:
            self.encoder.from_disk(ckpt)

        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def initialize_parameters_via_dataset(
        self, replay_memory: BCDataset, cameras: tuple[str], **kwargs
    ) -> None:
        logger.info("This policy does not use dataset initialization.")

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        pass  # implement in subclass if needed

    def update_params(self, batch: SceneObservation) -> dict:  # type: ignore
        raise NotImplementedError

    def evaluate(self, batch: SceneObservation) -> dict:  # type: ignore
        raise NotImplementedError

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    # def get_viz_encoder_callback(self) -> Callable:
    #     """
    #     Get the visualization encoder callback. For live visualization in real world environments before the episode starts.
    #     """
    #     return self.obs_encoder.get_viz_encoder_callback
