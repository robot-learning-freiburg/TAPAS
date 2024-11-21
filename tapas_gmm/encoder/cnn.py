from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from tapas_gmm.encoder.representation_learner import (
    RepresentationLearner,
    RepresentationLearnerConfig,
)
from tapas_gmm.utils.observation import SingleCamObservation


@dataclass
class CNNConfig(RepresentationLearnerConfig):
    pass


class CNN(RepresentationLearner):
    def __init__(self, config: DictConfig) -> None:
        self.config = config

        super().__init__(config)

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=2, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
        )

    def encode_single_camera(
        self, batch: SingleCamObservation
    ) -> tuple[torch.Tensor, dict]:
        camera_obs = batch.rgb

        rgb_resolution = camera_obs.shape[-2:]

        subsample_resolution = self._get_subsample_resolution(rgb_resolution)

        camera_obs = nn.functional.interpolate(
            camera_obs, size=subsample_resolution, mode="bilinear", align_corners=True
        )

        cam_emb = self.model(camera_obs)

        info = {}

        return cam_emb, info

    def _get_subsample_resolution(self, cam_res):
        return (cam_res[0] // 2, cam_res[1] // 2)

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=(None, None)):
        dim_mapping = {
            (256, 256): 256,
            (360, 480): 690,
        }

        return dim_mapping[image_dim] * n_cams

    def from_disk(self, chekpoint_path):
        logger.info("CNN encoder does not need snapshot loading. Skipping.")

    def update_params(self, batch, **kwargs):
        raise NotImplementedError  # not needed, as train this end-to-end


class CNNDepth(CNN):
    def __init__(self, config=None):
        RepresentationLearner.__init__(self, config)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
            ),
            nn.ELU(),
        )

    def forward(self, batch):
        rgb = batch.cam_rgb
        depth = batch.cam_d.unsqueeze(1)
        batch.cam_rgb = torch.cat((rgb, depth), dim=-3)

        if (cam_obs2 := batch.cam_rgb2) is not None:
            depth2 = batch.cam_d2.unsqueeze(1)
            batch.cam_rgb2 = torch.cat((cam_obs2, depth2), dim=-3)

        return self.encode(batch), {}
