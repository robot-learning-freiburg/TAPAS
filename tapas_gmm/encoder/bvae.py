from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

import tapas_gmm.encoder.models.bvae.beta_vae as bvae
from tapas_gmm.encoder.representation_learner import (
    RepresentationLearner,
    RepresentationLearnerConfig,
)
from tapas_gmm.utils.observation import SampleTypes, SingleCamObservation
from tapas_gmm.utils.select_gpu import device


@dataclass
class PretrainingConfig:
    lr: float = 1e-4
    beta: float = 1.5
    kld_correction: bool = True
    loss_type: str = "H"
    gamma: float = 10.0
    max_capacity: int | float = 25
    capacity_max_iter: float = 1e5


@dataclass
class BVAEConfig(RepresentationLearnerConfig):
    encoder: bvae.Config = bvae.Config()

    pretraining: PretrainingConfig = PretrainingConfig()


class BVAE(RepresentationLearner):
    sample_type = SampleTypes.CAM_SINGLE

    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.config = config.encoder_config

        self.model = bvae.BetaVAE(config.encoder_config.encoder)

        pretrain_conf = config.encoder_config.pretraining
        self.beta = pretrain_conf.beta
        self.kld_correction = pretrain_conf.kld_correction
        self.gamma = pretrain_conf.gamma
        self.loss_type = pretrain_conf.loss_type
        self.C_max = torch.Tensor([pretrain_conf.max_capacity])
        self.C_stop_iter = pretrain_conf.capacity_max_iter

        self.optimizer = torch.optim.Adam(self.model.parameters(), pretrain_conf.lr)

    def loss(self, recons, input, mu, log_var, kld_weight) -> dict:
        self.num_iter += 1

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
            "kLD_weighted": self.beta * kld_weight * kld_loss,
        }

    def calc_kld_weight(self, batch_size, dataset_size):
        return batch_size / dataset_size if self.kld_correction else 1

    def process_batch(self, batch, dataset_size, batch_size):
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        reconstruction, input, mu, log_var = self.model.forward(batch)
        kld_weight = self.calc_kld_weight(batch_size, dataset_size)
        metrics = self.loss(reconstruction, input, mu, log_var, kld_weight=kld_weight)

        return metrics

    def update_params(self, batch, dataset_size=None, batch_size=None, **kwargs):
        batch = batch.to(device)

        self.optimizer.zero_grad()
        training_metrics = self.process_batch(batch, dataset_size, batch_size)

        training_metrics["loss"].backward()
        self.optimizer.step()

        training_metrics = {
            "train-{}".format(k): v for k, v in training_metrics.items()
        }

        return training_metrics

    def encode_single_camera(
        self, batch: SingleCamObservation
    ) -> tuple[torch.Tensor, dict]:
        camera_obs = batch.rgb

        rgb_resolution = camera_obs.shape[-2:]

        subsample_resolution = self._get_subsample_resolution(rgb_resolution)

        camera_obs = nn.functional.interpolate(
            camera_obs, size=subsample_resolution, mode="bilinear", align_corners=True
        )

        mu, log_var = self.model.encode(camera_obs)

        info = {}

        return mu, info  # see paper: treating mu as embedding

    def _get_subsample_resolution(self, cam_res):
        return tuple((cam_res[0] // 2, cam_res[1] // 2))

    def evaluate(self, batch, dataset_size=None, batch_size=None, **kwargs):
        batch = batch.to(device)

        eval_metrics = self.process_batch(batch, dataset_size, batch_size)

        eval_metrics = {"eval-{}".format(k): v for k, v in eval_metrics.items()}

        return eval_metrics

    def reconstruct(self, batch):
        batch = batch.to(device)
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        return self.model(batch)[0]

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        return config.encoder.latent_dim * n_cams
