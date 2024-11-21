from dataclasses import dataclass

import torch
from omegaconf import DictConfig
from torch import nn

import tapas_gmm.encoder.models.transporter.transporter as transporter
import tapas_gmm.encoder.representation_learner as representation_learner
from tapas_gmm.utils.observation import SampleTypes, SingleCamObservation
from tapas_gmm.utils.select_gpu import device


@dataclass
class PretrainingConfig:
    lr: float = 5e-4


@dataclass
class TransporterConfig(representation_learner.RepresentationLearnerConfig):
    encoder: transporter.TransporterModelConfig = transporter.TransporterModelConfig()

    pretraining: PretrainingConfig = PretrainingConfig()


class Transporter(representation_learner.RepresentationLearner):
    sample_type = SampleTypes.CAM_PAIR

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config=config)

        self.config = config.encoder_config

        encoder_config = config.encoder_config.encoder
        feature_encoder = transporter.FeatureEncoder(
            encoder_config.image_channels, encoder_config.architecture["image_encoder"]
        )
        pose_regressor = transporter.PoseRegressor(
            encoder_config.image_channels,
            encoder_config.n_keypoints,
            encoder_config.architecture["image_encoder"],
        )
        refine_net = transporter.RefineNet(
            encoder_config.image_channels, encoder_config.architecture["image_encoder"]
        )

        self.model = transporter.Transporter(
            feature_encoder, pose_regressor, refine_net, std=encoder_config.keypoint_std
        )

        # print(feature_encoder)
        # print(pose_regressor)
        # print(refine_net)
        # exit()

        pretrain_conf = config.encoder_config.pretraining
        self.optimizer = torch.optim.Adam(self.model.parameters(), pretrain_conf.lr)
        self.loss = torch.nn.functional.mse_loss

    def process_batch(self, xs, xt):
        xs = nn.functional.interpolate(
            xs, size=(128, 128), mode="bilinear", align_corners=True
        )
        xt = nn.functional.interpolate(
            xt, size=(128, 128), mode="bilinear", align_corners=True
        )

        reconstruction = self.model(xs, xt)
        loss = self.loss(reconstruction, xt)

        metrics = {"loss": loss}

        return metrics

    def update_params(self, batch, **kwargs):
        source_batch, target_batch = batch

        xs = source_batch.to(device)
        xt = target_batch.to(device)

        self.optimizer.zero_grad()
        training_metrics = self.process_batch(xs, xt)

        training_metrics["loss"].backward()
        self.optimizer.step()

        training_metrics = {
            "train-{}".format(k): v for k, v in training_metrics.items()
        }

        return training_metrics

    def forward(self, batch, **kwargs):
        return self.encode(batch)

    def encode_single_camera(
        self, batch: SingleCamObservation
    ) -> tuple[torch.Tensor, dict]:
        camera_obs = batch.cam_rgb

        rgb_resolution = camera_obs.shape[-2:]

        subsample_resolution = self._get_subsample_resolution(rgb_resolution)

        camera_obs = nn.functional.interpolate(
            camera_obs, size=subsample_resolution, mode="bilinear", align_corners=True
        )

        cam_emb, info = self.model.encode(camera_obs)

        return cam_emb, info

    def _get_subsample_resolution(self, cam_res):
        return (cam_res[0] // 2, cam_res[1] // 2)

    def evaluate(self, batch, **kwargs):
        source_batch, target_batch = batch

        xs = source_batch.to(device)
        xt = target_batch.to(device)

        eval_metrics = self.process_batch(xs, xt)

        eval_metrics = {"eval-{}".format(k): v for k, v in eval_metrics.items()}

        return eval_metrics

    def reconstruct(self, batch):
        batch = batch.to(device)
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        features = self.model.feature_encoder(batch)
        reconstruction = self.model.refine_net(features)
        return reconstruction

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        return config.encoder.n_keypoints * 2 * n_cams
