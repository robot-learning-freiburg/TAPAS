from dataclasses import dataclass

import torch
from torch import nn

import tapas_gmm.encoder.models.monet.monet as monet
import tapas_gmm.encoder.representation_learner as representation_learner
from tapas_gmm.utils.observation import SampleTypes, SingleCamObservation
from tapas_gmm.utils.select_gpu import device


def initialize_weights(m, init_type="normal", init_gain=0.02):
    # if isinstance(m, nn.Conv2d):
    #     torch.nn.init.normal_(m.weight.data, std=.1)
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight.data, 1)
    #     nn.init.constant_(m.bias.data, 0)
    # elif isinstance(m, nn.Linear):
    #     torch.nn.init.normal_(m.weight.data)
    #     nn.init.constant_(m.bias.data, 0)
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        if init_type == "normal":
            torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == "xavier":
            torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == "kaiming":
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    # BatchNorm's weight is not a matrix; only normal distribution applies
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
        torch.nn.init.constant_(m.bias.data, 0.0)


@dataclass
class PretrainingConfig:
    lr: float = 1e-4
    beta: float = 5  # weight for the encoder KLD
    gamma: float = 0.5  # weight of mask KLD


@dataclass
class MonetConfig(representation_learner.RepresentationLearnerConfig):
    encoder: monet.MONetModelConfig = monet.MONetModelConfig()

    pretraining: PretrainingConfig = PretrainingConfig()


class Monet(representation_learner.RepresentationLearner):
    sample_type = SampleTypes.CAM_SINGLE

    def __init__(self, config=None):
        super().__init__(config=config)

        self.config = config.encoder_config

        self.model = monet.MONetModel(config.encoder_config.encoder)
        # self.model.apply(initialize_weights)

        # for submodule in self.model.modules():
        #     submodule.register_forward_hook(nan_hook)

        self.criterionKL = nn.KLDivLoss(reduction="batchmean")

        pretrain_conf = config.encoder_config.pretraining
        self.beta = pretrain_conf.beta
        self.gamma = pretrain_conf.gamma

        # NOTE: original paper used RMSProp, lr=0.0001, batch_size=64
        self.optimizer = torch.optim.Adam(self.model.parameters(), pretrain_conf.lr)

    def process_batch(self, batch, batch_size):
        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        x_masked, loss_E, b, m, m_tilde_logits, x_mu, x_m = self.model(batch)
        metrics = self.loss(loss_E, b, m, m_tilde_logits, batch_size)

        return metrics

    def update_params(self, batch, dataset_size=None, batch_size=None, **kwargs):
        batch = batch.to(device)

        self.optimizer.zero_grad()
        training_metrics = self.process_batch(batch, batch_size)
        training_metrics["loss"].backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
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

        z_mu, z_logvar = self.model.encode(camera_obs)

        info = {}

        return z_mu, info

    def _get_subsample_resolution(self, cam_res):
        return (cam_res[0] // 2, cam_res[1] // 2)

    def evaluate(self, batch, dataset_size=None, batch_size=None, **kwargs):
        batch = batch.to(device)

        eval_metrics = self.process_batch(batch, batch_size)

        eval_metrics = {"eval-{}".format(k): v for k, v in eval_metrics.items()}

        return eval_metrics

    def reconstruct(self, batch):
        batch = batch.to(device)

        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        x_masked, loss_E, b, m, m_tilde_logits, x_mu, x_m = self.model(batch)

        return x_masked

    def reconstruct_w_extras(self, batch):
        batch = batch.to(device)

        batch = nn.functional.interpolate(
            batch, size=(128, 128), mode="bilinear", align_corners=True
        )
        x_masked, loss_E, b, m, m_tilde_logits, x_mu, x_m = self.model(batch)

        return x_masked, loss_E, b, m, m_tilde_logits, x_mu, x_m

    @classmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        return n_cams * sum(
            config.encoder.latent_dims[v] for _, v in config.encoder.slots.items()
        )

    def loss(self, loss_E, b, m, m_tilde_logits, batch_size):
        loss_E /= batch_size
        loss_D = -torch.logsumexp(b, dim=1).sum() / batch_size
        loss_mask = self.criterionKL(m_tilde_logits.log_softmax(dim=1), m)
        loss = loss_D + self.beta * loss_E + self.gamma * loss_mask

        return {
            "loss": loss,
            "loss_E": loss_E,
            "loss_D": loss_D,
            "loss_mask": loss_mask,
            "loss_E_weighted": self.beta * loss_E,
            "loss_mask_weighted": self.gamma * loss_mask,
        }
