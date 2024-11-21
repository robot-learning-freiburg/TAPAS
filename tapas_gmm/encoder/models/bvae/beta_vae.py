from dataclasses import dataclass

import torch
from torch import nn

from tapas_gmm.encoder.models.bvae.base import BaseVAE

from .types_ import List, Tensor


@dataclass
class Config:
    image_channels: int = 3
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = tuple((32, 32, 32, 32, 128))
    stride: tuple[int, ...] = tuple((2, 2, 2, 2))
    filter_size: tuple[int, ...] = tuple((4, 4, 4, 4))
    padding: tuple[int, ...] = tuple((1, 1, 1, 1))


class BetaVAE(BaseVAE):
    def __init__(self, config, **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.config = config

        self.latent_dim = config["latent_dim"]

        modules = []

        # Build Encoder
        for i, o, k, s, p in zip(
            [config["image_channels"]] + config["hidden_dims"],
            config["hidden_dims"],
            config["filter_size"],
            config["stride"],
            config["padding"],
        ):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                    ),
                    nn.BatchNorm2d(num_features=o),
                    nn.ReLU(),
                )
            )

        self.cnn_encoder = nn.Sequential(*modules)
        # TODO: adapt for 256, also padding for decoder below.
        self.fc_encoder = nn.Sequential(  # 8*8 for image resolution 128
            nn.Linear(config["hidden_dims"][-2] * 8 * 8, config["hidden_dims"][-1]),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(config["hidden_dims"][-1], config["latent_dim"])
        self.fc_var = nn.Linear(config["hidden_dims"][-1], config["latent_dim"])

        # Build Decoder
        modules = []

        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"], config["hidden_dims"][-1]),
            nn.ReLU(),
            nn.Linear(config["hidden_dims"][-1], config["hidden_dims"][-2] * 8 * 8),
        )

        for i, o, k, s, p in reversed(
            list(
                zip(
                    config["hidden_dims"][1:],
                    config["hidden_dims"][:-1],
                    config["filter_size"][1:],
                    config["stride"][1:],
                    config["padding"][1:],
                )
            )
        ):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=k,
                        stride=s,
                        padding=p,
                        # NOTE 0 happens to works for 128 input
                        output_padding=0,
                    ),
                    nn.BatchNorm2d(num_features=o),
                    nn.ReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                config["hidden_dims"][0],
                config["hidden_dims"][0],
                kernel_size=config["filter_size"][0],
                stride=config["stride"][0],
                padding=config["padding"][0],
                output_padding=1,
            ),
            nn.BatchNorm2d(config["hidden_dims"][0]),
            nn.ReLU(),
            nn.Conv2d(
                config["hidden_dims"][0],
                out_channels=config["image_channels"],
                kernel_size=config["filter_size"][0],
                padding=config["padding"][0],
            ),
            nn.Sigmoid(),
        )  # or ReLU?

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.cnn_encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc_encoder(result)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.config["hidden_dims"][-2], 8, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
