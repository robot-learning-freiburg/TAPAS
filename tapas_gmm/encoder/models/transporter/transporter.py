from dataclasses import dataclass, field

import torch
from torch import nn

from tapas_gmm.encoder.models.transporter.utils import spatial_softmax


# TODO: break down further into image_encoder, keypoint_encoder, config
@dataclass
class TransporterModelConfig:
    image_channels: int = 3
    k: int = 4
    n_keypoints: int = 10
    keypoint_std: float = 0.1
    architecture: dict = field(
        default_factory=lambda: {
            "image_encoder": {
                "no_filters": (32, 32, 64, 128),
                "stride": (1, 1, 2, 1),
                "filter_size": (7, 3, 3, 3),
                "padding": (3, 1, 1, 1),
            },
            "keypoint_encoder": {
                "no_filters": (32, 32, 64, 128),
                "stride": (1, 1, 2, 1),
                "filter_size": (7, 3, 3, 3),
                "padding": (3, 1, 1, 1),
            },
        }
    )


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1
    ):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)


class FeatureEncoder(nn.Module):
    """Phi"""

    def __init__(self, in_channels=3, config=None):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            *[
                Block(
                    in_channels if i == 0 else config["no_filters"][i - 1],
                    f,
                    kernel_size=(k, k),
                    stride=s,
                    padding=p,
                )
                for i, (f, s, k, p) in enumerate(
                    zip(
                        config["no_filters"],
                        config["stride"],
                        config["filter_size"],
                        config["padding"],
                    )
                )
            ]
            # Block(in_channels, 32, kernel_size=(
            #     7, 7), stride=1, padding=3),  # 1
            # Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            # Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            # Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            # Block(64, 128, kernel_size=(3, 3), stride=2),  # 5
            # Block(128, 128, kernel_size=(3, 3), stride=1),  # 6
        )

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x)


class PoseRegressor(nn.Module):
    """Pose regressor"""

    # https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf

    def __init__(self, in_channels=3, n_keypoints=1, config=None):
        super(PoseRegressor, self).__init__()
        self.net = nn.Sequential(
            *[
                Block(
                    in_channels if i == 0 else config["no_filters"][i - 1],
                    f,
                    kernel_size=(k, k),
                    stride=s,
                    padding=p,
                )
                for i, (f, s, k, p) in enumerate(
                    zip(
                        config["no_filters"],
                        config["stride"],
                        config["filter_size"],
                        config["padding"],
                    )
                )
            ]
            # Block(in_channels, 32, kernel_size=(
            #     7, 7), stride=1, padding=3),  # 1
            # Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            # Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            # Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            # Block(64, 128, kernel_size=(3, 3), stride=2),  # 5
            # Block(128, 128, kernel_size=(3, 3), stride=1),  # 6
        )
        self.regressor = nn.Conv2d(128, n_keypoints, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, k, H', W') tensor.
        """
        x = self.net(x)
        return self.regressor(x)


class RefineNet(nn.Module):
    """Network that generates images from feature maps and heatmaps."""

    def __init__(self, num_channels, config=None):
        super(RefineNet, self).__init__()
        reversed_phi = [
            Block(
                f,
                num_channels if i == 0 else config["no_filters"][i - 1],
                kernel_size=(k, k),
                stride=1,
                padding=p,
            )
            for i, (f, k, p) in reversed(
                list(
                    enumerate(
                        zip(
                            config["no_filters"],
                            config["filter_size"],
                            config["padding"],
                        )
                    )
                )
            )
        ]
        striding_layers = [i for i, x in enumerate(config["stride"]) if x > 1]

        added_layers = 0
        for i in striding_layers:  # add in upsampling to reverse stride
            upsampling_layer = nn.UpsamplingBilinear2d(scale_factor=config["stride"][i])
            reversed_phi.insert(i + added_layers, upsampling_layer)
            added_layers += 1

        self.net = nn.Sequential(
            *reversed_phi
            # add upsampling to undo stride 2
            # NOTE: Paper was a bit unclear on net archi, using inverse of Phi
            # Block(128, 128, kernel_size=(3, 3), stride=1),  # 6
            # Block(128, 64, kernel_size=(3, 3), stride=1),  # 5
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            # Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            # Block(32, num_channels, kernel_size=(
            # 7, 7), stride=1, padding=3),  # 1
        )

    def forward(self, x):
        """
        x: the transported feature map.
        """
        return self.net(x)


def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(
        torch.linspace(
            -1, 1, S_row.size(-1), dtype=features.dtype, device=features.device
        )
    ).sum(-1)
    # N, K
    u_col = S_col.mul(
        torch.linspace(
            -1, 1, S_col.size(-1), dtype=features.dtype, device=features.device
        )
    ).sum(-1)
    return torch.stack((u_row, u_col), -1)  # N, K, 2


def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)
    # g_yx = g_yx.permute([0, 2, 3, 1])
    return g_yx


def transport(source_keypoints, target_keypoints, source_features, target_features):
    """
    Args
    ====
    source_keypoints (N, K, H, W)
    target_keypoints (N, K, H, W)
    source_features (N, D, H, W)
    target_features (N, D, H, W)

    Returns
    =======
    """
    out = source_features
    for s, t in zip(
        torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)
    ):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(
            1
        ) * target_features
    return out


class Transporter(nn.Module):
    def __init__(self, feature_encoder, point_net, refine_net, std=0.1):
        super(Transporter, self).__init__()
        self.feature_encoder = feature_encoder
        self.point_net = point_net
        self.refine_net = refine_net
        self.std = std

    def encode(self, batch):
        heatmap = spatial_softmax(self.point_net(batch))

        keypoints = compute_keypoint_location_mean(heatmap)

        # shape into same format as kp-encoder: x-components, y-components
        keypoints = torch.cat((keypoints[..., 1], keypoints[..., 0]), -1)

        info = {"heatmap": heatmap}

        return keypoints, info

    def forward(self, source_images, target_images):
        # print(source_images.size())
        source_features = self.feature_encoder(source_images)
        target_features = self.feature_encoder(target_images)

        # print(source_features.size())
        # print(self.point_net(source_images).size())

        source_keypoints = gaussian_map(
            spatial_softmax(self.point_net(source_images)), std=self.std
        )

        target_keypoints = gaussian_map(
            spatial_softmax(self.point_net(target_images)), std=self.std
        )

        transported_features = transport(
            source_keypoints.detach(),
            target_keypoints,
            source_features.detach(),
            target_features,
        )

        assert transported_features.shape == target_features.shape

        # print(transported_features.size())
        reconstruction = self.refine_net(transported_features)
        # print(reconstruction.size())
        return reconstruction
