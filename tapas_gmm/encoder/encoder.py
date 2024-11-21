from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from tapas_gmm.dataset.bc import BCDataset
from tapas_gmm.encoder import get_image_encoder_class
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.select_gpu import device

# TODO: rename encoders to ImageEncoders
# then make ImageEncoders part of the BatchEncoder?


empty_tensor = torch.Tensor().to(device)


@dataclass
class ObservationEncoderConfig:
    ee_pose: bool = False
    proprio_obs: bool = False
    object_poses: bool = False

    image_encoder: Any = None
    pre_encoding: str | None = None
    online_encoding: str | None = None

    constant_image_encoding_per_episode: bool = False
    # cameras: tuple[str, ...] = ()  # Currently iterating over all cameras and concatenating


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        config: ObservationEncoderConfig,
        image_encoder_checkpoint: str | None = None,
    ) -> None:
        assert not (
            config.constant_image_encoding_per_episode
            and (config.pre_encoding is not None)
        ), "Constant image encoding is a hack for getting fixed keypoints during online encoding. For training, use a constant pre_encoding instead."

        super().__init__()
        self.config = config

        ImageEncoder = get_image_encoder_class(config.image_encoder)

        if ImageEncoder is None:
            self.image_encoder = None
        else:
            self.image_encoder = ImageEncoder(config.image_encoder)

        print(self.image_encoder)

        if image_encoder_checkpoint and self.image_encoder:
            self.image_encoder.from_disk(image_encoder_checkpoint)

        self._last_image_encoding = None
        self._last_image_enc_info = None

    def encode(self, obs: SceneObservation) -> tuple[torch.Tensor, dict]:
        """
        Encoder an observation Tensorclass into a single Tensor.

        Parameters
        ----------
        batch : SceneObservation
            TensorClass holding the observation batch.

        Returns
        -------
        torch.Tensor, dict
            Concatenated observation data and additional visual encoding information.
        """
        ee_pose = obs.ee_pose if self.config.ee_pose else empty_tensor
        proprio_obs = obs.proprio_obs if self.config.proprio_obs else empty_tensor
        object_poses = (
            torch.cat([p for _, p in obs.object_poses.items()], dim=-1)
            if self.config.object_poses
            else empty_tensor
        )

        if (
            self.config.constant_image_encoding_per_episode
            and (self.image_encoder is not None)
            and (self._last_image_encoding is not None)
        ):
            stack_size = max(ee_pose.shape, proprio_obs.shape, object_poses.shape)[:-1]
            image_encoding = self._last_image_encoding.repeat(*stack_size, 1)
            image_enc_info = self._last_image_enc_info
        elif self.config.online_encoding and self.image_encoder is not None:
            image_encoding, image_enc_info = self.image_encoder.encode(obs)

            self._last_image_encoding = image_encoding
            self._last_image_enc_info = image_enc_info
        elif self.image_encoder is None:
            image_encoding = empty_tensor
            image_enc_info = {}
        else:
            image_encoding = (
                getattr(obs, self.config.pre_encoding).squeeze(2)
                if self.config.pre_encoding
                else empty_tensor
            )
            image_enc_info = {}

        low_dim_input = torch.cat(
            (ee_pose, proprio_obs, object_poses, image_encoding), dim=-1
        )

        return low_dim_input, image_enc_info

    def get_image_encoding(
        self, obs: SceneObservation
    ) -> tuple[torch.Tensor | None, dict]:
        """
        Get the image encoding of the observation if self has an image encoder else None.
        """
        if self.image_encoder is None:
            return None, {}
        else:
            return self.image_encoder.encode(obs)

    def get_obs_distribution(self, replay_memory: BCDataset) -> torch.Tensor:
        """
        Get the distribution of the data.

        Parameters
        ----------
        data : torch.Tensor
            The data to get the distribution of.

        Returns
        -------
        torch.Tensor
            All data points concatenated.
        """
        ee_pose = (
            replay_memory.scene_data.get_ee_pose_distribution()
            if self.config.ee_pose
            else torch.Tensor()
        )
        proprio_obs = (
            replay_memory.scene_data.get_proprio_distribution()
            if self.config.proprio_obs
            else torch.Tensor()
        )
        object_poses = (
            replay_memory.scene_data.get_object_poses_distribution(stacked=True)
            if self.config.object_poses
            else torch.Tensor()
        )
        if self.config.online_encoding:
            raise ValueError(
                "Can't use online encoding with fitted normalization. "
                "Use pre_encoding instead."
            )

        image_encoding = (
            replay_memory.get_pre_encoding_distribution()
            if self.config.pre_encoding
            else torch.Tensor()
        )

        joint = torch.cat((ee_pose, proprio_obs, object_poses, image_encoding), dim=-1)

        return joint

    def reset_episode(self):
        self._last_image_encoding = None
        self._last_image_enc_info = None

    def get_viz_encoder_callback(self) -> Callable | None:
        """
        Get the visualization encoder callback. For live visualization in real world environments before the episode starts.
        """
        if self.image_encoder is None:
            return None
        else:
            return self.image_encoder.encode
