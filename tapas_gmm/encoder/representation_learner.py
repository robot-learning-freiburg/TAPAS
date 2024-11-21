from abc import abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig

from tapas_gmm.dataset.dc import DenseCorrespondenceDataset
from tapas_gmm.utils.logging import indent_func_log
from tapas_gmm.utils.misc import get_and_log_failure as get_conf
from tapas_gmm.utils.observation import SceneObservation, SingleCamObservation
from tapas_gmm.utils.select_gpu import device

# TODO: split into VisualEncoder and RepresentationLearner. VisualEncoder does not
# have to be nn.Module and does not need training methods.


@dataclass(kw_only=True)
class RepresentationLearnerConfig:
    end_to_end: bool = False
    disk_read_embedding: bool = False


class RepresentationLearner(nn.Module):
    sample_type = None  # Needs to be set in subclass.

    embedding_name = "descriptor"

    def __init__(self, config: RepresentationLearnerConfig) -> None:
        super().__init__()

        self.config = config

        if self.config.disk_read_embedding:
            logger.info("Reading embedding from disk.")

    def forward(self, batch: SceneObservation, **kwargs):
        """
        For visualization, the keypoints encoder also returns and info dict,
        containing among other things the latent embedding- hence the second
        return value. If needed, other encoders can do so, too. Check the
        forward method of the keypoints encoder for reference.
        """
        return self.encode(batch)

    @abstractmethod
    def update_params(self, batch, **kwargs):
        """
        Different methods have different sample types. To unify the API, the
        batch is unpacked inside this function.
        Additionally, some methods (like BVAE) need additional information,
        hence the **kwargs.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, batch, batch_size=None, **kwargs) -> dict:
        raise NotImplementedError

    def encode(self, batch: SceneObservation) -> tuple[torch.Tensor, dict]:
        """
        Encode batches of SceneObservation, with potentially multiple cameras,
        into a single embedding per datapoint.

        Parameters
        ----------
        batch : SceneObservation
            TensorClass holding the observation batch.

        Returns
        -------
        torch.Tensor, dict
            Embedding an info dict.
        """
        camera_info = {}
        camera_embeddings = []

        for cam, obs in zip(batch.camera_names, batch.camera_obs):
            if self.disk_read_embedding:
                emb = getattr(obs, self.embedding_name)
                info = {}
            else:
                emb, info = self.encode_single_camera(obs)

            camera_embeddings.append(emb)
            camera_info[cam] = info

        return torch.cat(camera_embeddings, dim=-1), camera_info

    def reset_episode(self):
        """
        Method to signal to the encoder that a new trajectory is starting.
        Implementation is optional. Useful for maintaining stateful encoders.
        """
        pass

    def initialize_image_normalization(
        self, replay_memory: DenseCorrespondenceDataset, camera_name: tuple[str]
    ) -> None:
        """
        Initialize the image normalization parameters of the encoder.
        Implemenent in subclasses if needed.
        """
        pass

    def initialize_parameters_via_dataset(self, replay_memory, cam, **kwargs):
        logger.info("This encoder does not use dataset initialization.")

    @abstractmethod
    def encode_single_camera(
        self, batch: SingleCamObservation
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, batch):
        raise NotImplementedError

    # TODO: make this a static method?
    @classmethod
    @abstractmethod
    def get_latent_dim(cls, config, n_cams=1, image_dim=None):
        raise NotImplementedError

    @indent_func_log
    def from_disk(self, chekpoint_path):
        self.load_state_dict(torch.load(chekpoint_path, map_location=device))

    def to_disk(self, checkpoint_path):
        logger.info("Saving encoder checkpoint at {}", checkpoint_path)
        torch.save(self.state_dict(), checkpoint_path)

    @staticmethod
    def add_gaussian_noise(
        coordinates: torch.Tensor,
        noise_scale: float | torch.Tensor,
        skip_z: bool = False,
    ) -> torch.Tensor:
        stacked_coords = torch.stack(torch.chunk(coordinates, 3, dim=-1), dim=-1)

        gauss = torch.distributions.normal.Normal(0, noise_scale)
        noise = gauss.sample(stacked_coords.shape).to(device)

        if skip_z:
            assert noise.shape[-1] == 3
            noise[..., 2] = 0

        augmented = stacked_coords + noise

        return torch.cat(
            (augmented[..., 0], augmented[..., 1], augmented[..., 2]), dim=-1
        )
