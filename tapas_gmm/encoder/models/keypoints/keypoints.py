from enum import Enum

from loguru import logger
from torch import nn
from torchvision import transforms

from tapas_gmm.encoder.models.keypoints import networks


class KeypointsTypes(Enum):
    # NOTE: these are the ones from KeyPoints into the future.
    SD = 1  # sample 50 random points from masked reference image
    SDS = 2  # sample 100, keep 4 or 5 that have high certainty and spread out
    WDS = 3  # convex combination of keypoints via learned weights
    WSDS = 4  # combines SDS and WDS
    # following are the ones from self-supervised correspondence. SD is same.
    ODS = 5  # optimize the descriptor set together with the policy
    E2E = 6  # train end-to-end, ie. DC net together with policy
    EPT = 7  # end-to-end with DC pretraining. apply diff operation to DC net
    # eg. 2D channel-wise spatial expectations to each chennel of the
    # descriptor -> use 16-D descriptor iamge to get 16 2D keypoints


# NOTE: on beginning of episode, need to sample keypoints via sample_keypoints.
# whether or not they are optimized depends on setting requires_grad.
# Forward should encode the rgb and then compute the spatial expectation of the
# keypoints, which is then the embedding.


class KeypointsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # in: img(3,H,W), out: descriptor (D,H,W)
        self.dense_correspondence_net = networks.get_fcn(config)

        # TODO: this belongs to the Keypoints-into-the-future. should probably
        # be a different class
        # in: Z,A, out: Z
        # self.dynamics_model = networks.DynamicsModel(
        #     config["descriptor_dim"], 128, config["action_dim"],
        #     hidden_dims=config["dynamics_hidden_dims"])

        # if "image_size" in config and config["image_size"] is not None:
        #     image_size = config["image_size"]
        # else:
        #     image_size = (256, 256)
        # self.image_height, self.image_width = image_size

        self.img_normalization = None
        self.normalize_images = config.normalize_images
        logger.info("Normalizing images: {}", self.normalize_images)

    def compute_descriptors(self, batch, upscale=True):
        # from tapas_gmm.utils.debug import summarize_tensor
        # print(summarize_tensor(batch, "batch"))
        if self.normalize_images:
            batch = self.img_normalization(batch)
        return self.dense_correspondence_net(batch, upscale=upscale)

    def forward(self, source_images, target_images): ...  # return reconstrution

    def setup_image_normalization(self, mean, std):
        logger.info("Setting up image normalization with mean {}, std {}.", mean, std)
        self.norm_mean = mean
        self.norm_std = std
        self.img_normalization = transforms.Normalize(
            mean=self.norm_mean, std=self.norm_std
        )
