from dataclasses import dataclass

import torch
import torchvision
from omegaconf import DictConfig

import tapas_gmm.encoder.keypoints
from tapas_gmm.utils.geometry_torch import (
    batched_pinhole_projection_image_to_world_coordinates,
    batched_project_onto_cam,
)


@dataclass
class DiscreteFilterConfig:
    motion_model_noisy: bool
    motion_model_kernel: int
    motion_model_sigma: float
    use_motion_model: bool


class DiscreteFilter:
    def __init__(self, config: DictConfig) -> None:
        filter_config: DiscreteFilterConfig = config.encoder_config.filter

        self.n_cams = len(config.observation.cameras)

        if filter_config.motion_model_noisy:
            self.motion_blur = torchvision.transforms.GaussianBlur(
                kernel_size=filter_config.motion_model_kernel,
                sigma=filter_config.motion_model_sigma,
            )
        else:
            self.motion_blur = None

        self.use_motion_model = filter_config.use_motion_model

        self.reset()

    def reset(self):
        self.last_kp_raw_2d = tuple((None for _ in range(self.n_cams)))

        self.last_post = tuple((None for _ in range(self.n_cams)))

        self.last_d = tuple((None for _ in range(self.n_cams)))
        self.last_int = tuple((None for _ in range(self.n_cams)))
        self.last_ext = tuple((None for _ in range(self.n_cams)))

    def get_prior(
        self,
        rgb: tuple,
        depth: tuple,
        extr: tuple,
        intr: tuple,
    ) -> tuple:
        n_cams = self.n_cams

        assert len(rgb) == n_cams

        prior = self.last_post
        motion_model = tuple(
            self.get_motion_model(
                depth[i],
                intr[i],
                extr[i],
                self.last_d[i],
                self.last_int[i],
                self.last_ext[i],
            )
            for i in range(n_cams)
        )

        prior_after_motion = tuple(
            self.apply_motion_model(prior[i], motion_model[i]) for i in range(n_cams)
        )

        return prior_after_motion

    def get_motion_model(
        self, depth_a, intr_a, extr_a, depth_b, intr_b, extr_b
    ) -> torch.Tensor | None:
        """
        Simple wrapper first checking wether the motion model is needed.
        """
        if depth_b is None or not self.use_motion_model:
            return None

        return self._get_motion_model(depth_a, intr_a, extr_a, depth_b, intr_b, extr_b)

    @staticmethod
    def _get_motion_model(depth_a, intr_a, extr_a, depth_b, intr_b, extr_b):
        """
        Computes the new position of all pixels of img_b in img_a. Doing so
        in this way (and not the other way around), directly gives us pixel
        coordinates for all positions in descriptor_b to read from for descr_a.
        Makes things a bit smoother as the mapping might neither be injective,
        nor surjective.
        Instead of interpolating between pixels, just round to neareas pixel
        position (currently done in apply_motion_model).
        In case of a pixel position outside the view frustrum (eg. img_b has a
        wider perspective than img_a), just take the nearest value inside the
        frustrum, ie. pad descriptor_b with the outer-most values if needed.
        closest value

        Parameters
        ----------
        depth_a : Tensor (B, H, W)
        intr_a : Tensor (B, 3, 3)
        extr_a : Tensor (B, 4, 4)
        depth_b : Tensor (B, H, W)
        intr_b : Tensor (B, 3, 3)
        extr_b : Tensor (B, 4, 4)

        Returns
        -------
        Tensor (B, H, W, 2)
        """
        B, H, W = depth_a.shape

        # create pixel coordinates
        px_u = torch.arange(0, W, device=depth_a.device)
        px_v = torch.arange(0, H, device=depth_a.device)
        # (B, H*W, 2)
        # cartesian product varies first dim first, so need to swap dims as
        # u,v coordinates are 'right, down'
        px_vu = torch.cartesian_prod(px_u, px_v).unsqueeze(0).repeat(B, 1, 1)

        # project pixel coordinates of current img into pixel space of last cam
        world = batched_pinhole_projection_image_to_world_coordinates(
            px_vu[..., 1], px_vu[..., 0], depth_a.reshape(B, H * W), intr_a, extr_a
        )
        cam_b = batched_project_onto_cam(world, depth_b, intr_b, extr_b, clip=False)
        cam_b = cam_b.reshape((B, H, W, 2))

        return cam_b

    def apply_motion_model(self, descriptor, motion_model):
        """
        Simple wrapper suppling the motion_blur func of self.
        """

        return self._apply_motion_model(
            descriptor, motion_model, blur_func=self.motion_blur
        )

    @staticmethod
    def _apply_motion_model(descriptor, motion_model, blur_func=None):
        """
        Apply the motion model to the descriptor image, i.e. for each pixel,
        set the value of the returned image to the value of the pixel in the
        image specified by the motion model.

        The motion model is in pixel space, so subsample to descriptor size
        first.

        Parameters
        ----------
        descriptor :Tensor (B, N, H, W)
        motion_model : Tensor (B, H', W', 2)

        Returns
        -------
        Tensor (B, N, H, W)
        """
        if motion_model is None:
            return descriptor

        B, N, H, W = descriptor.shape
        _, H2, W2, _ = motion_model.shape

        # downsample motion model and map to new range
        motion_model = torch.movedim(motion_model, 3, 1)
        motion_model = torch.nn.functional.interpolate(
            motion_model, size=(H, W), mode="bilinear", align_corners=True
        )
        motion_model = torch.movedim(motion_model, 1, 3)
        motion_model[..., 0] = torch.clamp(
            torch.round(motion_model[..., 0] / W2 * W), 0, W - 1
        )
        motion_model[..., 1] = torch.clamp(
            torch.round(motion_model[..., 1] / H2 * H), 0, H - 1
        )

        # add extra kp-dim and flatten
        motion_model = motion_model.unsqueeze(1).repeat(1, N, 1, 1, 1)
        mm_flat = motion_model.reshape((B * N * H * W, 2)).long()

        batch_indeces = [i for i in range(B) for _ in range(N * H * W)]
        kp_indeces = [i for _ in range(B) for i in range(N) for _ in range(H * W)]
        # motion model is in uv coordinates, so 'swap' dim order
        new_img_flat = descriptor[
            batch_indeces, kp_indeces, mm_flat[..., 1], mm_flat[..., 0]
        ]

        new_img = new_img_flat.reshape((B, N, H, W))

        new_img_wo_blur = new_img

        if blur_func is not None:
            new_img = blur_func(new_img)

        assert not torch.equal(new_img_wo_blur, new_img)

        # re-normalize
        new_img /= torch.sum(new_img, dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)

        return new_img
