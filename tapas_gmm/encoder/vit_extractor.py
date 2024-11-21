from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from omegaconf import DictConfig

import tapas_gmm.encoder.representation_learner as representation_learner
from tapas_gmm.encoder.keypoints import (
    KeypointsConfig,
    KeypointsPredictor,
    PriorTypes,
    ProjectionTypes,
    ReferenceSelectionTypes,
)
from tapas_gmm.encoder.models.vit_extractor.extractor import (
    VitEncoderModel,
    VitFeatureModelConfig,
)
from tapas_gmm.filter.discrete_filter import DiscreteFilter, DiscreteFilterConfig
from tapas_gmm.filter.particle_filter import ParticleFilter, ParticleFilterConfig
from tapas_gmm.utils.debug import measure_runtime
from tapas_gmm.utils.observation import SampleTypes, SceneObservation
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.viz.operations import channel_back2front_batch


@dataclass
class VitFeatureEncoderConfig:
    encoder: VitFeatureModelConfig

    frozen: bool = True


class VitFeatureEncoder(representation_learner.RepresentationLearner):
    # sample_type = None  # does not need pretraining
    sample_type = SampleTypes.DC  # for correspondence viz

    def __init__(self, config: DictConfig):
        super().__init__(config=config)

        encoder_config: VitFeatureEncoderConfig = config.encoder_config
        self.config = encoder_config

        self.extractor = VitEncoderModel(encoder_config.encoder, device, model=None)

        self.torch_context = (
            torch.inference_mode() if encoder_config.frozen else nullcontext()
        )

        self._get_descriptor_resolution(
            encoder_config.encoder, config.observation.image_dim
        )

    def _get_descriptor_resolution(
        self, encoder_config: VitFeatureModelConfig, image_dim: tuple[int, int]
    ):
        # TODO: make more general

        if "dinov2" in encoder_config.vision_net:
            assert encoder_config.load_size is not None
            H_descr = W_descr = encoder_config.load_size // 14
            if encoder_config.stride == 7:
                H_descr = H_descr * 2 - 1
                W_descr = W_descr * 2 - 1
            else:
                assert encoder_config.stride == 14

            if encoder_config.center_crop:
                H_descr -= 4
                W_descr -= 4
            elif encoder_config.pad:
                H_descr += 2
                W_descr += 2
        else:
            H_descr = image_dim[0] // 8
            W_descr = image_dim[1] // 8
            # H_descr = W_descr = 32
            if encoder_config.stride == 4:
                H_descr = H_descr * 2 - 1
                W_descr = W_descr * 2 - 1
            else:
                assert encoder_config.stride == 8

        self.H_descr, self.W_descr = H_descr, W_descr

    def encode(self, batch: SceneObservation) -> tuple[torch.Tensor, dict]:
        camera_obs = batch.camera_obs

        rgb = tuple((o.rgb for o in camera_obs))

        enc = tuple(self.compute_descriptor_batch(img) for img in rgb)

        enc = torch.cat(enc, dim=-1)

        info = {}

        return enc, info

    @measure_runtime
    def compute_descriptor(
        self, camera_obs: torch.Tensor, upscale: bool = True
    ) -> torch.Tensor:
        camera_obs = camera_obs.to(device)

        _, H, W = camera_obs.shape

        with self.torch_context:
            prep = self.extractor.preprocess(camera_obs)
            descr = self.extractor.extract_descriptors(prep).squeeze(0)

        descr = descr.reshape(1, self.H_descr, self.W_descr, descr.shape[-1])
        descr = channel_back2front_batch(descr)

        if upscale:
            descr = torch.nn.functional.interpolate(
                input=descr, size=[H, W], mode="bilinear", align_corners=True
            )

        return descr

    def compute_descriptor_batch(
        self, camera_obs: torch.Tensor, upscale: bool = True
    ) -> torch.Tensor:
        camera_obs = camera_obs.to(device)

        B, _, H, W = camera_obs.shape

        with self.torch_context:
            prep = self.extractor.preprocess(camera_obs)
            descr = self.extractor.extract_descriptors(prep).squeeze(0)

        descr = descr.reshape(B, self.H_descr, self.W_descr, descr.shape[-1])
        descr = channel_back2front_batch(descr)

        if upscale:
            descr = torch.nn.functional.interpolate(
                input=descr, size=[H, W], mode="bilinear", align_corners=True
            )

        return descr

    def generate_mask(self, camera_obs: torch.Tensor) -> torch.Tensor:
        prep = self.extractor.preprocess(camera_obs)

        saliency_map = self.extractor.extract_saliency_maps(prep)[0]
        fg_mask = saliency_map > self.config.encoder.thresh
        fg_mask = fg_mask.reshape(self.H_descr, self.W_descr)

        return fg_mask

    def from_disk(self, *args, **kwargs):
        logger.info("This encoder needs no chekpoint loading.")


@dataclass
class VitKeypointsEncoderConfig:
    vit: VitFeatureModelConfig

    descriptor_dim: int
    keypoints: KeypointsConfig
    prior_type: PriorTypes
    projection: ProjectionTypes
    taper_sm: int | float
    use_spatial_expectation: bool
    cosine_distance: bool

    threshold_keypoint_dist: float | None
    overshadow_keypoints: bool

    add_noise_scale: float | None

    frozen: bool = True


@dataclass
class PreTrainingConfig:
    ref_selection: ReferenceSelectionTypes = ReferenceSelectionTypes.MANUAL
    ref_labels: tuple[int, ...] | None = None
    ref_traj_idx: int | None = None
    ref_obs_idx: int | None = None
    ref_selector: Any = None


@dataclass(kw_only=True)
class VitKeypointsPredictorConfig(representation_learner.RepresentationLearnerConfig):
    encoder: VitKeypointsEncoderConfig

    image_dim: tuple[int, int]

    # filter: DiscreteFilterConfig | ParticleFilterConfig | None = None
    filter: Any = None

    pretraining: PreTrainingConfig = PreTrainingConfig()

    debug_kp_selection: bool = False
    debug_kp_encoding: bool = False

    disk_read_keypoints: bool = False


class VitKeypointsPredictor(KeypointsPredictor):
    sample_type = SampleTypes.DC  # for correspondence viz

    def __init__(self, config: VitKeypointsPredictorConfig):
        representation_learner.RepresentationLearner.__init__(self, config=config)

        # NOTE: these are still used directly in KeyPointsPredictor
        # TODO: switch to self.config.<attr> everywhere
        self.disk_read_keypoints = config.disk_read_keypoints
        self.disk_read_embedding = config.disk_read_embedding

        if not (self.config.disk_read_keypoints or self.config.disk_read_embedding):
            self.configure_dc_maps(config)

        if self.config.disk_read_keypoints:
            logger.info("Reading precomputed keypoints from disk.")

        encoder_config = config.encoder

        self.config = encoder_config

        self.descriptor_dimension = encoder_config.descriptor_dim
        self.add_noise_scale = encoder_config.add_noise_scale
        self.prior_type = encoder_config.prior_type
        self.overshadow = encoder_config.overshadow_keypoints
        self.threshold = encoder_config.threshold_keypoint_dist
        self.taper = encoder_config.taper_sm
        self.use_spatial_expectation = encoder_config.use_spatial_expectation
        self.projection = encoder_config.projection
        self.cosine_distance = encoder_config.cosine_distance

        self.keypoint_dimension = 2 if self.projection is ProjectionTypes.NONE else 3

        self._register_buffers()

        self.debug_kp_selection = config.debug_kp_selection
        self._setup_pretraining(config.pretraining)

        self._setup_filter(config)

        self.reset_episode()

        self.extractor = VitEncoderModel(encoder_config.vit, device, model=None)

        self.torch_context = (
            torch.inference_mode() if encoder_config.frozen else nullcontext()
        )

        VitFeatureEncoder._get_descriptor_resolution(
            self, encoder_config.vit, config.image_dim
        )

    def _register_buffers(self):
        # Register references as buffers, such that they will be saved with the
        # module. Then, we can use the same reference vectors at inference.
        # NOTE: does not have mean, std buffers, as image norm is hardcoded in
        # the vit model.
        n_keypoints = self.get_no_keypoints()
        self.register_buffer("ref_pixels_uv", torch.Tensor(2, n_keypoints))
        self.register_buffer(
            "_reference_descriptor_vec",
            torch.Tensor(n_keypoints, self.descriptor_dimension),
        )

    def _setup_pretraining(self, pretrain_config):
        self.pretrain_config = pretrain_config
        logger.info("No pretraining`")

    def compute_descriptor(self, camera_obs: torch.Tensor) -> torch.Tensor:
        return VitFeatureEncoder.compute_descriptor(self, camera_obs)

    def compute_descriptor_batch(
        self, camera_obs: torch.Tensor, upscale=True
    ) -> torch.Tensor:
        with torch.inference_mode():
            return VitFeatureEncoder.compute_descriptor_batch(
                self, camera_obs, upscale=upscale
            )

    def set_model_image_normalization(self):
        logger.warning("No need for img norm setup for Vit? Done in model.")

    def from_disk(
        self, chekpoint_path: str, ignore: tuple[str, ...] | None = None
    ) -> None:
        logger.info("Loading encoder checkpoint from {}", chekpoint_path)
        if ignore is None:
            ignore = ()

        if set(ignore) == set(("_reference_descriptor_vec", "ref_pixels_uv")):
            logger.info(
                "Skipping ViT disk loading. Not needed if the model "
                "was not customized."
            )
            return
        else:
            state_dict = torch.load(chekpoint_path, map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if k not in ignore}

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing and set(ignore) != set(missing):
                logger.warning("Missing keys: {}".format(missing))
            if unexpected:
                logger.warning("Unexpected keys: {}".format(unexpected))
            self = self.to(device)
