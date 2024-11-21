import abc
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce as einops_reduce
from loguru import logger

from tapas_gmm.dataset.bc import BCDataset
from tapas_gmm.encoder.encoder import ObservationEncoder, ObservationEncoderConfig
from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.policy.models.diffusion.conditional_unet1d import (
    ConditionalUnet1D,
    ConditionalUnet1DConfig,
)
from tapas_gmm.policy.models.diffusion.lr_scheduler import get_scheduler
from tapas_gmm.policy.models.diffusion.mask_generator import LowdimMaskGenerator
from tapas_gmm.policy.models.diffusion.module_attr_mixin import ModuleAttrMixin
from tapas_gmm.policy.models.diffusion.normalizer import LinearNormalizer
from tapas_gmm.policy.policy import Policy, PolicyConfig
from tapas_gmm.utils.config import _SENTINELS
from tapas_gmm.utils.geometry_np import normalize_quaternion
from tapas_gmm.utils.logging import log_constructor
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig, SceneObservation
from tapas_gmm.utils.robot_trajectory import RobotTrajectory
from tapas_gmm.utils.select_gpu import device

EncoderConfig = Any  # BVAEConfig | CNNConfig | KeypointsPredictorConfig | \


# TODO: convert action labels to 6D? convert 6D predictions to quaternion?


@dataclass
class DiffusionPolicyTrainingConfig:
    lr: float = 1.0e-4
    betas: tuple[float, ...] = (0.95, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 1.0e-6

    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    lr_num_epochs: int = 5000
    gradient_accumulate_every: int = 1  # NOTE: did not implement for values > 1 yet
    use_ema: bool = True


@dataclass
class DDPMSchedulerConfig:
    num_train_timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "squaredcos_cap_v2"
    variance_type: str = "fixed_small"
    clip_sample: bool = True  # required when predict_epsilon=False
    prediction_type: str = "epsilon"  # or sample


@dataclass
class DiffusionPolicyConfig(PolicyConfig):
    suffix: str | None

    obs_as_local_cond: bool
    obs_as_global_cond: bool
    pred_action_steps_only: bool

    # visual_embedding_dim: int | _SENTINELS
    # proprio_dim: int
    action_dim: int
    obs_dim: int | _SENTINELS

    n_action_steps: int
    n_obs_steps: int
    horizon: int

    oa_step_convention: bool

    num_inference_steps: int | None

    # lstm_layers: int
    # use_ee_pose: bool
    # add_object_poses: bool

    training: DiffusionPolicyTrainingConfig | None

    action_scaling: bool

    unet: ConditionalUnet1DConfig
    scheduler: DDPMSchedulerConfig = DDPMSchedulerConfig()

    encoder_name: str | None = None
    encoder_config: EncoderConfig | None = None
    encoder_suffix: str | None | None = None
    encoder_naming: DataNamingConfig | None = None
    observation: ObservationConfig | None = None

    obs_encoder: ObservationEncoderConfig | None = None


def zero_pad_timed_tensor(obs_stack: torch.Tensor, n_steps: int) -> torch.Tensor:
    assert len(obs_stack.shape) == 2
    n_obs = len(obs_stack)
    assert n_obs > 0

    padded_steps = n_steps - n_obs
    zeros = torch.zeros_like(obs_stack[0]).unsqueeze(0).repeat(padded_steps, 1)

    return torch.cat((zeros, obs_stack), dim=0)


class DiffusionPolicy(Policy, ModuleAttrMixin):
    @log_constructor
    def __init__(
        self,
        config: DiffusionPolicyConfig,
        skip_module_init: bool = False,
        encoder_checkpoint: str | None = None,
        **kwargs,
    ):
        super().__init__(config, skip_module_init, **kwargs)

        assert not (config.obs_as_local_cond and config.obs_as_global_cond)
        if config.pred_action_steps_only:
            assert config.obs_as_global_cond

        self.model = ConditionalUnet1D(config.unet)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.scheduler.num_train_timesteps,
            beta_start=config.scheduler.beta_start,
            beta_end=config.scheduler.beta_end,
            beta_schedule=config.scheduler.beta_schedule,
            variance_type=config.scheduler.variance_type,
            clip_sample=config.scheduler.clip_sample,
            prediction_type=config.scheduler.prediction_type,
        )
        self.mask_generator = LowdimMaskGenerator(
            action_dim=config.action_dim,
            obs_dim=(
                0
                if (config.obs_as_local_cond or config.obs_as_global_cond)
                else config.obs_dim
            ),
            max_n_obs_steps=config.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        # TODO: move to BatchEncoder?
        self.normalizer = LinearNormalizer()

        if config.training is not None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config.training.lr,
                betas=config.training.betas,
                eps=config.training.eps,
                weight_decay=config.training.weight_decay,
            )

            self.lr_scheduler = None

        if config.num_inference_steps is None:
            config.num_inference_steps = config.scheduler.num_train_timesteps

        self.config = config

        self.kwargs = kwargs  # NOTE: from their code. Might conflict with mine

        self._obs_que = None  # is set in reset_episode

        self.obs_encoder = ObservationEncoder(config.obs_encoder, encoder_checkpoint)

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self.obs_encoder.reset_episode()

        self._obs_que = deque(maxlen=self.config.n_obs_steps)  # + 1)

        self._env = env

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_lr_scheduler(self, n_steps_per_epoch: int) -> None:
        self.lr_scheduler = get_scheduler(
            self.config.training.lr_scheduler,
            self.optimizer,
            num_warmup_steps=self.config.training.lr_warmup_steps,
            num_training_steps=self.config.training.lr_num_epochs * n_steps_per_epoch,
            # last_epoch=self.global_step-1  # NOTE: don't know what this is
        )

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.config.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert "obs" in obs_dict
        assert "past_action" not in obs_dict  # not implemented yet

        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.config.n_obs_steps
        assert Do == self.config.obs_dim
        T = self.config.horizon
        Da = self.config.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None

        if self.config.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.config.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.config.pred_action_steps_only:
                shape = (B, self.config.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        if self.config.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.config.oa_step_convention:
                start = To - 1
            end = start + self.config.n_action_steps
            action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        if not (self.config.obs_as_local_cond or self.config.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer["obs"].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result["action_obs_pred"] = action_obs_pred
            result["obs_pred"] = obs_pred

        return result

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch["obs"]
        action = nbatch["action"]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.config.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:, self.config.n_obs_steps :, :] = 0
        elif self.config.obs_as_global_cond:
            global_cond = obs[:, : self.config.n_obs_steps, :].reshape(obs.shape[0], -1)
            if self.config.pred_action_steps_only:
                To = self.config.n_obs_steps
                start = To
                if self.config.oa_step_convention:
                    start = To - 1
                end = start + self.config.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.config.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=trajectory.device,
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory, timesteps, local_cond=local_cond, global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = einops_reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss

    def _get_policy_input_training(self, obs: SceneObservation) -> torch.Tensor:
        low_dim_input, _ = self._get_policy_input(obs)

        # remove last timestep
        return low_dim_input[:, :-1]

    def _get_policy_input(self, obs: SceneObservation) -> tuple[torch.Tensor, dict]:
        return self.obs_encoder.encode(obs)

    def _get_policy_target(self, obs: SceneObservation) -> torch.Tensor:
        robo_state = torch.cat((obs.ee_pose, obs.gripper_state), dim=-1)

        # remove first time step
        return robo_state[:, 1:]

    def forward_step(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError("Stepwise prediction not used in Diffusion Policy.")

    def forward(self, batch: SceneObservation) -> torch.Tensor:  # type: ignore
        self.reset_episode(None)

        obs = self._get_policy_input_training(batch)
        action = self._get_policy_target(batch)

        obs_dict = {"obs": obs, "action": action}

        loss = self.compute_loss(obs_dict)

        return loss

    def update_params(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        training_metrics = {
            "train-loss": loss,
            "lr": self.lr_scheduler.get_last_lr()[0],
        }
        return training_metrics

    def evaluate(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        loss = self.forward(batch)
        training_metrics = {"eval-loss": loss}
        return training_metrics

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[RobotTrajectory, dict]:
        self._obs_que.append(obs)
        with torch.no_grad():
            # obs = obs.unsqueeze(0).to(device)
            # action_th, info = self.forward_step(obs)
            # action_th = action_th * 1 / self.action_scale
            # action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            # action[-1] = binary_gripper(action[-1])
            # get all elements from the self._obs_que deque
            obs_stack = torch.stack(list(self._obs_que)).to(device)
            policy_input, visual_enc_info = self._get_policy_input(obs_stack)
            padded_policy_input = zero_pad_timed_tensor(
                policy_input, self.config.n_obs_steps
            )
            pred_dict = self.predict_action({"obs": padded_policy_input.unsqueeze(0)})

            # TODO: this dict repacking is HACK-y because I did it weirdly somewhere else
            # Make it straightforward, eg just add the raw visual_enc_info or sth
            # NOTE: indexing [0] because of multistep predictions in diffusion policy
            pred_dict["vis_encoding"] = [c[0] for c in visual_enc_info["kp_raw_2d"]]
            pred_dict["heatmap"] = [c[0] for c in visual_enc_info["post"]]

            action_stack = pred_dict["action"].detach().cpu().squeeze(0).numpy()

            # action_stack = action_stack[0]  # Only return first action from stack
            action_stack[..., 3:7] = normalize_quaternion(action_stack[..., 3:7])

            traj = RobotTrajectory.from_np(ee=action_stack[:7], gripper=action_stack[7])

        return traj, pred_dict

    def initialize_parameters_via_dataset(
        self,
        replay_memory: BCDataset,
        cameras: tuple[str],
        epoch_length: int | None = None,
    ) -> None:
        assert epoch_length is not None
        self.set_lr_scheduler(epoch_length)
        if self.config.action_scaling:
            logger.info(
                "Setting action scaling for optimal policy performance. "
                "Using DP normalizer implementation."
            )

            action_distribution = torch.cat(
                (
                    replay_memory.scene_data.get_ee_pose_distribution(),
                    replay_memory.scene_data.get_gripper_state_distribution(),
                ),
                dim=-1,
            ).to(device)

            obs_distribution = self.obs_encoder.get_obs_distribution(replay_memory).to(
                device
            )

            data = {"action": action_distribution, "obs": obs_distribution}

            # to prevent blowing up dims that barely change
            kwargs = {"range_eps": 5e-2}
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode="limits", **kwargs)

            self.set_normalizer(normalizer)


def binary_gripper(gripper_action: float) -> float:
    if gripper_action >= 0.0:
        gripper_action = 0.9
    elif gripper_action < 0.0:
        gripper_action = -0.9
    return gripper_action
