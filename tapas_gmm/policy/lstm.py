import abc
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tensordict import TensorDict
from torch.distributions.normal import Normal

from tapas_gmm.dataset.bc import BCDataset
from tapas_gmm.env.environment import BaseEnvironment
from tapas_gmm.policy.models.diffusion.normalizer import LinearNormalizer
from tapas_gmm.policy.policy import Policy, PolicyConfig
from tapas_gmm.utils.config import _SENTINELS
from tapas_gmm.utils.geometry_torch import frame_transform_pos_quat
from tapas_gmm.utils.logging import log_constructor
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import SceneObservation
from tapas_gmm.utils.select_gpu import device  # , normalize_quaternion


@dataclass
class LSTMPolicyTrainingConfig:
    learning_rate: float
    weight_decay: float


@dataclass
class LSTMPolicyConfig(PolicyConfig):
    suffix: str | None

    visual_embedding_dim: int | _SENTINELS
    proprio_dim: int
    action_dim: int
    lstm_layers: int

    use_ee_pose: bool
    add_gripper_state: bool
    add_object_poses: bool
    poses_in_ee_frame: bool

    training: LSTMPolicyTrainingConfig | None

    action_scaling: bool


class LSTMPolicy(Policy):
    @log_constructor
    def __init__(
        self, config: LSTMPolicyConfig, skip_module_init: bool = False, **kwargs
    ):
        super().__init__(config, skip_module_init, **kwargs)

        assert type(config.visual_embedding_dim) is int

        lstm_dim = config.visual_embedding_dim + config.proprio_dim

        self.lstm = nn.LSTM(lstm_dim, lstm_dim, num_layers=config.lstm_layers)

        self.linear_out = nn.Linear(lstm_dim, config.action_dim)

        self.normalizer = LinearNormalizer()

        if config.training is None:
            logger.info("Got None training config, skipping optimizer init.")
        else:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
            )

        self.std = 0.1 * torch.ones(config.action_dim, dtype=torch.float32)
        self.std = self.std.to(device)

        if self.config.add_object_poses:
            logger.warning(
                "Did not implement elegant way to auto-determine the dim of the object poses. "
                "Currently need to be set in visual_embedding_dim in the config."
            )
            logger.warning(
                "Object poses are currently given as pos + quaternion. Should probably switch "
                "the rotation representation, eg to euler angles."
            )

        self._lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_episode(self, env: BaseEnvironment | None = None) -> None:
        self._lstm_state = None

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward_step(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[torch.Tensor, dict]:
        low_dim_input, info = self._get_policy_input(obs)

        if self.config.action_scaling:
            low_dim_input = self.normalizer["obs"].normalize(low_dim_input)

        lstm_out, self._lstm_state = self.lstm(low_dim_input, self._lstm_state)

        out = torch.tanh(self.linear_out(lstm_out))

        return out, info

    def _prep_pose_dict(
        self, object_poses: TensorDict, ee_poses: torch.Tensor
    ) -> torch.Tensor:
        if self.config.add_object_poses:
            obj_poses = torch.cat(
                [
                    (
                        frame_transform_pos_quat(v, ee_poses)
                        if self.config.poses_in_ee_frame
                        else v
                    )
                    for _, v in sorted(object_poses.items())
                ],
                dim=-1,
            )

        else:
            obj_poses = torch.Tensor()

        return obj_poses

    def _get_policy_input(self, obs: SceneObservation) -> tuple[torch.Tensor, dict]:
        vis_encoding, info = self._get_visual_input(obs)

        robo_state = obs.ee_pose if self.config.use_ee_pose else obs.proprio_obs

        obj_poses = self._prep_pose_dict(obs.object_poses, obs.ee_pose)

        if self.config.add_gripper_state:
            robo_state = torch.cat((robo_state, obs.gripper_state), dim=1)

        low_dim_input = torch.cat(
            (vis_encoding, robo_state, obj_poses), dim=-1
        ).unsqueeze(0)

        info["vis_encoding"] = vis_encoding

        return low_dim_input, info

    def _get_visual_input(self, obs: SceneObservation) -> tuple[torch.Tensor, dict]:
        return torch.empty(0, device=device), {}

    def forward(self, batch: SceneObservation) -> torch.Tensor:  # type: ignore
        losses = []
        self.reset_episode(None)

        time_steps = batch.shape[1]

        for t in range(time_steps):
            obs = batch[:, t, ...]
            mu, _ = self.forward_step(obs)
            distribution = Normal(mu, self.std)

            label = obs.action
            if self.config.action_scaling:
                label = self.normalizer["action"].normalize(label)

            log_prob = distribution.log_prob(label)
            loss = -log_prob * obs.feedback
            losses.append(loss)

        total_loss = torch.cat(losses).mean()

        return total_loss

    def update_params(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        loss.backward()
        self.optimizer.step()
        training_metrics = {"train-loss": loss}
        return training_metrics

    def evaluate(self, batch: SceneObservation) -> dict:  # type: ignore
        batch = batch.to(device)
        loss = self.forward(batch)
        training_metrics = {"eval-loss": loss}
        return training_metrics

    def predict(
        self,
        obs: SceneObservation,  # type: ignore
    ) -> tuple[np.ndarray, dict]:
        with torch.no_grad():
            obs = obs.unsqueeze(0).to(device)
            action_th, info = self.forward_step(obs)

            if self.config.action_scaling:
                action_th = self.normalizer["action"].unnormalize(action_th)

            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])

        return action, info

    def initialize_parameters_via_dataset(
        self, replay_memory: BCDataset, cameras: tuple[str], **kwargs
    ) -> None:
        if self.config.action_scaling:
            logger.info("Setting action scaling for optimal LSTM performance. ")
            action_distribution = replay_memory.scene_data.get_action_distribution().to(
                device
            )

            viz_encoding_distribution = torch.Tensor()

            ee_pose_distribution = replay_memory.scene_data.get_ee_pose_distribution()
            robot_state_distribution = (
                ee_pose_distribution
                if self.config.use_ee_pose
                else replay_memory.scene_data.get_proprio_distribution()
            )
            object_poses_distribution = self._prep_pose_dict(
                replay_memory.scene_data.get_object_poses_distribution(stacked=False),
                ee_pose_distribution,
            )
            gripper_state_distribution = (
                replay_memory.scene_data.get_gripper_state_distribution()
                if self.config.add_gripper_state
                else torch.Tensor()
            )

            obs_distribution = torch.cat(
                (
                    viz_encoding_distribution,
                    robot_state_distribution,
                    object_poses_distribution,
                    gripper_state_distribution,
                ),
                dim=-1,
            ).to(device)

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
