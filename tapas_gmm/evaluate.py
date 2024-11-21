import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa
import wandb
from tapas_gmm.env import Environment, import_env
from tapas_gmm.env.environment import BaseEnvironment, BaseEnvironmentConfig
from tapas_gmm.policy import import_policy
from tapas_gmm.policy.policy import Policy, PolicyConfig
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.config import value_not_set
from tapas_gmm.utils.disturbance import disturbe_at_step_no
from tapas_gmm.utils.keyboard_observer import (
    KeyboardObserver,
    wait_for_environment_reset,
)

# from tapas_gmm.utils.misc import loop_sleep
from tapas_gmm.utils.misc import DataNamingConfig, policy_checkpoint_name
from tapas_gmm.utils.observation import SceneObservation, random_obs_dropout
from tapas_gmm.utils.random import configure_seeds
from tapas_gmm.utils.robot_trajectory import RobotTrajectory
from tapas_gmm.utils.select_gpu import device
from tapas_gmm.utils.tasks import get_task_horizon
from tapas_gmm.viz.live_keypoint import LiveKeypoints

init_griper_state = 0.9 * torch.ones(1, device=device)


@dataclass
class EvalConfig:
    n_episodes: int
    seed: int

    obs_dropout: float | None

    viz: bool
    kp_per_channel_viz: bool
    show_channels: list[int] | None

    horizon: int | None = None
    fragment_length: int = -1

    disturbe_at_step: int | None = None
    hold_until_step: int | None = None

    sample_freq: int | None = None  # downsample the trajectories to this freq
    sample_correction: int = 1  # correction factor for subsampled data


@dataclass
class Config:
    env: BaseEnvironmentConfig
    eval: EvalConfig
    policy: PolicyConfig
    data_naming: DataNamingConfig

    policy_type: str = "encoder"
    wandb_mode: str = "online"


def calculate_repeat_action(config: EvalConfig, def_freq=20):
    if config.sample_freq:
        repeat_action = int(config.sample_correction * def_freq / config.sample_freq)
        logger.info(
            "Sample freq {}, correction {}, thus repeating actions {}x.",
            config.sample_freq,
            config.sample_correction,
            repeat_action,
        )
    else:
        repeat_action = 1

    return repeat_action


def run_simulation(
    env: BaseEnvironment,
    policy: Policy,
    episodes: int,
    keypoint_viz: LiveKeypoints | None = None,
    horizon: int | None = 300,
    repeat_action: int = 1,
    fragment_len: int = -1,
    obs_dropout: float | None = None,
    keyboard_obs: KeyboardObserver | None = None,
    disturbe_at_step: int | None = None,
    hold_until_step: int | None = None,
):
    successes = 0
    episodes_count = 0

    # time.sleep(10)

    # env.reset()  # extra reset to ensure proper camera placement in RLBench

    try:
        with tqdm(total=episodes) as pbar:
            while episodes_count < episodes:
                wait_for_environment_reset(env, keyboard_obs)

                episode_reward, episode_length = run_episode(
                    env=env,
                    keyboard_obs=keyboard_obs,
                    policy=policy,
                    horizon=horizon,
                    keypoint_viz=keypoint_viz,
                    obs_dropout=obs_dropout,
                    repeat_action=repeat_action,
                    fragment_len=fragment_len,
                    disturbe_at_step=disturbe_at_step,
                    hold_until_step=hold_until_step,
                )

                if episode_reward > 0:
                    successes += 1

                wandb.log(
                    {
                        "reward": episode_reward,
                        "episode": episodes_count,
                        "eps_lenght": episode_length,
                    }
                )

                episodes_count += 1
                pbar.update(1)

                if keypoint_viz is not None:
                    keypoint_viz.reset()

                env.recover_from_errors()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ..")
        success_rate = successes / (episodes_count + 1)
        logger.info("Succes rate so far: {}", success_rate)
    else:
        logger.info("Done!")
        success_rate = successes / episodes
        wandb.run.summary["success_rate"] = success_rate  # type: ignore
        logger.info("Succes rate: {}", success_rate)
    finally:
        env.close()


def run_episode(
    env: BaseEnvironment,
    keyboard_obs: KeyboardObserver | None,
    policy: Policy,
    horizon: int | None,
    keypoint_viz: LiveKeypoints | None,
    obs_dropout: float | None,
    repeat_action: int,
    fragment_len: int,
    disturbe_at_step: int | None,
    hold_until_step: int | None,
) -> tuple[float, int]:
    episode_reward = 0
    done = False
    obs = env.reset()

    if keyboard_obs is not None:
        keyboard_obs.reset()

    policy.reset_episode(env=env)

    action = None
    step_no = 0

    pbar = tqdm(total=horizon or 1000)
    while True:
        action, step_reward, obs, done = process_step(
            obs=obs,
            obs_dropout=obs_dropout,
            policy=policy,
            action=action,
            repeat_action=repeat_action,
            keypoint_viz=keypoint_viz,
            fragment_len=fragment_len,
            env=env,
            keyboard_obs=keyboard_obs,
            horizon=horizon,
            step_no=step_no,
            disturbe_at_step=disturbe_at_step,
            hold_until_step=hold_until_step,
        )

        if step_reward > 0:
            logger.info("Got step reward: {}", step_reward)

        episode_reward += step_reward

        if done:
            logger.info("Episode done.")
            break

        step_no += 1
        pbar.update(1)

    if keyboard_obs is not None:
        keyboard_obs.reset()

    return episode_reward, step_no + 1


def process_step(
    obs: SceneObservation,
    obs_dropout: float | None,
    policy: Policy,
    action: np.ndarray | None,
    repeat_action: int,
    keypoint_viz: LiveKeypoints | None,
    fragment_len: int,
    env: BaseEnvironment,
    keyboard_obs: KeyboardObserver | None,
    horizon: int | None,
    step_no: int,
    disturbe_at_step: int | None,
    hold_until_step: int | None,
) -> tuple[np.ndarray, float, SceneObservation, bool]:
    # start_time = time.time()

    if fragment_len != -1 and step_no % fragment_len == 0:
        logger.info("Resetting LSTM state.")
        policy.reset_episode(env=env)

    disturbe_at_step_no(env, step_no, disturbe_at_step)
    obs = random_obs_dropout(obs, obs_dropout)

    # TODO: still need the case where action is None. But can rewrite else?
    obs.gripper_state = (
        init_griper_state
        if action is None
        else (
            action[-1].gripper
            if isinstance(action, RobotTrajectory)
            else torch.Tensor(action[-1, None]).to(device)
        )
    )

    action, info = policy.predict(obs)

    if hold_until_step is not None and step_no < hold_until_step:
        action = np.nan * np.ones_like(action)

    step_reward = 0
    done = False

    if type(action) is RobotTrajectory:
        # NOTE: hack-y: abusing the repeat_action to step through the trajectory
        repeat_action = len(action)
        # TODO: with a real robot, this still desynchronizes this loop and the robot
        # This loop needs essentially zero time, so it will run away from the robot
        # leading to unexpected behavior (no prediction for the robot before the
        # trajectory is done).
        # This is because it only gets the done flag from the env when the latter
        # has stepped through the traj: https://github.com/vonHartz/MT-GMM/blob/0ae7fb92c78fd1ebd8b6108820c55367107a3bdc/tapas_gmm/env/franka.py#L520C13-L520C59
        # That happens in a thread now and this loop here is faster.
        # I can hold it in check through updating the visualization
        # which takes some time: https://github.com/vonHartz/MT-GMM/blob/0ae7fb92c78fd1ebd8b6108820c55367107a3bdc/conf/evaluate/franka/coffee.py#L22
        # Should solve this properly.
        # Eg if we have a time RobotTrajectory already, we can use the timing
        # of the steps to synchronize the loop with the robot by waiting in
        # this loop until the next step is due.

    assert repeat_action > 0

    logger.info("Repeating action {}x.", repeat_action)

    for _ in tqdm(range(repeat_action)):
        if keypoint_viz is not None:
            keypoint_viz.update_from_info(info, obs)

        # NOTE: made diffusion policy return a robot trajectory now as well.
        # TODO: make this work with simulated envs. Either step through the traj
        # here or enable the envs to step through the traj themselves and aggregate
        # the rewards.
        next_obs, reward, done, env_info = env.step(action)

        if type(action) is RobotTrajectory:
            action = None  # Only pass the trajectory once, let the env step through it

        # if len(action.shape) == 1:
        #     next_obs, reward, done, env_info = env.step(action)
        # elif len(action.shape) == 2:
        #     # NOTE: hack-y: does not respect the horizon (counted as single step)
        #     reward = 0
        #     for a in action:
        #         next_obs, r, done, env_info = env.step(a)
        #         reward += r
        #         if done:
        #             break
        # else:
        #     raise ValueError(f"Unexpected number of action shape dims {action.shape}")

        done = done or env_info.get("done", False)

        step_reward += reward

        if step_no == horizon:
            done = True

        if keyboard_obs is not None:  # For real robot only
            if keyboard_obs.success:
                logger.info("Resetting with success.")
                reward = 1
                done = True
            elif keyboard_obs.reset_button:
                logger.info("Resetting without success.")
                reward = 0
                done = True
            else:
                reward = 0

        env.propose_update_visualization(info)

        obs = next_obs

        # loop_sleep(start_time)

        if done:
            break

    return action, step_reward, obs, done


def main(config: Config):
    Env = import_env(config.env)

    if config.env.env_type is Environment.PANDA:
        keyboard_obs = KeyboardObserver()
    else:
        keyboard_obs = None

    Policy = import_policy(config.policy_type)
    policy = Policy(config.policy).to(device)

    file_name, _ = policy_checkpoint_name(config)  # type: ignore
    logger.info("Loading policy checkpoint from {}", file_name)
    policy.from_disk(file_name)
    policy.eval()

    logger.info("Creating env.")

    env = Env(
        config.env,
        viz_encoder_callback=policy.obs_encoder.get_viz_encoder_callback(),
    )  # type: ignore

    keypoint_viz = LiveKeypoints.setup_from_conf(config, policy)

    repeat_action = calculate_repeat_action(config.eval)

    logger.info("Running simulation.")

    run_simulation(
        env,
        policy,
        config.eval.n_episodes,
        keypoint_viz=keypoint_viz,
        horizon=config.eval.horizon,
        repeat_action=repeat_action,
        fragment_len=config.eval.fragment_length,
        obs_dropout=config.eval.obs_dropout,
        keyboard_obs=keyboard_obs,
        disturbe_at_step=config.eval.disturbe_at_step,
        hold_until_step=config.eval.hold_until_step,
    )


def complete_config(config: DictConfig) -> DictConfig:
    if hasattr(config.policy, "encoder_naming"):
        if value_not_set(config.policy.encoder_naming.data_root):
            config.policy.encoder_naming.data_root = config.data_naming.data_root

        if value_not_set(config.policy.encoder_naming.task):
            config.policy.encoder_naming.task = config.data_naming.task

        if value_not_set(config.policy.encoder_naming.feedback_type):
            config.policy.encoder_naming.feedback_type = (
                config.data_naming.feedback_type
            )

    if hasattr(config.policy, "observation") and value_not_set(
        config.policy.observation.image_dim
    ):
        config.policy.observation.image_dim = config.env.image_size
        # TODO: also need update for crop in panda env?

    config.env.task = config.data_naming.task

    config.eval.horizon = config.eval.horizon or get_task_horizon(config.env)

    return config


def entry_point():
    args, dict_config = parse_and_build_config()
    dict_config = complete_config(dict_config)  # type: ignore

    seed = configure_seeds(dict_config.eval.seed)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    wandb.init(
        config=OmegaConf.to_container(dict_config, resolve=True),  # type: ignore
        project="TAPAS-GMM",
        mode=config.wandb_mode,
    )  # type: ignore

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
