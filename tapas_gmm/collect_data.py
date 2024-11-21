import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.env import Environment, import_env
from tapas_gmm.env.environment import BaseEnvironmentConfig
from tapas_gmm.policy import PolicyEnum, import_policy
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.utils.misc import (
    DataNamingConfig,
    get_dataset_name,
    get_full_task_name,
    loop_sleep,
)

# from tapas_gmm.utils.random import configure_seeds


@dataclass
class Config:
    n_episodes: int
    sequence_len: int | None

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    env_type: Environment
    env_config: BaseEnvironmentConfig

    policy_type: PolicyEnum
    policy: Any

    pretraining_data: bool = MISSING

    horizon: int | None = 300  # None


def main(config: Config) -> None:
    Env = import_env(config.env_config)

    Policy = import_policy(config.policy.value)

    task_name = get_full_task_name(config)  # type: ignore

    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / task_name

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    env = Env(config.env_config)  # type: ignore

    keyboard_obs = KeyboardObserver()

    replay_memory = SceneDataset(
        allow_creation=True,
        config=config.dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    env.reset()  # extra reset to correct set up of camera poses in first obs

    policy = Policy(config.policy, env=env, keyboard_obs=keyboard_obs)

    obs = env.reset()

    time.sleep(5)

    logger.info("Go!")

    episodes_count = 0
    timesteps = 0

    horizon = config.horizon or np.inf

    try:
        with tqdm(total=config.n_episodes) as ebar:
            with tqdm(total=horizon) as tbar:
                tbar.set_description("Time steps")
                while episodes_count < config.n_episodes:
                    ebar.set_description("Running episode")
                    start_time = time.time()

                    action, info = policy.predict(obs)
                    next_obs, _, done, _ = env.step(action)
                    obs.action = torch.Tensor(action)
                    obs.feedback = torch.Tensor([1])
                    replay_memory.add_observation(obs)
                    obs = next_obs

                    timesteps += 1
                    tbar.update(1)

                    if done or keyboard_obs.success:
                        # logger.info("Saving trajectory.")
                        ebar.set_description("Saving trajectory")
                        replay_memory.save_current_traj()

                        obs = env.reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        episodes_count += 1
                        ebar.update(1)

                        timesteps = 0
                        tbar.reset()

                        done = False

                    elif keyboard_obs.reset_button or timesteps >= horizon:
                        # logger.info("Resetting without saving traj.")
                        ebar.set_description("Resetting without saving traj")
                        replay_memory.reset_current_traj()

                        obs = env.reset()
                        keyboard_obs.reset()
                        policy.reset_episode(env)

                        timesteps = 0
                        tbar.reset()

                    else:
                        loop_sleep(start_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Attempting graceful env shutdown ...")
        env.close()


def complete_config(args: argparse.Namespace, config: DictConfig) -> DictConfig:
    config.pretraining_data = args.pretraining

    config.env_config.task = config.data_naming.task
    config.dataset_config.data_root = config.data_naming.data_root

    config.data_naming.feedback_type = get_dataset_name(config)

    return config


def entry_point():
    extra_args = (
        {
            "name": "--pretraining",
            "action": "store_true",
            "help": "Whether the data is for pretraining. Used to name the dataset.",
        },
    )
    args, dict_config = parse_and_build_config(data_load=False, extra_args=extra_args)
    dict_config = complete_config(args, dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    main(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
