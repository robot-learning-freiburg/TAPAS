import argparse
import pathlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode

import tapas_gmm.utils.logging  # noqa
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.env import Environment, import_env
from tapas_gmm.env.environment import (
    BaseEnvironmentConfig,
    RestoreActionMode,
    RestoreEnvState,
)

# from tapas_gmm.policy import PolicyEnum, import_policy
from tapas_gmm.utils.argparse import parse_and_build_config

# from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.utils.misc import (  # loop_sleep,
    DataNamingConfig,
    get_dataset_name,
    get_full_task_name,
)

# from tqdm.auto import tqdm


# from tapas_gmm.utils.random import configure_seeds


@dataclass
class Config:
    n_episodes: int
    sequence_len: int | None

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    env_type: Environment
    env_config: BaseEnvironmentConfig

    # policy: PolicyEnum
    # policy_config: Any

    pretraining_data: bool = MISSING

    # horizon: int | None = 300  # None


def main(config: Config) -> None:
    Env = import_env(config.env_config)

    task_name = get_full_task_name(config)  # type: ignore

    assert config.data_naming.data_root is not None

    save_path = pathlib.Path(config.data_naming.data_root) / task_name

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    env = Env(config.env_config)  # type: ignore

    replay_memory = SceneDataset(
        allow_creation=True,
        config=config.dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    # env.reset()  # extra reset to correct set up of camera poses in first obs

    # obs = env.reset()

    # time.sleep(5)

    # logger.info("Go!")

    def step_call(obs):
        return
        # pbar.update(1)
        # env.set_camera_pose(env.camera_pose)

    try:
        for demo_id in range(config.n_episodes):
            # with RestoreActionMode(env), RestoreEnvState(env):
            #     env.set_world_action_mode()
            traj = env.task_env.get_demos(
                1, live_demos=True, callable_each_step=step_call
            )[0]

            for i, raw_obs in enumerate(traj):
                obs = env.process_observation(raw_obs)

                next_obs = traj[i + 1] if i + 1 < len(traj) else raw_obs
                obs.action = torch.Tensor(env._get_action(raw_obs, next_obs))
                obs.feedback = torch.Tensor([1])

                replay_memory.add_observation(obs)

            replay_memory.save_current_traj()

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
