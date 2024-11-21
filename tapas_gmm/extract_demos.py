import argparse
import multiprocessing as mp
import os
import pathlib
import re
from copy import deepcopy
from dataclasses import dataclass

import gymnasium as gym
import h5py
import numpy as np
import sapien.core as sapien
from loguru import logger
from mani_skill2.utils.io_utils import load_json
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode
from tqdm.auto import tqdm

import tapas_gmm.utils.logging  # noqa

# from config import camera_pose
from tapas_gmm.dataset.scene import SceneDataset, SceneDatasetConfig
from tapas_gmm.env.mani_skill import (
    ACTION_MODE,
    OBS_MODE,
    ManiSkillEnv,
    ManiSkillEnvironmentConfig,
)
from tapas_gmm.utils.argparse import parse_and_build_config
from tapas_gmm.utils.keyboard_observer import KeyboardObserver
from tapas_gmm.utils.maniskill_replay import from_pd_joint_delta_pos, from_pd_joint_pos
from tapas_gmm.utils.misc import DataNamingConfig


@dataclass
class Config:
    num_demos: int
    sequence_len: int | None

    data_naming: DataNamingConfig
    dataset_config: SceneDatasetConfig

    vis: bool
    verbose: bool

    env_config: ManiSkillEnvironmentConfig

    max_retry: int = 0

    use_env_states: bool = False
    allow_failure: bool = False
    discard_timeout: bool = False

    target_control_mode: str = ACTION_MODE
    obs_mode: str = OBS_MODE

    traj_path: str | None = None


def _main(config: Config, save_path: pathlib.Path) -> None:
    # Load HDF5 containing trajectories
    traj_path = config.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    if "fixed_target_link_idx" in ori_env_kwargs:
        logger.warning("TODO: Double check handling of fixed_target_link_idx!!")
        config.env_config.fixed_target_link_idx = ori_env_kwargs[
            "fixed_target_link_idx"
        ]

    # Create a twin env with the original kwargs
    if config.target_control_mode is not None:
        ori_env = gym.make(env_id, **ori_env_kwargs)
    else:
        ori_env = None

    # Create a main env for replay
    env_kwargs = ori_env_kwargs.copy()
    # if assert succeeds, no need to update any kwargs from tapas_gmm.env_kwargs
    assert set(env_kwargs.keys()).issubset(
        {
            "obs_mode",
            "control_mode",
            "reward_mode",
            "model_ids",
            "fixed_target_link_idx",
        }
    )

    env = ManiSkillEnv(config.env_config)

    camera_names = config.env_config.cameras

    dataset_config = SceneDatasetConfig(
        camera_names=camera_names,
        subsample_by_difference=False,
        subsample_to_length=config.sequence_len,
        data_root=config.dataset_config.data_root,
        image_size=config.env_config.image_size,
    )

    replay_memory = SceneDataset(
        allow_creation=True,
        config=dataset_config,
        data_root=save_path / config.data_naming.feedback_type,
    )

    episodes = json_data["episodes"]
    n_ep = len(episodes)
    if config.num_demos is not None:
        n_ep = min(n_ep, int(config.num_demos))
    inds = np.arange(n_ep)

    pbar = tqdm(leave=None, unit="step", dynamic_ncols=True)
    tbar = tqdm(total=n_ep, unit="traj", desc="Converting demos")

    if pbar is not None:
        # pbar.set_postfix(
        #     {
        #         "control_mode": env_kwargs.get("control_mode"),
        #         "obs_mode": env_kwargs.get("obs_mode"),
        #     }
        # )
        pbar.set_postfix(
            {
                "control_mode": config.env_config.action_mode,
                "obs_mode": config.env_config.obs_mode,
            }
        )

    # Prepare for recording
    # output_dir = os.path.dirname(traj_path)
    # ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    # new_traj_name = ori_traj_name + "." + suffix
    # if num_procs > 1:
    #     new_traj_name = new_traj_name + "." + str(proc_id)
    # env = RecordEpisode(
    #     env,
    #     output_dir,
    #     save_on_reset=False,
    #     save_trajectory=args.save_traj,
    #     trajectory_name=new_traj_name,
    #     save_video=args.save_video,
    # )

    # if env.save_trajectory:
    #     output_h5_path = env._h5_file.filename
    #     assert not os.path.samefile(output_h5_path, traj_path)
    # else:
    #     output_h5_path = None

    # Replay
    for ind in inds:
        ep = episodes[ind]
        episode_id = ep["episode_id"]
        traj_id = f"traj_{episode_id}"
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")

        if traj_id not in ori_h5_file:
            tqdm.write(f"{traj_id} does not exist in {traj_path}")
            continue

        reset_kwargs = ep["reset_kwargs"].copy()
        if "seed" in reset_kwargs:
            assert reset_kwargs["seed"] == ep["episode_seed"]
        else:
            reset_kwargs["seed"] = ep["episode_seed"]
        seed = reset_kwargs.pop("seed")

        ori_control_mode = ep["control_mode"]

        for _ in range(config.max_retry + 1):
            env.reset(seed=seed, options=reset_kwargs)
            if ori_env is not None:
                ori_env.reset(seed=seed, options=reset_kwargs)

            if config.vis:
                env.render()

            # Original actions to replay
            ori_actions = ori_h5_file[traj_id]["actions"][:]

            # Original env states to replay
            if config.use_env_states:
                ori_env_states = ori_h5_file[traj_id]["env_states"][1:]

            info = {}

            # Without conversion between control modes
            if config.target_control_mode is None:
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    obs, reward, done, info = env.step(a)
                    obs.action = a
                    obs.feedback = [1]
                    replay_memory.add_observation(obs)
                    if config.vis:
                        env.render()
                    if config.use_env_states:
                        env.set_state(ori_env_states[t])

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = from_pd_joint_pos(
                    config.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=config.vis,
                    pbar=pbar,
                    verbose=config.verbose,
                    replay_memory=replay_memory,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = from_pd_joint_delta_pos(
                    config.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=config.vis,
                    pbar=pbar,
                    verbose=config.verbose,
                    replay_memory=replay_memory,
                )

            success = info.get("success", False)
            if config.discard_timeout:
                timeout = "TimeLimit.truncated" in info
                success = success and (not timeout)

            if success or config.allow_failure:
                replay_memory.save_current_traj()
                break
            else:
                replay_memory.reset_current_traj()
                # Rollback episode id for failed attempts
                # env._episode_id -= 1
                if config.verbose:
                    print("info", info)
        else:
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

        tbar.update()

    # Cleanup
    env.close()
    if ori_env is not None:
        ori_env.close()

    if pbar is not None:
        pbar.close()


def complete_config(
    args: argparse.Namespace, config: DictConfig
) -> tuple[DictConfig, str]:
    task_path = pathlib.Path(args.traj_path)
    task_name = task_path.parent.stem

    if task_path.suffix != ".h5":
        raise ValueError("Trajectory file should be in HDF5 format")

    variant_specs = []

    if not task_path.stem == "trajectory":
        variant_specs.append(task_path.stem)

    while not re.match("^.*v[0-9]$", task_name):
        variant_specs.append(task_name)
        task_path = task_path.parent
        task_name = task_path.parent.stem

    variant_name = "-".join(reversed(variant_specs))

    if variant_name != "":
        variant_name = "-" + variant_name

    save_name = task_name + variant_name

    config.data_naming.task = save_name

    config.env_config.task = task_name
    config.dataset_config.data_root = config.data_naming.data_root

    config.traj_path = args.traj_path

    return config, save_name


def main():
    extra_args = (
        {
            "name": "--traj_path",
            "required": True,
        },
    )
    args, dict_config = parse_and_build_config(
        data_load=False, need_task=False, extra_args=extra_args
    )
    dict_config, save_name = complete_config(args, dict_config)

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )

    # TODO: make obs mode confable? Ie allow to skip rgb to save space

    save_path = pathlib.Path(config.dataset_config.data_root) / save_name

    if not save_path.is_dir():
        logger.warning(
            "Creating save path. This should only be needed for " "new tasks."
        )
        save_path.mkdir(parents=True)

    _main(config, save_path)  # type: ignore


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
