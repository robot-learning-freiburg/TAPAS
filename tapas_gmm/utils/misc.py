import collections.abc
import importlib
import os
import pathlib
import random
import string
import sys
import time
from dataclasses import dataclass
from functools import lru_cache, reduce
from typing import Optional

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from tapas_gmm.utils.logging import indent_logs


@dataclass
class DataNamingConfig:
    feedback_type: str | None
    task: str | None
    data_root: str | None

    path: Optional[str | None] = None


def invert_dict(dictionary):
    return {v: k for k, v in dictionary.items()}


def random_string(str_len=8):
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choices(alphabet, k=str_len))


def get_full_task_name(config: DictConfig):
    task_name = config.data_naming.task

    if hasattr(config, "env_config"):
        if (bg := config.env_config.background) is not None:
            task_name += "-" + bg

        if (mid := config.env_config.model_ids) is not None:
            task_name += "-" + "_".join(mid)

    return task_name


def get_dataset_name(config: DictConfig):
    if config.pretraining_data:
        name = "pretrain_" + config.policy.value.lower()
    else:
        name = "demos"

    return name


def load_scene_data(config: DataNamingConfig):
    import tapas_gmm.dataset.scene as scene_dataset

    logger.info("Loading dataset(s): ")

    def get_data_path(config: DataNamingConfig, task: str, root: str) -> str:
        file_name = config.feedback_type

        return root + "/" + task + "/" + file_name

    with indent_logs():
        if (path := config.path) is not None:
            memory = scene_dataset.SceneDataset(data_root=pathlib.Path(path))
        else:
            task = config.task
            if task == "Mixed":
                # task = task_switch.keys()
                raise NotImplementedError(
                    "Disabled to avoid QT mixup between CoppeliaSim and ROS."
                )
            else:
                task = [task]

            data = []

            for t in task:
                data_path = pathlib.Path(
                    get_data_path(config, t, root=config.data_root)
                )
                dset = scene_dataset.SceneDataset(data_root=data_path)
                data.append(dset)

            memory = scene_dataset.SceneDataset.join_datasets(*data)

        logger.info("Done! Data contains {} trajectories.", len(memory))

    return memory


def policy_checkpoint_name(
    config: DictConfig, create_suffix: bool = False
) -> tuple[pathlib.Path, str]:
    if create_suffix and config.policy.suffix:
        raise ValueError("Should not pass suffix AND ask to create one.")
    if v := config.policy.suffix:
        suffix = "-" + v
    elif create_suffix:
        suffix = "-" + random_string()
    else:
        suffix = ""

    policy_type = str(config.policy_type)

    if hasattr(config.policy, "encoder_name"):
        encoder_name = "_" + str(config.policy.encoder_name)
    else:
        encoder_name = ""

    if (
        hasattr(config.policy, "encoder_naming")
        and config.policy.encoder_naming is not None
    ):
        pretrained_on = "_pon-" + config.policy.encoder_naming.task
    else:
        pretrained_on = ""

    task_name = get_full_task_name(config)

    return (
        pathlib.Path(config.data_naming.data_root)
        / task_name
        / (
            config.data_naming.feedback_type
            + encoder_name
            + pretrained_on
            + "_"
            + policy_type
            + "_policy"
            + suffix
        )
    ).with_suffix(".pt"), suffix


def pretrain_checkpoint_name(config) -> pathlib.Path:
    # print("==========")
    # print(config)
    naming_config: DataNamingConfig = config.encoder_naming
    feedback = naming_config.feedback_type
    task = naming_config.task
    suffix = config.encoder_suffix
    suffix = "-" + suffix if suffix else ""
    encoder_name = feedback + "_" + config.encoder_name + "_encoder" + suffix
    return (pathlib.Path(naming_config.data_root) / task / encoder_name).with_suffix(
        ".pt"
    )


def even(int_number):
    return int_number // 2 * 2


def correct_action(keyboard_obs, action):
    if keyboard_obs.has_joints_cor():
        ee_step = keyboard_obs.get_ee_action()
        action[:-1] = action[:-1] * 0.5 + ee_step
        action = np.clip(action, -0.9, 0.9)
    if keyboard_obs.has_gripper_update():
        action[-1] = keyboard_obs.get_gripper()
    return action


def loop_sleep(start_time, dt=0.05):
    sleep_time = dt - (time.time() - start_time)
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def import_config_file(config_file):
    """
    Import a config file as a python module.

    Parameters
    ----------
    config_file : str or path
        Path to the config file.

    Returns
    -------
    module
        Python module containing the config file.

    """
    config_file = str(config_file)
    if config_file[-3:] == ".py":
        config_file = config_file[:-3]

    config_file_path = os.path.abspath("/".join(config_file.split("/")[:-1]))
    sys.path.insert(1, config_file_path)
    config = importlib.import_module(config_file.split("/")[-1], package=None)

    return config


def get_variables_from_module(module):
    """
    Get (non-magic, non-private) variables defined in a module.

    Parameters
    ----------
    module : module
        The python module from which to extract the variables.

    Returns
    -------
    dict
        Dict of the vars and their values.
    """
    confs = {
        k: v
        for k, v in module.__dict__.items()
        if not (k.startswith("__") or k.startswith("_"))
    }
    return confs


def recursive_dict_update(base, update):
    for k, v in update.items():
        if isinstance(v, collections.abc.Mapping):
            base[k] = recursive_dict_update(base.get(k, {}), v)
        else:
            base[k] = v
    return base


def apply_machine_config(config, machine_config_path="src/_machine_config.py"):
    path = pathlib.Path(machine_config_path)
    if path.is_file():
        mc = import_config_file(machine_config_path).config
        logger.info("Applying machine config to config dict.")
        config = recursive_dict_update(config, mc)
    else:
        logger.info("No machine config found at {}.".format(machine_config_path))
    return config


def configure_class_instance(instance, class_keys, config):
    for k in class_keys:
        v = config.get(k, None)
        if v is None:
            logger.warning("Key {} not in encoder config. Assuming False.", k)
            v = False
        setattr(instance, k, v)


def get_and_log_failure(dictionary, key, default=None):
    if key in dictionary:
        return dictionary[key]
    else:
        logger.info("Key {} not in config. Assuming {}.", key, default)
        return default


def multiply_iterable(l):
    return reduce(lambda x, y: x * y, l)
