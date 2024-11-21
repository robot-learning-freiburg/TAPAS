import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any  # , NewType

import jsonpickle
from omegaconf import DictConfig, OmegaConf

# Sentinel value to indicate that a value is to be set programmatically.
# Can't implement sentinal value as a class because of OmegaConf, which does
# not allow custom classes in type hints. Hence, using enum instead.
# SET_PROGRAMMATICALLY = NewType("_SetProgramaticallySentinel", object)


class _SENTINELS(Enum):
    SET_PROGRAMMATICALLY = "?!?"
    COPY_FROM_MAIN_FITTING = "copy_from_main_fitting"


SET_PROGRAMMATICALLY = _SENTINELS.SET_PROGRAMMATICALLY
COPY_FROM_MAIN_FITTING = _SENTINELS.COPY_FROM_MAIN_FITTING


def value_not_set(val: Any) -> bool:
    return val is None or val == SET_PROGRAMMATICALLY


def dict_to_disk(filename: str | Path, data_dict: dict) -> None:
    with open(filename, "w") as f:
        f.write(jsonpickle.encode(data_dict))  # type: ignore


def dict_from_disk(filename: str | Path) -> dict:
    with open(filename) as f:
        data_dict = jsonpickle.decode(f.read())
    assert type(data_dict) is dict

    return data_dict


def yaml_to_disk(filename: str | Path, yaml_config: str) -> None:
    with open(filename, "w") as f:
        f.write(yaml_config)


def structured_config_to_dict(config: Any) -> dict:
    dict_conf: DictConfig = OmegaConf.structured(config)
    conf_dict: dict = OmegaConf.to_container(dict_conf, resolve=True)  # type: ignore

    return conf_dict


def structured_config_to_yaml(config: Any) -> str:
    return OmegaConf.to_yaml(config, resolve=True)


def save_config_along_path(config: Any, path: Path) -> None:
    json_path = path.with_suffix(".json")
    yaml_path = path.with_suffix(".yaml")

    dict_conf = structured_config_to_dict(config)
    yaml_conf = structured_config_to_yaml(config)

    dict_to_disk(json_path, dict_conf)
    yaml_to_disk(yaml_path, yaml_conf)


# Adapted from: https://stackoverflow.com/a/53818532
def recursive_compare_dict(d1: dict, d2: dict, level: str = "root") -> str:
    result = ""
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            result += "{:<20} + {} - {}\n".format(level, s1 - s2, s2 - s1)
            common_keys = s1 & s2
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            result += recursive_compare_dict(
                d1[k], d2[k], level="{}.{}".format(level, k)
            )

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            result += "{:<20} len1={}; len2={}\n".format(level, len(d1), len(d2))
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            result += recursive_compare_dict(
                d1[i], d2[i], level="{}[{}]".format(level, i)
            )

    else:
        if d1 != d2:
            result += "{:<20} {} != {}\n".format(level, d1, d2)

    return result


def recursive_compare_dataclass(d1: Any, d2: Any) -> str:
    if not (dataclasses.is_dataclass(d1) and dataclasses.is_dataclass(d2)):
        raise ValueError("Both parameters must be dataclass instances.")

    if type(d1) != type(d2):
        return "Mismatched types: {} != {}".format(type(d1), type(d2))
    else:
        return recursive_compare_dict(dataclasses.asdict(d1), dataclasses.asdict(d2))
