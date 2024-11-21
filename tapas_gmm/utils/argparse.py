import argparse

from omegaconf import DictConfig, OmegaConf

from tapas_gmm.utils.misc import import_config_file


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("as_tuple", resolve_tuple)


def parse_args(
    data_load: bool, need_task: bool, extra_args: tuple = tuple()
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to config file to use.",
    )

    if need_task:
        parser.add_argument(
            "-t",
            "--task",
            default="PhoneOnBase",
            help="Name of the task to train on.",
        )

    if data_load:
        parser.add_argument(
            "-f",
            "--feedback_type",
            default="pretrain_manual",
            help="The training data type. Cloning, dcm, ...",
        )
        parser.add_argument(
            "--path",
            default=None,
            help="Path to a dataset. May be provided instead of f-t.",
        )

    for arg in extra_args:
        name = arg.pop("name")
        parser.add_argument(name, **arg)

    parser.add_argument(
        "-o",
        "--overwrite",
        nargs="+",
        default=[],
        help="Overwrite config values. Format: key=value. Keys need to be "
        "fully qualified. E.g. "
        "'training.steps=1000 observation.cameras=overhead'",
    )

    args = parser.parse_args()

    return args


def build_config(
    data_load: bool, need_task: bool, args: argparse.Namespace
) -> DictConfig:
    conf_file = import_config_file(args.config)
    config = conf_file.config
    config = OmegaConf.structured(config)

    if need_task:
        config.data_naming.task = args.task
    if data_load:
        config.data_naming.feedback_type = args.feedback_type
        config.data_naming.path = args.path

    overwrites = OmegaConf.from_dotlist(args.overwrite)
    config = OmegaConf.merge(config, overwrites)

    assert type(config) == DictConfig

    return config


def parse_and_build_config(
    data_load: bool = True, need_task: bool = True, extra_args: tuple = tuple()
) -> tuple[argparse.Namespace, DictConfig]:
    args = parse_args(data_load, need_task, extra_args)
    config = build_config(data_load, need_task, args)

    return args, config
