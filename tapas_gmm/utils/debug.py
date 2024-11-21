import functools
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from loguru import logger


def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    elif isinstance(output[0], (tuple, list)):
        outputs = (j for i in output for j in i)
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            print("In tuple length", len(inp))
            print("Due to ", summarize_tensor(inp[0], "input to this layer"))
            print(f"Found NAN in output {i} at indices: ", nan_mask.nonzero())
            # , "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
            breakpoint()


def summarize_tensor(tensor, name):
    return {
        "name": name,
        "shape": tensor.shape,
        "mean": tensor.mean(),
        "var": tensor.var(),
        "max": tensor.max(),
        "min": tensor.min(),
    }


# @logger.catch
def summarize_list_of_tensors(tensor_list, keep_last_dim=False):
    shapes = [list(t.shape) for t in tensor_list]
    info = summarize_by_dim(torch.stack(tensor_list), keep_last_dim=keep_last_dim)
    info = {k: list(v.numpy()) if keep_last_dim else v.item() for k, v in info.items()}
    info.update({"shapes": shapes})
    return info


# @logger.catch
def summarize_by_dim(tensor, keep_last_dim):
    tensor = tensor.flatten(end_dim=-2 if keep_last_dim else -1)
    return {
        "mean": tensor.mean(dim=0),
        "var": tensor.var(dim=0),
        "max": tensor.max(dim=0)[0],
        "min": tensor.min(dim=0)[0],
    }


def summarize_dataclass(obj):
    for field in obj.__dataclass_fields__:
        value = getattr(obj, field)
        if torch.is_tensor(value):
            print("{}: {}".format(field, value.shape))
        else:
            print("{}: {}".format(field, value))


def summarize_class(obj):
    for field in dir(obj):
        if not field.startswith("__"):
            value = getattr(obj, field)
            if torch.is_tensor(value):
                print("{}: {}".format(field, value.shape))
            else:
                print("{}: {}".format(field, value))


def measure_runtime(func: Callable):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        with logger.contextualize(filter=False):
            delta = end - start
            logger.info(f"Runtime: {delta} H:M:S.ms")
        return result

    return wrapper


def save_q_traj_dbg(poses, path):
    # save as human readable csv
    output_size = np.prod(poses.shape)
    np.set_printoptions(threshold=output_size)

    with open(f"{path}.txt", "w") as outfile:
        outfile.write(repr(poses))

    np.set_printoptions(threshold=1000)

    # save for np.load
    np.save(f"{path}.npy", poses)
