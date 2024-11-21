import math
from copy import deepcopy
from functools import wraps

import torch
from loguru import logger

from tapas_gmm.utils.typing import NDArrayOrNDArraySeq, TensorOrTensorSeq


def list_or_tensor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], list):
            return [func(x, *args[1:], **kwargs) for x in args[0]]
        elif isinstance(args[0], tuple):
            return tuple(func(x, *args[1:], **kwargs) for x in args[0])
        else:
            return func(*args, **kwargs)

    return wrapper


def list_or_tensor_mult_return(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if isinstance(args[0], list):
            rets = [func(x, *args[1:], **kwargs) for x in args[0]]
            return [tuple(x) for x in zip(*rets)]
        elif isinstance(args[0], tuple):
            rets = tuple(func(x, *args[1:], **kwargs) for x in args[0])
            return tuple((tuple(x) for x in zip(*rets)))
        else:
            return func(*args, **kwargs)

    return wrapper


def list_or_tensor_mult_args(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if any(isinstance(args[i], list) for i in range(len(args))):
            return [func(*x, **kwargs) for x in zip(*args)]
        elif any(isinstance(args[i], tuple) for i in range(len(args))):
            return tuple(func(*x, **kwargs) for x in zip(*args))
        else:
            return func(*args, **kwargs)

    return wrapper


@list_or_tensor_mult_args
def cat(*tensors, dim=0):
    return torch.cat(tensors, dim=dim)


@list_or_tensor
def unsqueeze(tensor, dim):
    return torch.unsqueeze(tensor, dim=dim)


@list_or_tensor_mult_args
def stack(*tensors, dim=0):
    return torch.stack(tensors, dim=dim)


def eye_like(batched_homtrans):
    batched_eye = torch.zeros_like(batched_homtrans)
    batched_eye[:, range(4), range(4)] = 1

    return batched_eye


def heatmap_from_pos(center, var, size=(32, 32), use_gaussian=False):
    B, N = center.shape[0:2]
    x_cord = torch.arange(size[0], device=center.device)
    x_grid = x_cord.repeat(size[1]).view(*size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    xy_grid = xy_grid.unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1, 1)
    center = center.unsqueeze(2).unsqueeze(2)
    # TODO: map x,y components separately for non-square cams
    center = (center + 1) * size[0] / 2  # map from [-1, 1] to pixel coordinates
    var = var.unsqueeze(2)

    if use_gaussian:
        density = (1.0 / (2.0 * math.pi * var)) * torch.exp(
            -torch.sum((xy_grid - center) ** 2.0, dim=-1) / (2 * var)
        )
    else:
        density = torch.where(
            torch.sum((xy_grid - center) ** 2.0, dim=-1) > var, 0.0, 1.0
        )

    # normalize to [0, 1]
    density /= torch.amax(density, dim=(2, 3)).unsqueeze(2).unsqueeze(2)

    return density


def compare_state_dicts(state_a, state_b):
    identical = True
    for (ka, va), (kb, vb) in zip(state_a.items(), state_b.items()):
        if ka != kb:
            identical = False
            logger.warning("Non-matching keys {}, {}", ka, kb)
        elif not torch.equal(va, vb):
            identical = False
            logger.warning("Non-matching vals for key {}", ka)
    if identical:
        logger.info("Models match!")

    return identical


def stack_trajs(list_of_trajs):
    stacked = deepcopy(list_of_trajs[0])

    for field in dir(list_of_trajs[0]):
        if not field.startswith("__"):
            values = [getattr(t, field) for t in list_of_trajs]
            if torch.is_tensor(values[0]):
                stacked_vals = torch.cat(values)
                setattr(stacked, field, stacked_vals)
            elif values[0] is None:
                pass
            else:
                raise TypeError(
                    "Encountered unexpected type {} in stacking".format(type(values[0]))
                )

    return stacked


def flipped_logistic(x, mid_point=0, growth_rate=1):
    return 1 - 1 / (1 + torch.exp(-growth_rate * (x - mid_point)))


def gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + torch.erf((x - mu) / (sigma * math.sqrt(2))))


def to_numpy(x: TensorOrTensorSeq) -> NDArrayOrNDArraySeq:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return [to_numpy(y) for y in x]
    elif isinstance(x, tuple):
        return tuple(to_numpy(y) for y in x)
    else:
        raise TypeError("Unknown type {}".format(type(x)))


def batched_block_diag(*arrs):
    # Adapted from https://github.com/pytorch/pytorch/issues/31932#issuecomment-585374009
    shapes = torch.tensor([a.shape[-2:] for a in arrs])
    shape_sum = torch.sum(shapes, dim=0).tolist()
    batch_shape = list(arrs[0].shape[:-2])
    out = torch.zeros(
        batch_shape + shape_sum, dtype=arrs[0].dtype, device=arrs[0].device
    )
    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[..., r : r + rr, c : c + cc] = arrs[i]
        r += rr
        c += cc
    return out


@list_or_tensor
def slice_any_tensor_dim(
    tens: torch.Tensor, dim: int, start: int, stop: int
) -> torch.Tensor:
    start = start if start >= 0 else tens.shape[dim] + start
    stop = stop if stop >= 0 else tens.shape[dim] + stop + 1
    length = stop - start
    return tens.narrow(dim, start, length)


@list_or_tensor
def single_index_any_tensor_dim(tens: torch.Tensor, idx: int, dim: int):
    return tens.select(dim, idx)
