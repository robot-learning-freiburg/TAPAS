from typing import Sequence

import torch

from tapas_gmm.utils.geometry_torch import identity_quaternions


def get_keypoint_distance(set1: torch.Tensor, set2: torch.Tensor) -> torch.Tensor:
    # pairwise distance needs both inputs to have shape (N, D), so flatten
    B, N_kp, d_kp = set1.shape
    set1 = set1.reshape((B * N_kp, d_kp))
    set2 = set2.reshape((B * N_kp, d_kp))

    distance = torch.nn.functional.pairwise_distance(set1, set2)
    distance = distance.reshape((B, N_kp))

    return distance


def unflatten_keypoints(kp: torch.Tensor, kp_dim: int = 3) -> torch.Tensor:
    # unflatten from stacked x, y, (z) to n_kp, d_kp
    return torch.stack(torch.chunk(kp, kp_dim, dim=-1), dim=-1)


def poses_from_keypoints(kp: torch.Tensor) -> torch.Tensor:
    id_quat = identity_quaternions(kp.shape[:-1])
    return torch.cat((kp, id_quat), dim=-1)


def tp_from_keypoints(
    kp: torch.Tensor, indeces: Sequence[int] | None
) -> list[torch.Tensor]:
    unflattened_kp = unflatten_keypoints(kp)

    if indeces is None:
        selected_kp = unflattened_kp
    else:
        selected_kp = unflattened_kp[..., indeces, :]

    poses = poses_from_keypoints(selected_kp)

    assert len(poses.shape) == 3

    return [p for p in poses.swapdims(0, 1)]
