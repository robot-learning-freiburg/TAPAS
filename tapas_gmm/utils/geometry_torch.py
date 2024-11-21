import math
from typing import Tuple

import torch
import torch.nn.functional as F
from loguru import logger

from tapas_gmm.utils.torch import (
    list_or_tensor,
    list_or_tensor_mult_return,
    single_index_any_tensor_dim,
    slice_any_tensor_dim,
)

# NOTE: reusing some functions from https://github.com/facebookresearch/pytorch3d

identity_quaternion = torch.tensor([1, 0, 0, 0], dtype=torch.float32)

y180_quaternion = torch.tensor([0, 0, 1, 0], dtype=torch.float32)

identity_7_pose = torch.cat((torch.zeros(3), identity_quaternion), dim=-1)

quarter_rot_angle = torch.tensor(math.pi / 2, dtype=torch.float32)


def identity_quaternions(shape):
    """
    Create a batch of identity quaternions.

    Parameters
    ----------
    shape : iterable[int]
        The batch shape.

    Returns
    -------
    torch.Tensor
        Batch of identity quaternions.
        Shape: <shape>, 4
    """
    shape = list(shape) + [1]
    q = identity_quaternion.repeat(shape)

    return q


@list_or_tensor
def conjugate_quat(quaternion: torch.Tensor) -> torch.Tensor:
    assert quaternion.shape[-1] == 4
    conj = torch.clone(quaternion)
    # Both lines are equivalent, though the second one preserves the sign of
    # the real part.
    # conj[..., 0] = - conj[..., 0]
    conj[..., 1:4] = -conj[..., 1:4]
    return conj


def quaternion_multiply(
    quaternion0: torch.Tensor, quaternion1: torch.Tensor
) -> torch.Tensor:
    """
    Aka Hamilton product.
    """
    w0, x0, y0, z0 = torch.unbind(quaternion0, -1)
    w1, x1, y1, z1 = torch.unbind(quaternion1, -1)

    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1

    return torch.stack((w, x, y, z), -1)


def quaternion_diff(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError(
        "THIS FUNCTION IS LIKELY WRONG! SHOULD BE quaternion_multiply(conjugate_quat(q1), q2), where q1 is the current pose and q2 is the next pose."
    )
    return quaternion_multiply(q1, conjugate_quat(q2))


def quaternion_pose_diff(current: torch.Tensor, next: torch.Tensor) -> torch.Tensor:
    return quaternion_multiply(conjugate_quat(current), next)


def compute_angle_between_quaternions(q: torch.Tensor, r: torch.Tensor) -> float:
    """
    Computes the angles between two batches of quaternions.

    theta = arccos(2 * <q1, q2>^2 - 1)

    See https://math.stackexchange.com/questions/90081/quaternion-distance
    :param q: numpy array in form [w,x,y,z]. As long as both q,r are consistent
              it doesn't matter
    :type q:
    :param r:
    :type r:
    :return: angle between the quaternions, in radians
    :rtype:
    """
    ac = 2 * (q * r).sum(-1) ** 2 - 1  # prevent NaNs by clamping
    ac = torch.clamp(ac, -1, 1)
    assert not torch.isnan(ac).any()
    theta = torch.acos(ac)
    return theta


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def homogenous_transform_from_rot_shift(
    rot: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    n_dim = rot.shape[-1]
    if len(rot.shape) == 3:
        B = rot.shape[0]
        matrix = torch.zeros((B, n_dim + 1, n_dim + 1))
        matrix[:, :n_dim, :n_dim] = rot
        matrix[:, :n_dim, n_dim] = shift
        matrix[:, n_dim, n_dim] = 1
    else:
        matrix = torch.zeros((n_dim + 1, n_dim + 1))
        matrix[:n_dim, :n_dim] = rot
        matrix[:n_dim, n_dim] = shift
        matrix[n_dim, n_dim] = 1

    return matrix.to(rot.device)


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.
    Taken from github.com/facebookresearch/pytorch3d
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # type: ignore
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(
    matrix: torch.Tensor, prefer_positives: bool = False, prefer_continuous: bool = True
) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    quat = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    if prefer_positives:
        quat = torch.where((quat[..., 0] >= 0).unsqueeze(-1), quat, -quat)

    if prefer_continuous:
        preference_indicator = torch.Tensor([0.5, 0.5, 0.5, 0.5]).unsqueeze(0)
        # [0.9238795, 0.2209424, 0.2209424, 0.2209424]

        # preference_indicator = torch.Tensor(
        #     [0.5, 0.5, 0.5]).unsqueeze(0)

        quat_shape = quat.shape

        dist = torch.mm(quat.reshape(-1, 4), preference_indicator.T).reshape(
            quat_shape[:-1]
        )
        # dist = torch.mm(quat[..., 1:].reshape(-1, 3), preference_indicator.T).reshape(
        #     quat_shape[:-1])

        quat = torch.where(dist.unsqueeze(-1) >= -1e-2, quat, -quat)

    # import matplotlib.pyplot as plt
    # n_trajs = quat_shape[0]
    # if len(quat_shape) == 3:
    #     fig, ax = plt.subplots(n_trajs, subplot_kw=dict(projection='3d'))
    #     fig.set_size_inches(4, n_trajs*4)
    #     for i in range(n_trajs):
    #         ax[i].plot(quat[i, ..., 1], quat[i, ..., 2], quat[i, ..., 3])
    # elif len(quat_shape) == 4:
    #     n_frames = quat_shape[2]
    #     fig, ax = plt.subplots(n_trajs, n_frames,
    #         subplot_kw=dict(projection='3d'))
    #     fig.set_size_inches(4*n_frames, n_trajs*4)
    #     for i in range(n_trajs):
    #         for j in range(n_frames):
    #             ax[i, j].scatter(
    #                 quat[i, :, j, 1], quat[i, :, j, 2], quat[i, :, j, 3])
    # plt.show()

    return quat


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)  # type: ignore
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


@list_or_tensor_mult_return
def quaternion_to_axis_and_angle(
    quaternions: torch.Tensor, prefer_small_rots: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    scaled_axis_angle = quaternion_to_axis_angle(quaternions)
    angle = torch.norm(scaled_axis_angle, p=2, dim=-1, keepdim=True)
    axis = scaled_axis_angle / angle

    # catch NaNs from division by zero
    # TODO: smooth, ie fill context-sensitively instead of just with constants
    # Fill with [0, 0, 1] for NaNs
    mask = torch.isnan(axis)
    axis = torch.where(mask, torch.tensor([0, 0, 1], device=axis.device), axis)

    if prefer_small_rots:
        # angle = torch.where(angle > math.pi, 2 * math.pi - angle, angle)

        # If rotation is larger than 90 degrees, flip the axis and subtract 180 degrees
        mask = angle > math.pi / 2
        axis = torch.where(mask, -axis, axis)
        angle = torch.where(mask, math.pi - angle, angle)

    angle = angle.squeeze(-1)

    return axis, angle


@list_or_tensor_mult_return
def translation_to_direction_and_magnitude(
    translations: torch.Tensor,
    prefer_positive_x_axis: bool = False,
    prefer_positive_magnitude: bool = False,
    smooth_over_zero_magnitude: bool = True,
    smooth_over_sign_change: bool = False,
    zero_threshold: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:

    magnitude = torch.norm(translations, p=2, dim=-1, keepdim=True)
    direction = translations / magnitude

    zero_mask = magnitude < zero_threshold

    time_dim = 1 if len(zero_mask.shape) == 3 else 0

    if smooth_over_zero_magnitude:
        # direction will be degenerated after division by (near-) zero magnitude, so
        # replace with value from last timestep with non-zero magnitude
        if not zero_mask.shape[time_dim] == max(zero_mask.shape):
            logger.warning(
                "Assuming that time is second dim in translation_to_direction_and_magnitude"
                f"However, tensor has shape {zero_mask.shape}. Is this correct?"
            )
        for t in range(1, zero_mask.shape[time_dim]):
            smoothed = torch.where(
                zero_mask.select(time_dim, t),
                direction.select(time_dim, t - 1),
                direction.select(time_dim, t),
            )
            # NOTE: inelegant, but works
            if time_dim == 1:
                direction[:, t] = smoothed
            else:
                direction[t] = smoothed

    if smooth_over_sign_change:
        if not translations.shape[time_dim] == max(translations.shape):
            logger.warning(
                "Assuming that time is second dim in translation_to_direction_and_magnitude"
                f"However, tensor has shape {translations.shape}. Is this correct?"
            )

        # Detect direction changes via dot product of consecutive steps
        sign_changes_at_t = (
            slice_any_tensor_dim(direction, time_dim, 0, -2)
            * slice_any_tensor_dim(direction, time_dim, 1, -1)
        ).sum(-1) < 0

        # Take previous sign change into account
        flip_mask = torch.cumsum(sign_changes_at_t, dim=time_dim) % 2 == 1

        leading_zero = torch.zeros_like(
            single_index_any_tensor_dim(flip_mask, 0, time_dim).unsqueeze(time_dim)
        )
        flip_mask = torch.concatenate([leading_zero, flip_mask], dim=time_dim)

        flip_mask = flip_mask.unsqueeze(-1)

        direction = torch.where(flip_mask, -direction, direction)
        magnitude = torch.where(flip_mask, -magnitude, magnitude)

        # TODO: also enforce consistency across trajs by comparing with first step of
        # first traj

    if prefer_positive_x_axis:
        mask = torch.less(direction[..., 0], 0).unsqueeze(-1)
        direction = torch.where(mask, -direction, direction)
        magnitude = torch.where(mask, -magnitude, magnitude)

    if prefer_positive_magnitude:
        # get average magnitude over time and flip sign of trajectory if it is negative
        avg_magnitude = magnitude.mean(dim=time_dim, keepdim=True)
        mask = avg_magnitude < 0
        direction = torch.where(mask, -direction, direction)
        magnitude = torch.where(mask, -magnitude, magnitude)

    magnitude = magnitude.squeeze(-1)

    return direction, magnitude


def quaternion_to_euler(
    quaternions: torch.Tensor, prefer_positives: bool = False, threshold: float = -3.0
) -> torch.Tensor:
    logger.warning(
        "Euler angles should only be used for human "
        "interfaces, such as plotting, and not for "
        "any computations. Use quaternions or matrices!"
    )
    w, x, y, z = torch.unbind(quaternions, -1)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    euler = torch.stack([roll_x, pitch_y, yaw_z], dim=-1)

    if prefer_positives:
        euler = torch.where(euler < threshold, euler + 2 * math.pi, euler)

    return euler


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(
    euler_angles: torch.Tensor, convention: str = "XYZ"
) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def invert_homogenous_transform(matrix: torch.Tensor) -> torch.Tensor:
    """
    Works with batched as well.
    """
    rot = matrix[..., :3, 0:3]
    shift = matrix[..., :3, 3:4]

    rot_inv = torch.transpose(rot, -1, -2)
    shift_inv = torch.matmul(-rot_inv, shift)

    inverse = torch.zeros_like(matrix)
    inverse[..., :3, 0:3] = rot_inv
    inverse[..., :3, -1:] = shift_inv
    inverse[..., 3, -1:] = 1

    return inverse


def invert_intrinsics(matrix):
    """
    Works with batched as well.
    """
    # rot = matrix[..., :2, 0:2]
    # shift = matrix[..., :2, 2:3]
    #
    # rot_inv = torch.transpose(rot, -1, -2)
    # shift_inv = torch.matmul(-rot_inv, shift)
    #
    # inverse = torch.zeros_like(matrix)
    # inverse[..., :2, 0:2] = rot_inv
    # inverse[..., :2, -1:] = shift_inv
    # inverse[..., 2, -1:] = 1
    #
    # return inverse
    return matrix.inverse()


def batched_rigid_transform(xyz: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Transform: (B,4,4), Pointcloud (N,3) -> (B,N,3)
    """
    xyz_h = torch.cat(
        (xyz, torch.ones((*xyz.shape[:-1], 1), device=xyz.device)), dim=-1
    )
    xyz_t_h = torch.transpose(transform @ torch.transpose(xyz_h, -1, -2), -1, -2)
    return xyz_t_h[..., :3]


def batchwise_rigid_transform(
    xyz: torch.Tensor, transform: torch.Tensor
) -> torch.Tensor:
    """
    Transform: (B,4,4), Pointcloud (B,3) -> (B,3)
    """
    xyz_h = torch.cat(
        (xyz, torch.ones((*xyz.shape[:-1], 1), device=xyz.device)), dim=-1
    )
    xyz_t_h = torch.bmm(transform, xyz_h.unsqueeze(-1)).squeeze(-1)
    return xyz_t_h[..., :3]


def cam2pix(
    cam_pts: torch.Tensor, cam_intr: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    pix = torch.transpose(cam_intr @ torch.transpose(cam_pts, -1, -2), -1, -2)

    z = pix[..., 2]

    pix = pix[..., :2] / pix[..., 2:3].repeat(1, 1, 2)

    return pix, z


def batchwise_cam2pix(
    cam_pts: torch.Tensor, cam_intr: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    pix = torch.bmm(cam_intr, cam_pts.unsqueeze(-1)).squeeze(-1)

    z = pix[..., 2]

    pix = pix[..., :2] / pix[..., 2:3].repeat(1, 2)
    pix = torch.round(pix)

    return pix, z


def batched_project_onto_cam(
    point_cloud, depth_im, cam_intr, cam_pose, clip=True, clip_value=0, get_depth=False
):
    cam_pts = batched_rigid_transform(
        point_cloud, invert_homogenous_transform(cam_pose)
    )
    pix_z = cam_pts[:, :, 2]

    pix, depth = cam2pix(cam_pts, cam_intr)

    pix_x, pix_y = pix[:, :, 0], pix[:, :, 1]

    if clip:
        B, im_h, im_w = depth_im.shape
        # Eliminate pixels outside view frustum. Ie map them to zero here.
        # TODO: instead clip to image border to allow easier navigation?
        valid_pix = torch.logical_and(
            pix_x >= 0,
            torch.logical_and(
                pix_x < im_w,
                torch.logical_and(
                    pix_y >= 0, torch.logical_and(pix_y < im_h, pix_z > 0)
                ),
            ),
        )

        final = torch.where(
            valid_pix.unsqueeze(-1).repeat(1, 1, 2),
            pix,
            torch.tensor(clip_value, dtype=torch.float32, device=pix.device),
        )
    else:
        final = pix

    if get_depth:
        return final, depth
    else:
        return final


def batchwise_project_onto_cam(
    point_cloud, depth_im, cam_intr, cam_pose, clip=True, clip_value=0
):
    cam_pts = batchwise_rigid_transform(
        point_cloud, invert_homogenous_transform(cam_pose)
    )
    pix_z = cam_pts[:, 2]

    pix, pix_depth = batchwise_cam2pix(cam_pts, cam_intr)
    pix_x, pix_y = pix[:, 0], pix[:, 1]

    if clip:
        B, im_h, im_w = depth_im.shape
        # Eliminate pixels outside view frustum. Ie map them to zero here.
        # TODO: instead clip to image border to allow easier navigation?
        valid_pix = torch.logical_and(
            pix_x >= 0,
            torch.logical_and(
                pix_x < im_w,
                torch.logical_and(
                    pix_y >= 0, torch.logical_and(pix_y < im_h, pix_z > 0)
                ),
            ),
        )

        final = torch.where(
            valid_pix.unsqueeze(-1).repeat(1, 2),
            pix,
            torch.tensor(clip_value, dtype=torch.float32, device=pix.device),
        )
    else:
        final = pix

    return final, pix_depth


def batched_pinhole_projection_image_to_camera_coordinates_orig(u, v, z, K):
    uv1 = torch.stack((u, v, torch.ones(u.shape, device=u.device)), dim=-1)
    K_inv = invert_intrinsics(K)

    pos = torch.transpose(torch.matmul(K_inv, torch.transpose(uv1, -1, -2)), -1, -2)

    pos = z.unsqueeze(2).repeat(1, 1, 3) * pos
    return pos


def batched_pinhole_projection_image_to_world_coordinates_orig(
    u, v, z, K, camera_to_world
):
    pos_in_camera_frame = batched_pinhole_projection_image_to_camera_coordinates_orig(
        u, v, z, K
    )
    pos_in_camera_frame_homog = torch.cat(
        (
            pos_in_camera_frame,
            torch.ones(
                (*pos_in_camera_frame.shape[:-1], 1), device=pos_in_camera_frame.device
            ),
        ),
        dim=-1,
    )

    pos_in_world_homog = torch.transpose(
        torch.matmul(
            camera_to_world, torch.transpose(pos_in_camera_frame_homog, -1, -2)
        ),
        -1,
        -2,
    )

    return pos_in_world_homog[..., :3]


def hard_pixels_to_3D_world(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W
    camera_to_world,  # N, 4, 4
    K,  # N, 3, 3
    img_width,
    img_height,
):
    u_normalized, v_normalized = y_vision.chunk(2, dim=-1)
    u_pixel = ((u_normalized / 2.0 + 0.5) * img_width).long().detach()
    v_pixel = ((v_normalized / 2.0 + 0.5) * img_height).long().detach()

    B, N_kp = v_pixel.shape
    batch_indeces = torch.arange(
        B, device=depth.device, dtype=torch.long
    ).repeat_interleave(N_kp)
    z = depth[batch_indeces, v_pixel.flatten(), u_pixel.flatten()]
    z = z.reshape(B, N_kp)

    pos = batched_pinhole_projection_image_to_world_coordinates_orig(
        u_pixel, v_pixel, z, K, camera_to_world
    )

    return pos.permute(0, 2, 1).reshape((B, -1))


def noisy_pixel_coordinates_to_world(
    y_vision,  # N, k, 2
    depth,  # N, H, W
    camera_to_world,  # N, 4, 4
    K,  # N, 3, 3
    relative_noise_scale,
    absolute_noise_scale,
):
    u_pixel, v_pixel = y_vision[..., 0], y_vision[..., 1]

    B, N_kp = v_pixel.shape
    batch_indeces = torch.arange(
        B, device=depth.device, dtype=torch.long
    ).repeat_interleave(N_kp)
    z = depth[batch_indeces, v_pixel.flatten(), u_pixel.flatten()]
    z = z.reshape(B, N_kp)

    gauss = torch.distributions.normal.Normal(
        0, z * relative_noise_scale + absolute_noise_scale
    )
    noise = gauss.sample().to(z.device)
    z = z + noise
    z = z.reshape(B, N_kp)

    pos = batched_pinhole_projection_image_to_world_coordinates_orig(
        u_pixel, v_pixel, z, K, camera_to_world
    )

    return pos


def append_depth_to_uv(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W
    img_width,
    img_height,
):
    u_normalized, v_normalized = y_vision.chunk(2, dim=-1)
    u_pixel = ((u_normalized / 2.0 + 0.5) * img_width).long().detach()
    v_pixel = ((v_normalized / 2.0 + 0.5) * img_height).long().detach()

    B, N_kp = v_pixel.shape
    batch_indeces = torch.arange(
        B, device=depth.device, dtype=torch.long
    ).repeat_interleave(N_kp)
    z = depth[batch_indeces, v_pixel.flatten(), u_pixel.flatten()]
    z = z.reshape(B, N_kp)

    return torch.cat((y_vision, z), dim=-1)


def batched_pinhole_projection_image_to_camera_coordinates(u, v, depth, intr):
    """
    Apply pinhole projection batch-wise, ie. the same projection to each point
    of the same batch.

    Parameters
    ----------
    u : Tensor (B, N)
    v : Tensor (B, N)
    depth : Tensor (B, N)
    intr : Tensor (B, 3, 3)

    Returns
    -------
    Tensor (B, N, 3)
    """
    uv1 = torch.stack((u, v, torch.ones(u.shape, device=u.device)), dim=-1)
    K_inv = invert_intrinsics(intr)

    pos = torch.transpose(torch.matmul(K_inv, torch.transpose(uv1, -1, -2)), -1, -2)

    pos = depth.unsqueeze(2).repeat(1, 1, 3) * pos
    return pos


def batched_pinhole_projection_image_to_world_coordinates(u, v, depth, intr, extr):
    """
    Apply pinhole projection batch-wise, ie. the same projection to each point
    of the same batch.

    Parameters
    ----------
    u : Tensor (B, N)
    v : Tensor (B, N)
    depth : Tensor (B, N)
    intr : Tensor (B, 3, 3)
    extr : Tensor (B, 4, 4)

    Returns
    -------
    Tensor (B, N, 3)
    """
    pos_in_camera_frame = batched_pinhole_projection_image_to_camera_coordinates(
        u, v, depth, intr
    )
    pos_in_camera_frame_homog = torch.cat(
        (
            pos_in_camera_frame,
            torch.ones(
                (*pos_in_camera_frame.shape[:-1], 1), device=pos_in_camera_frame.device
            ),
        ),
        dim=-1,
    )

    pos_in_world_homog = torch.transpose(
        torch.matmul(extr, torch.transpose(pos_in_camera_frame_homog, -1, -2)), -1, -2
    )

    return pos_in_world_homog[..., :3]


def get_b_from_homogenous_transforms(transforms):
    if type(transforms) is list:
        return [t[..., :3, 3] for t in transforms]
    elif type(transforms) is tuple:
        return tuple(t[..., :3, 3] for t in transforms)
    else:
        assert transforms.shape[-2:] == (4, 4)
        return transforms[..., :3, 3]


def set_b_in_homogenous_transforms(transforms, b):
    if type(transforms) is torch.Tensor:
        cloned = transforms.clone()
        cloned[..., :3, 3] = b
    elif type(transforms) is list:
        cloned = [t.clone() for t in transforms]
        for i in range(len(cloned)):
            cloned[i][..., :3, 3] = b[i]
    elif type(transforms) is tuple:
        cloned = tuple(t.clone() for t in transforms)
        for i in range(len(cloned)):
            cloned[i][..., :3, 3] = b[i]
    # elif type(transforms) in [list, tuple]:
    #     for i in range(len(transforms)):
    #         transforms[i][..., :3, 3] = b[i]
    else:
        raise NotImplementedError
    return cloned


def get_R_from_homogenous_transforms(transforms):
    if type(transforms) is list:
        return [t[..., :3, :3] for t in transforms]
    elif type(transforms) is tuple:
        return tuple(t[..., :3, :3] for t in transforms)
    else:
        return transforms[..., :3, :3]


def quaternion_is_unit(quaternion):
    return torch.allclose(
        torch.sum(quaternion**2, dim=-1), torch.ones_like(quaternion[..., 0]), atol=5e-4
    )


def quaternion_is_standard(quaternion):
    return torch.allclose(quaternion[..., 0], torch.abs(quaternion[..., 0]))


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


@list_or_tensor
def remove_quaternion_dim(quaternions, remove_dim=None):
    if remove_dim is not None:
        idx = [0, 1, 2, 3]
        idx.pop(remove_dim)
        quaternions = quaternions[..., idx]

    return quaternions


@list_or_tensor
def reconstruct_quaternion(quaternions, removed_dim=None):
    if removed_dim is not None:
        # quaternions = quaternions.double()
        a, b, c = torch.unbind(quaternions, dim=-1)
        d_squared = 1 - a**2 - b**2 - c**2
        # Ensure that d_squared is always positive to avoid NaNs in sqrt.
        neg_mask = d_squared < 0
        d_squared = torch.where(neg_mask, -d_squared, d_squared)
        d = torch.sqrt(d_squared)
        a = torch.where(neg_mask, -a, a)
        b = torch.where(neg_mask, -b, b)
        c = torch.where(neg_mask, -c, c)
        # disnan = torch.isnan(d)
        # assert not disnan.any()

        components = [a, b, c]
        components.insert(removed_dim, d)
        quaternions = torch.stack(components, dim=-1)

    return quaternions


@list_or_tensor
def hom_to_shift_quat(hom, skip_quat_dim=None, prefer_positives=False):
    """
    Convert a homogeneous transformation matrix to a shift and quaternion
    representation.

    Parameters
    ----------
    hom : Tensor (..., 4, 4)
        Homogeneous transformation matrices.
    skip_quat_dim : int or None
        If not None, remove the quaternion dim at this index.

    Returns
    -------
    torch.Tensor (..., 6 or 7)
        Concatenation of shift and quaternion.
    """
    shift = hom[..., :3, 3]
    rot = hom[..., :3, :3]
    quat = matrix_to_quaternion(rot, prefer_positives=prefer_positives)
    quat = remove_quaternion_dim(quat, skip_quat_dim)
    return torch.cat((shift, quat), dim=-1)


@list_or_tensor
def axis_angle_to_quaternion_batched(axis, angle):
    """
    Convert axis-angle representation to quaternion representation.

    Args:
        axis: Tensor of shape (..., 3) representing the axis of rotation.
        angle: Tensor of shape (..., ) representing the angle of rotation in
               radians.

    Returns:
        Tensor of shape (..., 4) representing the quaternion.
    """
    norm = torch.norm(axis, dim=-1, keepdim=True)
    normed_axis = axis / norm
    half_angle = angle / 2
    qw = torch.cos(half_angle)
    qxyz = torch.sin(half_angle).unsqueeze(-1) * normed_axis
    return torch.cat([qw.unsqueeze(-1), qxyz], dim=-1)


@list_or_tensor
def quaternion_to_axis_and_angle_batched(q):
    """
    Convert quaternion representation to axis-angle representation.

    Args:
        q: Tensor of shape (..., 4) representing the quaternion.

    Returns:
        axis: Tensor of shape (..., 3) representing the axis of rotation.
        angle: Tensor of shape (..., ) representing the angle of rotation in
               radians.
    """
    if torch.abs(torch.norm(q, dim=-1) - 1).max() > 2e-3:
        raise ValueError("Input quaternions must be normalized.")
    angle = 2 * torch.arccos(q[..., 0])
    axis = q[..., 1:] / torch.sqrt(1 - q[..., 0] ** 2).unsqueeze(-1)
    return axis, angle


def mod_quat(quaternion, mod_angle, dim=2):
    axis, angle = quaternion_to_axis_and_angle_batched(quaternion)
    aa = axis * angle.unsqueeze(-1)
    aa[..., dim] = torch.round(aa[..., dim] / mod_angle) * mod_angle
    angle_mod = torch.norm(aa, dim=-1)
    axis_mod = aa / angle_mod.unsqueeze(-1)
    quaternion_mod = axis_angle_to_quaternion_batched(axis_mod, angle_mod)

    return quaternion_mod


def _modulo_rotation_angle(
    pose,
    mod_angle,
    dim=2,
    skip_first=True,
    ensure_positive_rot=True,
    eps=0.02,
    sub_one=False,
):
    """
    Modulo the rotation angle of a pose around a given dim (axis).

    Args:
        pose: Tensor of shape (..., 7) representing the pose.
        mod_angle: The angle in radians to modulo the rotation angle by.
        dim: The axis to modulo the rotation angle around. Default: 2 (z-axis).
        skip_first: Whether to skip the rotation of the first frame (second
                    dimension). Used to skip the EE pose prepended to the frame
                    poses. Default: True.

    Returns:
        The pose with the rotation angle moduloed.
    """
    quaternion = pose[..., 3:]
    # if ensure_positive_rot:
    #     ref = quaternion[0]
    #     ref_axis, ref_angle = quaternion_to_axis_and_angle_batched(ref)
    #     ref_aa = ref_axis * ref_angle.unsqueeze(-1)
    if skip_first:
        quaternion = quaternion[1:]
    axis, angle = quaternion_to_axis_and_angle_batched(quaternion)
    aa = axis * angle.unsqueeze(-1)
    aa_mod = torch.remainder(aa[..., dim], mod_angle)
    if ensure_positive_rot:
        # print("=====================================")
        # print(aa[:, -1, dim])
        # print(aa_mod[0, -1])
        # print(ref_aa[-1, dim])
        # print(ref_aa[-1, dim] > aa_mod[-1, dim])
        # print(aa_mod[0, -1], mod_angle / 2 - eps, aa_mod[0, -1] - mod_angle / 2)
        # print(aa_mod[0, 0])
        aa_mod += torch.where(
            aa_mod < mod_angle / 2 + eps,
            torch.ones_like(aa_mod) * mod_angle,
            torch.zeros_like(aa_mod),
        )
        if sub_one:
            aa_mod -= mod_angle
    aa[..., dim] = aa_mod
    angle_mod = torch.norm(aa, dim=-1)
    axis_mod = aa / angle_mod.unsqueeze(-1)
    quaternion_mod = axis_angle_to_quaternion_batched(axis_mod, angle_mod)
    if skip_first:
        pose[1:, :, 3:] = quaternion_mod
    else:
        pose[..., 3:] = quaternion_mod
    return pose


# @list_or_tensor
def modulo_rotation_angle(
    pose, mod_angle, dim=2, skip_first=True, ensure_positive_rot=True, eps=0.02
):
    return tuple(
        _modulo_rotation_angle(
            p, mod_angle, dim, skip_first, ensure_positive_rot, eps, sub_one=i == 8
        )  # i==16)
        for i, p in enumerate(pose)
    )


@list_or_tensor
def quaternion_lot_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def rotate_quat_y180(quaternion):
    return quaternion_lot_multiply(y180_quaternion, quaternion)


@list_or_tensor
def sin(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)


@list_or_tensor
def cos(x: torch.Tensor) -> torch.Tensor:
    return torch.cos(x)


def rotate_vector_by_quaternion(
    vector: torch.Tensor, quat: torch.Tensor
) -> torch.Tensor:
    """
    Rotate a (batch of) vector(s) by a (batch of) quaternion(s).

    See: https://math.stackexchange.com/a/535223

    Args:
        vector: Tensor of shape (..., 3) representing the vector.
        quaternion: Tensor of shape (..., 4) representing the quaternion.

    Returns:
        Tensor of shape (..., 3) representing the rotated vector.
    """
    assert quaternion_is_unit(quat)

    quat_inv = quaternion_invert(quat)

    vector = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)

    # R = H(H(Q, P), Q')
    rotated = quaternion_multiply(quaternion_multiply(quat, vector), quat_inv)

    assert torch.allclose(rotated[..., 0], torch.zeros_like(rotated[..., 0]), atol=5e-7)

    return rotated[..., 1:]


def frame_transform_pos_quat(
    obj_pose: torch.Tensor, new_frame: torch.Tensor
) -> torch.Tensor:
    """
    Given the pose of and object and the pose of a new frame, compute the pose of the object
    in the new frame.
    Takes and returns position and quaternion.
    """
    assert obj_pose.shape[-1] == 7
    assert new_frame.shape[-1] == 7

    obj_b, obj_q = obj_pose[..., :3], obj_pose[..., 3:]
    f_b, f_q = new_frame[..., :3], new_frame[..., 3:]

    obj_A = quaternion_to_matrix(obj_q)
    obj_hom = homogenous_transform_from_rot_shift(obj_A, obj_b)

    f_A = quaternion_to_matrix(f_q)

    world2f = invert_homogenous_transform(homogenous_transform_from_rot_shift(f_A, f_b))

    obj_local_hom = world2f @ obj_hom
    pos_local = obj_local_hom[..., :3, 3]
    rot_local = quaternion_pose_diff(f_q, obj_q)

    return torch.cat([pos_local, rot_local], dim=-1)
