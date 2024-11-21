import math

import numpy as np
import torch
from loguru import logger

identity_quaternion_np = np.array([1, 0, 0, 0])

# NOTE: all quaternion functions are for real-first quaternions! NOTE

# NOTE: translated some functions from https://github.com/facebookresearch/pytorch3d to numpy


def quaternion_to_axis_angle(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions[..., 1:], axis=-1, keepdims=True)
    half_angles = np.arctan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = np.abs(angles) < eps
    sin_half_angles_over_angles = np.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        np.sin(half_angles[~small_angles]) / angles[~small_angles]
    )

    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_angles_over_angles], axis=-1
    )
    return quaternions


def quaternion_to_axis_and_angle(q):
    q = np.array(q)
    if np.abs(np.linalg.norm(q) - 1) > 2e-3:
        raise ValueError("Input quaternion must be normalized.")
    angle = 2 * np.arccos(q[0])
    axis = q[1:] / np.sqrt(1 - q[0] ** 2)
    return axis, angle


def axis_and_angle_to_quaternion(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    axis /= np.linalg.norm(axis)
    quaternion = np.zeros(4)
    quaternion[0] = np.cos(angle / 2)
    quaternion[1:] = np.sin(angle / 2) * axis
    return quaternion


def compute_angle_between_quaternions(q, r):
    """
    Computes the angle between two quaternions.

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

    theta = 2 * np.arccos(2 * np.dot(q, r) ** 2 - 1)
    return theta


def compute_distance_between_poses(pose_a, pose_b):
    """
    Computes the linear difference between pose_a and pose_b
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Distance between translation component of the poses
    :rtype:
    """

    pos_a = pose_a[0:3, 3]
    pos_b = pose_b[0:3, 3]

    return np.linalg.norm(pos_a - pos_b)


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array(
            [
                [m00 - m11 - m22, 0.0, 0.0, 0.0],
                [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
            ]
        )
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


matrix_to_quaternion = quaternion_from_matrix


def compute_angle_between_poses(pose_a, pose_b):
    """
    Computes the angle distance in radians between two homogenous transforms
    :param pose_a: 4 x 4 homogeneous transform
    :type pose_a:
    :param pose_b:
    :type pose_b:
    :return: Angle between poses in radians
    :rtype:
    """

    quat_a = quaternion_from_matrix(pose_a)
    quat_b = quaternion_from_matrix(pose_b)

    return compute_angle_between_quaternions(quat_a, quat_b)


def conjugate_quat(quaternion: np.ndarray) -> np.ndarray:
    assert quaternion.shape[-1] == 4
    conj = np.copy(quaternion)
    conj[..., 1:4] = -conj[..., 1:4]
    # conj[..., 0] = -conj[..., 0]

    return conj


def quaternion_multiply(quaternion0: np.ndarray, quaternion1: np.ndarray) -> np.ndarray:
    """
    Aka Hamilton product.
    """
    aw, ax, ay, az = (
        quaternion0[..., 0],
        quaternion0[..., 1],
        quaternion0[..., 2],
        quaternion0[..., 3],
    )
    bw, bx, by, bz = (
        quaternion1[..., 0],
        quaternion1[..., 1],
        quaternion1[..., 2],
        quaternion1[..., 3],
    )

    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw

    return np.stack([ow, ox, oy, oz], axis=-1)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat / norm


def _axis_angle_rotation(axis: str, angle: np.ndarray) -> np.ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert XYZ Euler angles in radians to rotation matrices.
    """
    X, Y, Z = euler_angles[..., 0], euler_angles[..., 1], euler_angles[..., 2]
    matrices = [_axis_angle_rotation(c, e) for c, e in zip("XYZ", (X, Y, Z))]
    return np.matmul(np.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def euler_angles_to_axis_angle(euler_angles: np.ndarray) -> np.ndarray:
    return matrix_to_axis_angle(euler_angles_to_matrix(euler_angles))


# taken from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
# def quaternion_to_matrix(Q):
#     """
#     Covert a quaternion into a full three-dimensional rotation matrix.

#     Input
#     :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

#     Output
#     :return: A 3x3 element matrix representing the full 3D rotation matrix.
#              This rotation matrix converts a point in the local reference
#              frame to a point in the global reference frame.
#     """
#     # Extract the values from Q
#     q0 = Q[0]
#     q1 = Q[1]
#     q2 = Q[2]
#     q3 = Q[3]

#     # First row of the rotation matrix
#     r00 = 2 * (q0 * q0 + q1 * q1) - 1
#     r01 = 2 * (q1 * q2 - q0 * q3)
#     r02 = 2 * (q1 * q3 + q0 * q2)

#     # Second row of the rotation matrix
#     r10 = 2 * (q1 * q2 + q0 * q3)
#     r11 = 2 * (q0 * q0 + q2 * q2) - 1
#     r12 = 2 * (q2 * q3 - q0 * q1)

#     # Third row of the rotation matrix
#     r20 = 2 * (q1 * q3 - q0 * q2)
#     r21 = 2 * (q2 * q3 + q0 * q1)
#     r22 = 2 * (q0 * q0 + q3 * q3) - 1

#     # 3x3 rotation matrix
#     rot_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

#     return rot_matrix


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    r, i, j, k = (
        quaternion[..., 0],
        quaternion[..., 1],
        quaternion[..., 2],
        quaternion[..., 3],
    )

    two_s = 2.0 / (quaternion * quaternion).sum(-1)

    o = np.stack(
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
        axis=-1,
    )
    return o.reshape(quaternion.shape[:-1] + (3, 3))


def homogenous_transform_from_rot_shift(rot, shift):
    if len(rot.shape) == 3:
        B = rot.shape[0]
        matrix = np.zeros((B, 4, 4))
        matrix[:, :3, :3] = rot
        matrix[:, :3, 3] = shift
        matrix[:, 3, 3] = 1
    elif len(rot.shape) == 2:
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = rot
        matrix[:3, 3] = shift
        matrix[3, 3] = 1
    else:
        raise ValueError(
            f"rot must be 2D or 3D array, got shapes {rot.shape} and {shift.shape}"
        )

    return matrix


def invert_homogenous_transform(matrix):
    """
    Works with batched as well.
    """
    rot = matrix[..., :3, 0:3]
    shift = matrix[..., :3, 3:4]

    rot_inv = np.swapaxes(rot, -1, -2)
    shift_inv = np.matmul(-rot_inv, shift)

    inverse = np.zeros_like(matrix)
    inverse[..., :3, 0:3] = rot_inv
    inverse[..., :3, -1:] = shift_inv
    inverse[..., 3, -1:] = 1

    return inverse


def torch_np_wrapper(func, tensor, device):
    return torch.from_numpy(func(tensor.cpu())).to(device)


def arccos_star(rho):
    if type(rho) is not np.ndarray:
        # Check rho
        if abs(rho) > 1:
            # Check error:
            if abs(rho) - 1 > 1e-6:
                print("arcos_star: abs(rho) > 1+1e-6:".format(abs(rho) - 1))

            # Fix error
            rho = 1 * np.sign(rho)

        # Single mode:
        if -1.0 <= rho < 0.0:
            return np.arccos(rho) - np.pi
        else:
            return np.arccos(rho)
    else:
        # Batch mode:
        rho = np.array([rho])

        ones = np.ones(rho.shape)
        rho = np.max(np.vstack((rho, -1 * ones)), axis=0)
        rho = np.min(np.vstack((rho, 1 * ones)), axis=0)

        acos_rho = np.zeros(rho.shape)
        sl1 = np.ix_((-1.0 <= rho) * (rho < 0.0) == 1)
        sl2 = np.ix_((1.0 > rho) * (rho >= 0.0) == 1)

        acos_rho[sl1] = np.arccos(rho[sl1]) - np.pi
        acos_rho[sl2] = np.arccos(rho[sl2])

        return acos_rho


def quat_log_e(q, reg=1e-6):
    if abs(q[0] - 1.0) > reg:
        return arccos_star(q[0]) * (q[1:] / np.linalg.norm(q[1:]))
    else:
        return np.zeros(3)


def log_e(dp):
    assert len(dp.shape) == 1
    assert dp.shape[0] % 7 == 0

    n_frames = dp.shape[0] // 7

    out = []
    # split into per frame, then process the quaternion parts
    for i in range(n_frames):
        frame = dp[i * 7 : (i + 1) * 7]
        out.append(frame[:3])
        out.append(quat_log_e(frame[3:7]))

    out = np.concatenate(out, axis=0)

    return out


def quaternion_diff(q1, q2):
    raise NotImplementedError(
        "THIS FUNCTION IS LIKELY WRONG! SHOULD BE quaternion_multiply(conjugate_quat(q1), q2), where q1 is the current pose and q2 is the next pose."
    )
    return quaternion_multiply(q1, conjugate_quat(q2))


def quaternion_pose_diff(current: np.ndarray, next: np.ndarray) -> np.ndarray:
    return quaternion_multiply(conjugate_quat(current), next)


def quaternion_is_unit(quat: np.ndarray) -> bool:
    return np.isclose(np.linalg.norm(quat, axis=-1), 1.0).all()


def quaternion_invert(quat: np.ndarray) -> np.ndarray:
    assert quaternion_is_unit(quat)

    return conjugate_quat(quat)


def rotate_vector_by_quaternion(vector: np.ndarray, quat: np.ndarray) -> np.ndarray:
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

    vector = np.concatenate([np.zeros_like(vector[..., :1]), vector], axis=-1)

    # R = H(H(Q, P), Q')
    rotated = quaternion_multiply(quaternion_multiply(quat, vector), quat_inv)

    assert np.isclose(rotated[..., 0], 0.0, atol=1e-7).all()

    return rotated[..., 1:]


def overlapping_split(arr: np.ndarray, idcs: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Like np.split, but resulting segments overlap at first and last index.
    """
    idcs = np.concatenate(([0], idcs, [arr.shape[axis]]))

    segments = []
    for i in range(len(idcs) - 1):
        start = idcs[i]
        end = idcs[i + 1]

        if i > 0:
            start -= 1

        segments.append(np.take(arr, range(start, end), axis=axis))

    return segments


def quat_real_last_to_real_first(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from real-last to real-first representation.
    """
    assert quat.shape[-1] == 4
    return np.concatenate([quat[..., -1:], quat[..., :-1]], axis=-1)


def quat_real_first_to_real_last(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from real-first to real-last representation.
    """
    assert quat.shape[-1] == 4
    return np.concatenate([quat[..., 1:], quat[..., :1]], axis=-1)


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """
    Ensure quaternion continuity by flipping the sign of the quaternion if
    the dot product with the previous quaternion is negative.
    """
    assert len(quats.shape) == 2
    assert quats.shape[-1] == 4

    for i in range(1, len(quats)):
        if np.dot(quats[i - 1], quats[i]) < 0:
            quats[i] *= -1

    return quats


def ensure_quat_positive_real_part(quat: np.ndarray) -> np.ndarray:
    """
    Ensure quaternion has positive real part.
    """
    assert quat.shape[-1] == 4
    mask = quat[..., 0] < 0
    quat[mask] *= -1
    return quat


def frame_transform_pos_quat(obj_pose: np.ndarray, new_frame: np.ndarray) -> np.ndarray:
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
    pos_local = obj_local_hom[:3, 3]
    rot_local = quaternion_pose_diff(f_q, obj_q)

    return np.concatenate([pos_local, rot_local], axis=-1)
