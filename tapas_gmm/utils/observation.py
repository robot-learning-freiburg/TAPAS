from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from omegaconf import MISSING
from tensordict import TensorDict, tensorclass
from tensordict.tensordict import TensorDictBase, pad_sequence

from tapas_gmm.utils.config import _SENTINELS, SET_PROGRAMMATICALLY
from tapas_gmm.utils.geometry_np import (
    compute_angle_between_poses,
    compute_angle_between_quaternions,
    compute_distance_between_poses,
)


# TODO: not sure if this is the best place for this
@dataclass
class ObservationConfig:
    cameras: tuple[str, ...]
    image_crop: tuple[int, int, int, int] | None
    # image_dim: tuple[int, int] | _SENTINELS = SET_PROGRAMMATICALLY
    image_dim: Any = SET_PROGRAMMATICALLY

    # TODO: these should be encoder keys
    disk_read_embedding: bool = False
    disk_read_keypoints: bool = False


tc_collate_fn = torch.stack

empty_batchsize = torch.Size([])


class MaskTypes(Enum):
    NONE = None
    GT = "gt"
    TSDF = "tsdf"


class SampleTypes(Enum):
    GT = 0
    DC = 1
    CAM_SINGLE = 2
    CAM_PAIR = 3


# Mapping from internal names to name on disk. Allows to adapt to different
# naming conventions.
# NOTE: Using MappingProxyType to make the dict immutable.
RAW_CAMERA_ATTRIBUTES = MappingProxyType(
    {
        "rgb": "rgb",
        "depth": "d",
        "extr": "ext",
        "intr": "int",
    }
)


CAMERA_MASK_ATTRIBUTES = MappingProxyType(
    {
        "mask_gt": "mask_gt",
        "mask_tsdf": "mask_tsdf",
    }
)

CAMERA_EMBEDDING_ATTRIBUTES = MappingProxyType(
    {
        "descriptor": "descriptor",
        # 'kp': 'kp',
    }
)


ALL_CAMERA_ATTRIBUTES = (
    RAW_CAMERA_ATTRIBUTES | CAMERA_MASK_ATTRIBUTES | CAMERA_EMBEDDING_ATTRIBUTES
)


ROBOT_ATTRIBUTES = MappingProxyType(
    {
        "action": "action",
        "ee_pose": "ee_pose",
        "joint_pos": "joint_pos",
        "joint_vel": "joint_vel",
        "gripper_state": "gripper_state",
    }
)


SCENE_ATTRIBUTES = MappingProxyType(
    {
        "feedback": "feedback",
        "object_poses": "object_poses",
    }
)

OPTIONAL_SCENE_ATTRIBUTES = MappingProxyType(
    {
        "kp": "kp",
    }
)

GENERIC_ATTRIBUTES = ROBOT_ATTRIBUTES | SCENE_ATTRIBUTES

CAMERA_OBS_ATTRIBUTES = (
    list(RAW_CAMERA_ATTRIBUTES.keys())
    + list(CAMERA_EMBEDDING_ATTRIBUTES.keys())
    + ["mask"]
)
GENERIC_STATE_ATTRIBUTES = list(GENERIC_ATTRIBUTES.keys())

OPTIONAL_STATE_ATTRIBUTES = list(OPTIONAL_SCENE_ATTRIBUTES.keys())

SCENE_OBS_NON_CAM_ATTRIBUTES = GENERIC_STATE_ATTRIBUTES + OPTIONAL_STATE_ATTRIBUTES


def is_flat_attribute(attr: str) -> bool:
    """
    Helper function to determine if an attribute is flat, ie. not
    time-dependent, ie. constant across the trajectory.
    Currently this only applies to camera intrinsics.

    Parameters
    ----------
    attr : str
        Attribute name

    Returns
    -------
    bool
        Whether the attribute is flat.
    """

    return attr.endswith("intr") or attr.endswith("int")


def tc_ss(list_of_tensordicts: Sequence[TensorDictBase]) -> tuple[TensorDictBase]:
    lens = tuple(len(td) for td in list_of_tensordicts)
    mean_len = int(np.mean(lens))
    ss_idcs = tuple(get_idx_by_target_len(l, target_len=mean_len) for l in lens)
    subsampled = tuple(
        downsample_tensordict_by_idx(td, idx)
        for td, idx in zip(list_of_tensordicts, ss_idcs)
    )
    return subsampled


def collate(
    observations: tuple[TensorDictBase], pad: bool = False, subsample: bool = False
) -> TensorDictBase:
    """
    Collate a list of TensorDict/TensorClass observations into a batch.
    Supports padding of the observations to the same length.

    Parameters
    ----------
    observations : tuple[TensorDict/TensorClass]
        The observations to collate.
    pad : bool, optional
        Whether to pad the sequences to common length. Only needed if obs have
        different lengths, by default False
    subsample : bool, optional
        Whether to subsample the sequences to a their mean length. Only needed if obs have different lengths, by default False

    Returns
    -------
    TensorDict/TensorClass
        The collated batch.
    """

    if pad:
        return pad_sequence(observations)
    elif subsample:
        obs_ss = tc_ss(observations)
        return tc_collate_fn(obs_ss)
    else:
        return tc_collate_fn(observations)  # type: ignore


def make_cam_attr_name(cam_name: str, attr_name: str) -> str:
    """
    Helper function to map camera names and attribute names to the names used
    on disk.

    Parameters
    ----------
    cam_name : str
        Name of the camera.
    attr_name : str
        Name of the attribute.

    Returns
    -------
    str
        File name of the attribute.
    """
    return "cam_" + cam_name + "_" + attr_name


def get_cam_attributes(
    cam_name: str, mask_type: MaskTypes | None, skip_rgb=False, shorten_cam_names=True
) -> dict[str, str]:
    """
    Returns a dictionary mapping the requested camera attributes to their
    respective names on disk.

    Parameters
    ----------
    cam_name : string
        Name of the camera.
    mask_type : MaskTypes
        Type of mask to use. None, GT or TSDF.
    skip_raw_cam_attr : bool, optional
        Exclude RGB and and only load masks, depth, ext, int. Useful when using
        pre-embedding images. Even finer-grained control would further reduce
        IO when not needing, eg depth. But RGB are the most important culprit.
        Masks are further skipped when mask_type is None.
        By default False
    shorten_cam_names : bool, optional
        Only use the first letter of the camera name, by default True

    Returns
    -------
    dict
        Maps the attributes to the file names.
    """
    # TODO: when skipping raw cam attributes, might still need to add depth
    # for KP/BASK model.
    cam = cam_name[0] if shorten_cam_names else cam_name

    cam_attr = list(RAW_CAMERA_ATTRIBUTES.keys())

    if skip_rgb:
        cam_attr.remove("rgb")

    cam_attributes = {
        attr: make_cam_attr_name(cam, RAW_CAMERA_ATTRIBUTES[attr]) for attr in cam_attr
    }

    if mask_type not in [MaskTypes.NONE, None]:
        mask_attr_name = "mask_" + mask_type.value  # type: ignore
        cam_attributes.update({"mask": make_cam_attr_name(cam, mask_attr_name)})

    return cam_attributes


def get_idx_by_target_len(traj_len: int, target_len: int) -> list[int]:
    """
    Given a source and target length, return a list of indeces to subsample
    the trajectory to the target length at regular intervals.

    Parameters
    ----------
    traj_len : int
        The source length.
    target_len : int
        The target length.

    Returns
    -------
    list[int]
        Indeces for subsampling.
    """
    if traj_len == target_len:
        return list(range(traj_len))
    elif traj_len < target_len:
        return list(range(traj_len)) + [traj_len - 1] * (target_len - traj_len)
    else:
        indeces = np.linspace(start=0, stop=traj_len - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
        return indeces.tolist()


def get_idx_by_pose_difference_threshold(
    poses: Sequence[np.ndarray], dist_threshold: float, angle_threshold: float
) -> list[int]:
    """
    Given a trajectory of poses, return a list of indeces to subsample st that
    the resulting trajectory has a minimum distance between poses, either in
    terms of linear distance or angle between quaternions.

    Poses are assumed to be 7-dimensional, with the first 3 dimensions being
    the position and the last 4 the quaternion.

    Parameters
    ----------
    poses : Iterable[np.ndarray]
        The trajectory of poses.
    dist_threshold : float
        The euclidean distance threshold.
    angle_threshold : float
        The angular distance threshold.

    Returns
    -------
    list[int]
        The indeces for subsampling.
    """
    idx = [0]
    current_pose = poses[0]

    for i in range(1, len(poses)):
        next_pose = poses[i]
        dist = np.linalg.norm(current_pose[:3] - next_pose[:3])
        angle = compute_angle_between_quaternions(current_pose[3:], next_pose[3:])

        if dist > dist_threshold or angle > angle_threshold:
            idx.append(i)
            current_pose = next_pose

    return idx


def get_idx_by_pose_difference_threshold_matrix(
    poses: Sequence[np.ndarray], dist_threshold: float, angle_threshold: float
) -> list[int]:
    """
    Given a trajectory of poses, return a list of indeces to subsample st that
    the resulting trajectory has a minimum distance between poses, either in
    terms of linear distance or angle between quaternions.

    Poses are assumed to be 4x4 matrices.

    Parameters
    ----------
    poses : Sequence[np.ndarray]
        The trajectory of poses.
    dist_threshold : float
        The euclidean distance threshold.
    angle_threshold : float
        The angular distance threshold.

    Returns
    -------
    list[int]
        The indeces for subsampling.
    """
    idx = [0]
    current_pose = poses[0]

    for i in range(1, len(poses)):
        next_pose = poses[i]
        dist = compute_distance_between_poses(current_pose, next_pose)
        angle = compute_angle_between_poses(current_pose, next_pose)

        if dist > dist_threshold or angle > angle_threshold:
            idx.append(i)
            current_pose = next_pose

    return idx


def get_idx_by_img_difference_threshold(rgb, depth):
    assert rgb.shape[0] == depth.shape[0]

    idx = [0]
    current_rgb = rgb[0]
    current_depth = depth[0]

    for i in range(1, rgb.shape[0]):
        next_rgb = rgb[i]
        next_depth = depth[i]
        if not (
            torch.isclose(current_rgb, next_rgb).all()
            and torch.isclose(current_depth, next_depth).all()
        ):
            idx.append(i)
            current_rgb, current_depth = next_rgb, next_depth

    return idx


def downsample_traj_by_idx(traj: Sequence, indeces: Iterable[int]) -> list:
    """
    Downsample a trajectory by picking the observations at the given indeces.

    Parameters
    ----------
    traj : Sequence
        The original trajectory.
    indeces : Iterable[int]
        The indeces to pick.

    Returns
    -------
    list
        The subsampled trajectory.
    """
    return [traj[i] for i in indeces]


def downsample_traj_torch(traj: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Downsample a trajectory given as a torch tensor to the target length.
    Equally spaces the target length points in the trajectory and picks the
    closest observation from the original trajectory.
    The first dimension is assumed to be the time dimension.

    Parameters
    ----------
    traj : torch.Tensor
        The trajectory.
    target_len : int
        Number of target time steps.

    Returns
    -------
    torch.Tensor
        The subsampled trajectory.

    Raises
    ------
    ValueError
        If the target length is larger than the trajectory length.
        Could remove to also allow upsampling.
    """
    if len(traj) == target_len:
        return traj
    elif len(traj) < target_len:
        raise ValueError("Traj shorter than target length.")
    else:
        indeces = np.linspace(start=0, stop=len(traj) - 1, num=target_len)
        indeces = np.round(indeces).astype(int)
    return traj.index_select(dim=0, index=torch.tensor(indeces))


def downsample_to_target_freq(
    traj: torch.Tensor, target_freq: int, source_freq=20
) -> torch.Tensor:
    """
    Downsample a trajectory given as a torch tensor from the source frequency
    to the target frequency.
    Equally spaces the target length points in the trajectory and picks the
    closest observation from the original trajectory.
    The first dimension is assumed to be the time dimension.

    Parameters
    ----------
    traj : torch.Tensor
        The trajectory, given with the source frequency.
    target_freq : int
        The target observation frequency.
    source_freq : int, optional
        The original observation frequency, by default 20.

    Returns
    -------
    torch.Tensor
        The subsampled trajectory.
    """

    target_len = int(len(traj) * target_freq / source_freq)

    return downsample_traj_torch(traj, target_len)


def downsample_tensordict_by_idx(
    traj: TensorDictBase, indeces: list[int]
) -> TensorDictBase:
    """
    Downsampling function for TensorDicts and TensorClasses.
    """
    return traj.get_sub_tensordict(torch.tensor(indeces))


def get_ordered_attributes_from_tensordict(
    data: TensorDictBase, order: Iterable[str], stack=False
) -> torch.Tensor | tuple[torch.Tensor]:
    """
    Helper function to get the attributes of a TensorDict in the given order.

    Parameters
    ----------
    data : TensorDict
        The TensorDict to get the attributes from.
    order : iterable[str]
        Iterable of attribute names in the desired order.
    stack : bool, optional
        Whether to stack the resulting Tensors. Else returns a tuple of them.
        By default False

    Returns
    -------
    torch.Tensor/tuple[torch.Tensor]
        The attributes in the given order.
    """
    gen = (data[k] for k in order)

    if stack:
        return torch.stack(list(gen))  # type: ignore
    else:
        return tuple(gen)  # type: ignore


def get_camera_obs(self, stack=False) -> torch.Tensor | tuple[torch.Tensor]:
    """
    Helper function to get the camera observations from a MultiCamObservation
    in order.

    Parameters
    ----------
    stack : bool, optional
        Whether to return a stacked tensor or a tuple, by default False

    Returns
    -------
    torch.Tensor/tuple[torch.Tensor]
        The camera observations in the internally stored order.
    """
    return get_ordered_attributes_from_tensordict(
        self.cameras, self.get_camera_names(), stack=stack
    )


def get_camera_names(self) -> tuple[str]:
    """
    Helper function to get the camera names from a MultiCamObservation in
    order.

    Returns
    -------
    tuple[str]
        Tuple of camera names in the internally stored order.
    """
    return self.cameras["_order"].order


def get_object_labels(masks: Iterable[torch.Tensor]) -> list:
    """
    Get the sorted, unique object labels from an Iterable of object masks.

    Parameters
    ----------
    tensors : Iterable[torch.Tensor]
        Object masks.

    Returns
    -------
    list
        The sorted, unique object labels.
    """
    return sorted(list(set().union(*[np.unique(t).tolist() for t in masks])))


@tensorclass  # type: ignore
class CameraOrder:
    """
    Helper class to store the camera order of a MultiCamObservations as
    TensorDicts can't store tuples directly.
    """

    order: tuple[str]

    @classmethod
    def _create(cls, cam_order: tuple):
        return cls(order=cam_order, batch_size=empty_batchsize)  # type: ignore

    def __iter__(self):
        # Make the class iterable by iterating over wrapped tuple.
        return iter(self.order)


def dict_to_tensordict(data: dict, batch_size=None):
    """
    Transform a dict to a TensorDict.
    In contrast to the TensorClass, the TensorDict does not have a fixed
    set of attributes. Thus, we can have dynamically determined attributes.
    Needed for example for object poses, which differ betwen datasets.

    Parameters
    ----------
    data : dict
        Mapping strings to tensors.
    batch_size : torch.Size, optional
        The batch size of the tensors. By default None, ie non-batched.

    Returns
    -------
    TensorDict
        The same data as a TensorDict.
    """
    if batch_size is None:
        batch_size = empty_batchsize

    return TensorDict(data, batch_size=batch_size)


def make_tensorclass(
    name,
    tensor_fields=None,
    other_fields=None,
    methods=None,
    properties=None,
    bases=tuple(),
):
    """
    Generates a tensorclass with the given name and fields.

    Parameters
    ----------
    name : str
        Name of the class.
    tensor_fields : iterable[str]
        Iterable of field names for fields containing tensors.
    other_fields : dict[str, type]
        Dictionary mapping field names to their type. Used to annotate fields
        that are not tensors.
    methods : dict[str, callable]
        Dictionary mapping method names to their implementation.
    properties : dict[str, callable]
        Dictionary mapping property names to their getter-implementation.
        These properties will have no setter, as we want to keep observations
        immutable.

    Returns
    -------
    TensorClass
        The configured class.
    """
    # Tensorclass uses dataclasses under the hood. To get the attributes of the
    # original class, dataclass relies on the attributes' annotations. So add
    # annotations for automatic attribute creation.

    if tensor_fields is None:
        tensor_fields = []

    if other_fields is None:
        other_fields = {}

    if methods is None:
        methods = {}

    if properties is None:
        properties = {}

    attr = {f: None for f in tensor_fields} | {f: None for f in other_fields}

    tensor_annotations = {f: torch.Tensor for f in tensor_fields}

    attr["__annotations__"] = tensor_annotations | other_fields  # type: ignore

    Class = type(name, bases, attr)

    TensorClass = tensorclass(Class)

    for method_name, method in methods.items():
        setattr(TensorClass, method_name, method)

    for prop_name, prop_func in properties.items():
        setattr(TensorClass, prop_name, property(prop_func))

    # logger.info(f"Registered TensorClass {name} in global scope.")
    # globals()[name] = TensorClass

    return TensorClass


SingleCamObservation = make_tensorclass(
    "SingleCamObservation", tensor_fields=CAMERA_OBS_ATTRIBUTES
)

SceneObservation = make_tensorclass(
    "SceneObservation",
    tensor_fields=SCENE_OBS_NON_CAM_ATTRIBUTES,
    other_fields={"cameras": TensorDict},
    methods={
        "get_camera_obs": get_camera_obs,
        "get_camera_names": get_camera_names,
    },
    properties={
        "camera_obs": get_camera_obs,
        "camera_names": get_camera_names,
    },
)

SingleCamSceneObservation = make_tensorclass(
    "SingleCamSceneObservation",
    tensor_fields=GENERIC_STATE_ATTRIBUTES + CAMERA_OBS_ATTRIBUTES,
    # methods={'get_camera_obs': get_camera_obs_single},
    # properties={'camera_obs': get_camera_obs_single}
)


def make_bespoke_scene_observation_class_from_data(data: dict, cls_name: str | None):
    keys = list(data.keys())
    new_keys = tuple(set(keys) - set(SCENE_OBS_NON_CAM_ATTRIBUTES) - {"cameras"})

    cls = make_tensorclass(cls_name, tensor_fields=new_keys, bases=(SceneObservation,))

    return cls


def tensorclass_from_tensordict(td: TensorDictBase, name: str | None = None):
    tensor_keys = [k for k, v in td.items() if torch.is_tensor(v)]
    other_keys = {k: type(v) for k, v in td.items() if not torch.is_tensor(v)}

    if "cameras" in other_keys.keys():
        methods = {
            "get_camera_obs": get_camera_obs,
            "get_camera_names": get_camera_names,
        }
        properties = {
            "camera_obs": get_camera_obs,
            "camera_names": get_camera_names,
        }
    else:
        methods = dict()
        properties = dict()

    TensorClass = make_tensorclass(
        name,
        tensor_fields=tensor_keys,
        other_fields=other_keys,
        methods=methods,
        properties=properties,
    )

    return TensorClass


def dropout_single_camera(
    obs: SingleCamObservation, attributes: tuple[str, ...]  # type: ignore
) -> SingleCamObservation:  # type: ignore
    """
    Dropout, ie set to zero, the given attributes of a
    SingleCameraObservation.

    Parameters
    ----------
    obs : SceneObservation
        _description_
    attributes : tuple[str]
        _description_

    Returns
    -------
    SceneObservation
        _description_
    """
    for attr in attributes:
        val = getattr(obs, attr)

        if val is not None:
            setattr(obs, attr, torch.zeros_like(val))

    return obs


# TODO: need to drop out sequence of obs, eg a second, hence 20 obs?
def random_obs_dropout(
    obs: SceneObservation,  # type: ignore
    p: float | None = None,
    drop_all: bool = False,
    attributes: tuple[str, ...] = ("rgb", "depth"),
) -> SceneObservation:  # type: ignore
    """
    Dropout, ie set to zero, random camera observations with probability p.
    If p is None, no dropout is performed.
    If drop_all, all cameras are dropped out together with probability p, else
    each camera is dropped out independently with probability p.

    Parameters
    ----------
    obs : SceneObservation
        The original observation.
    p : float | None, optional
        Dropout probability, by default None
    drop_all : bool, optional
        Whether to drop all cams together, by default False
    attributes : tuple[str], optional
        The attributes to drop out, by default tuple rgb and depth.

    Returns
    -------
    SceneObservation
        Observation with dropout.
    """
    if p is None:
        return obs

    if drop_all:
        sample = np.random.binomial(1, p)
        if sample:
            for cam in obs.camera_names:
                obs = dropout_single_camera(obs.cameras[cam], attributes)
    else:
        for cam in obs.camera_names:
            sample = np.random.binomial(1, p)
            if sample:
                obs = dropout_single_camera(obs.cameras[cam], attributes)

    return obs


def tensor_dict_equal(td1: TensorDictBase, td2: TensorDictBase) -> bool:
    """
    Check if two TensorDicts are equal.

    Parameters
    ----------
    td1 : TensorDict
        First TensorDict.
    td2 : TensorDict
        Second TensorDict.

    Returns
    -------
    bool
        Whether the TensorDicts are equal.
    """
    if td1.sorted_keys != td2.sorted_keys:
        return False

    for k in td1.keys():  # type: ignore
        if not torch.equal(td1[k], td2[k]):  # type: ignore
            return False

    return True
