import random
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property, lru_cache
from typing import Any, Iterable

import numpy as np
import torch
import torch.multiprocessing
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

# NOTE: using Dataset in type annotations as replacement for SceneDataset,
# to avoid a circular import. This is a bit hacky. TODO: fix.
# from tapas_gmm.dataset.scene import SceneDataset
import tapas_gmm.dense_correspondence.correspondence_augmentation as correspondence_augmentation  # noqa 501
import tapas_gmm.dense_correspondence.correspondence_finder as correspondence_finder
from tapas_gmm.utils.misc import configure_class_instance
from tapas_gmm.utils.observation import MaskTypes, SampleTypes, SingleCamObservation
from tapas_gmm.viz.correspondence_plotter import cross_debug_plot, debug_plots
from tapas_gmm.viz.operations import (
    channel_front2back,
    get_image_tensor_mean,
    get_image_tensor_std,
    uv_to_flattened_pixel_locations,
)

torch.multiprocessing.set_sharing_strategy("file_system")


class DcDatasetDataType(Enum):
    SINGLE_OBJECT_WITHIN_SCENE = 0
    SINGLE_OBJECT_ACROSS_SCENE = 1
    DIFFERENT_OBJECT = 2
    MULTI_OBJECT = 3
    SYNTHETIC_MULTI_OBJECT = 4


@dataclass
class DataTypeProbabilities:
    SINGLE_OBJECT_WITHIN_SCENE: float
    SINGLE_OBJECT_ACROSS_SCENE: float
    DIFFERENT_OBJECT: float
    MULTI_OBJECT: float
    SYNTHETIC_MULTI_OBJECT: float


@dataclass
class DCDataConfig:
    contrast_set: Any
    contr_cam: tuple[str] | None
    debug: bool
    domain_randomize: bool
    random_crop: bool
    crop_size: tuple[int, int]
    sample_crop_size: bool
    random_flip: bool
    sample_matches_only_off_mask: bool
    num_matching_attempts: int
    num_non_matches_per_match: int
    _use_image_b_mask_inv: bool
    fraction_masked_non_matches: float | str
    cross_scene_num_samples: int  # for different/same obj cross scene
    data_type_probabilities: Any  # tuple[tuple[DcDatasetDataType, float | int], ...]
    only_use_labels: tuple[int] | None
    pose_indeces: tuple[int] | None  # needed if use_object_pose and only_use_labels
    only_use_first_object_label: bool
    conflate_so_object_labels: bool
    use_object_pose: bool
    mask_type: MaskTypes


class DenseCorrespondenceDataset(Dataset):
    def __init__(
        self,
        scene_dataset: Dataset,
        dc_config: DCDataConfig | None,
        sample_type: SampleTypes,
        cameras: tuple[str, ...],
    ) -> None:
        self.scene_data = scene_dataset

        self.dc_config = dc_config

        self.cameras = cameras
        self.sample_type = sample_type

        self._verbose = False

        if dc_config is not None and sample_type is SampleTypes.DC:
            assert self.dc_config is not None  # for pylance

            if self.dc_config.fraction_masked_non_matches == "auto":
                self.num_masked_non_matches_per_match = "auto"
                self.num_background_non_matches_per_match = "auto"
            else:
                self.fraction_background_non_matches = (
                    1 - self.dc_config.fraction_masked_non_matches
                )  # type: ignore

                self.num_masked_non_matches_per_match = int(
                    self.dc_config.fraction_masked_non_matches
                    * self.dc_config.num_non_matches_per_match
                )
                self.num_background_non_matches_per_match = int(
                    self.fraction_background_non_matches
                    * self.dc_config.num_non_matches_per_match
                )

            self._data_load_types = []
            self._data_load_type_probabilities = []
            for data_type, p in self.dc_config.data_type_probabilities:
                self._data_load_types.append(data_type)
                self._data_load_type_probabilities.append(p)

            if dc_config.only_use_labels is not None and dc_config.use_object_pose:
                assert dc_config.pose_indeces is not None and len(
                    dc_config.pose_indeces
                ) == len(dc_config.only_use_labels), (
                    "Need pose indeces if using object pose and only_use_labels to "
                    "associate labels and object poses. Thus both need identical "
                    "length."
                )

        self.contrast_set = None

    def sample_traj_idx(self):
        return self.scene_data.sample_traj_idx()

    def sample_img_idx(self, traj_idx):
        return self.scene_data.sample_img_idx(traj_idx)

    @cached_property
    def object_labels(self):
        if self.dc_config.mask_type is MaskTypes.GT:
            labels = self.scene_data.object_labels_gt
        elif self.dc_config.mask_type is MaskTypes.TSDF:
            labels = self.scene_data.object_labels_tsdf
        else:
            labels = self.scene_data.object_labels

        if label_subset := self.dc_config.only_use_labels:
            for l in label_subset:
                assert l in labels, f"Label {l} not in dataset."
            logger.info(f"Only using labels {label_subset}")
            labels = label_subset

        return labels

    def get_random_object_label(self):
        idx = np.random.choice(range(len(self.object_labels)), 1)[0]
        label = self.object_labels[idx]

        return idx, label

    @lru_cache
    def object_labels_for_other_set(self, dataset):
        if self.dc_config.mask_type is MaskTypes.GT:
            return dataset.object_labels_gt
        elif self.dc_config.mask_type is MaskTypes.TSDF:
            return dataset.object_labels_tsdf
        else:
            return dataset.object_labels

    def get_random_object_label_for_other_set(self, dataset):
        labels = self.object_labels_for_other_set(dataset)
        idx = np.random.choice(range(len(labels)), 1)[0]
        label = self.object_labels[idx]

        return idx, label

    def get_two_different_object_labels(self):
        idx = np.random.choice(range(len(self.object_labels)), 2, replace=False)
        labels = [self.object_labels[i] for i in idx]

        return idx, labels

    def __len__(self):
        return self._len

    @property
    def no_obs(self):
        return self.scene_data.no_obs

    @cached_property
    def no_cams(self):
        return len(self.cameras)

    @cached_property
    def _len(self):
        t = self.sample_type

        if t is SampleTypes.CAM_SINGLE:
            return self.no_obs * self.no_cams
        elif t is SampleTypes.CAM_PAIR:
            return (self.no_obs * self.no_cams) ** 2
        elif t is SampleTypes.DC:
            # The actual number of combinations is bigger due to varying sample
            # types (single object, multi-object, etc.), contrast set and cams.
            # However, to precisely calculate would be tedious. Thus, only
            # pretend to follow dataloader format here and sample everything
            # but the image_a's index ourselves.
            return self.no_obs
        else:
            raise ValueError("Unexpected sample type {}.".format(t))

    def __getitem__(self, index: int):
        contr_cams = self.dc_config.contr_cam if self.dc_config else None
        return self._getitem(
            index, self.sample_type, cams=self.cameras, contr_cams=contr_cams
        )

    def _getitem(
        self,
        index: int,
        sample_type: SampleTypes,
        cams: tuple[str, ...] | None = None,
        contr_cams: tuple[str] | None = None,
    ) -> tuple | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if cams is None:
            cams = ("wrist",)
        if contr_cams is None:
            contr_cams = ("overhead",)

        if sample_type is SampleTypes.DC:
            traj_idx, img_idx = self.scene_data._index_split(index)

            return self.sample_dc(cams, contr_cams, traj_idx=traj_idx, img_idx=img_idx)

        elif sample_type is SampleTypes.CAM_SINGLE:
            cam_idx, obs_idx = index // self.no_obs, index % self.no_obs
            traj_idx, img_idx = self.scene_data._index_split(obs_idx)
            cam = cams[cam_idx]

            return self.sample_camera_single(traj_idx, img_idx, cam=cam)

        elif sample_type is SampleTypes.CAM_PAIR:
            cam_idx, obs_idx = index // self.no_obs**2, index % self.no_obs**2
            index1, index2 = obs_idx // self.no_obs, obs_idx % self.no_obs
            cam_idx1, cam_idx2 = (
                cam_idx // self.no_cams,
                cam_idx % self.no_cams,
            )  # noqa 501
            traj_idx1, img_idx1 = self.scene_data._index_split(index1)
            traj_idx2, img_idx2 = self.scene_data._index_split(index2)
            cam1 = cams[cam_idx1]
            cam2 = cams[cam_idx2]

            return self.sample_camera_pair(
                traj_idx1, img_idx1, cam1, traj_idx2, img_idx2, cam2
            )

        else:
            raise NotImplementedError

    def get_collate_func(self):
        # TODO: custom_collates for each. single: standard, pair: std on tuple,
        # DC: will see (look up encoder funcs)
        t = self.sample_type

        # default collate can handle single tensors and tuples of tensors
        if t in (SampleTypes.CAM_SINGLE, SampleTypes.CAM_PAIR):
            return default_collate
        elif t is SampleTypes.DC:
            # Currently the training function accepts a list of datapoints.
            # TODO: might this more elegant and true batched by collating.
            return lambda x: x
        else:
            raise ValueError("Unexpected sample type {}.".format(t))

    def sample_camera_single(
        self,
        traj_idx: int | None = None,
        img_idx: int | None = None,
        cam: str = "wrist",
    ) -> torch.Tensor:
        kwargs = {
            "mask_type": None,
            "raw_mask": True,
            "collapse_labels": False,
            "labels": None,
            "get_rgb": True,
            "get_int": False,
            "get_ext": False,
            "get_depth": False,
            "get_mask": False,
            "get_action": False,
            "get_feedback": False,
            "get_gripper_pose": False,
            "get_proprio_obs": False,
            "get_wrist_pose": False,
        }

        obs = self.scene_data.get_observation(
            traj_idx=traj_idx, img_idx=img_idx, cam=cam, **kwargs
        )

        return obs.rgb

    def sample_camera_pair(
        self,
        traj_idx1: int | None = None,
        img_idx1: int | None = None,
        cam1: str = "wrist",
        traj_idx2: int | None = None,
        img_idx2: int | None = None,
        cam2: str = "wrist",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {
            "mask_type": None,
            "raw_mask": True,
            "collapse_labels": False,
            "labels": None,
            "get_rgb": True,
            "get_int": False,
            "get_ext": False,
            "get_depth": False,
            "get_mask": False,
            "get_action": False,
            "get_feedback": False,
            "get_gripper_pose": False,
            "get_proprio_obs": False,
            "get_wrist_pose": False,
        }

        obs1 = self.scene_data.get_observation(
            traj_idx=traj_idx1, img_idx=img_idx1, cam=cam1, **kwargs
        )

        obs2 = self.scene_data.get_observation(
            traj_idx=traj_idx2, img_idx=img_idx2, cam=cam2, **kwargs
        )

        return obs1.rgb, obs2.rgb

    def sample_dc(
        self,
        cams: tuple[str],
        contr_cam: tuple[str] | None,
        traj_idx: int,
        img_idx: int,
    ) -> tuple:
        """
        Randomly chooses one of our different img pair types and camera conf,
        then returns that type of data.
        """

        cam_a = random.choice(cams)
        cam_b = random.choice(cams)

        data_load_type = self._get_data_load_type()

        # Case 0: Same scene, same object
        if data_load_type == DcDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE:
            if self._verbose:
                logger.info("Same scene, same object")
            return self.get_single_object_within_scene_data(
                cam_a=cam_a, cam_b=cam_b, traj_idx=traj_idx, img_idx=img_idx
            )

        # Case 1: Same object, different scene
        elif data_load_type == DcDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE:
            raise NotImplementedError
            # if self._verbose:
            #     logger.info("Same object, different scene")
            # return self.get_single_object_across_scene_data(
            #     cam_a=cam_a, cam_b=cam_b, traj_idx=traj_idx, img_idx=img_idx)

        # Case 2: Different object
        elif data_load_type == DcDatasetDataType.DIFFERENT_OBJECT:
            if self._verbose:
                logger.info("Different object")
            if type(contr_cam) is list:
                cam_b = random.choice(contr_cam)
            return self.get_different_object_data(
                cam_a=cam_a, cam_b=cam_b, traj_idx=traj_idx, img_idx=img_idx
            )

        # Case 3: Multi object
        elif data_load_type == DcDatasetDataType.MULTI_OBJECT:
            if self._verbose:
                logger.info("Multi object")
            return self.get_multi_object_within_scene_data(
                cam_a=cam_a, cam_b=cam_b, traj_idx=traj_idx, img_idx=img_idx
            )

        # Case 4: Synthetic multi object
        elif data_load_type == DcDatasetDataType.SYNTHETIC_MULTI_OBJECT:
            if self._verbose:
                logger.info("Synthetic multi object")
            return self.get_synthetic_multi_object_within_scene_data(
                cam_a="wrist", cam_b="wrist", traj_idx=traj_idx, img_idx=img_idx
            )

        else:
            raise ValueError("Unknown data type {}".format(data_load_type))

    def _get_data_load_type(self):
        """
        Gets a random data load type from the allowable types.
        """
        return np.random.choice(
            self._data_load_types, 1, p=self._data_load_type_probabilities
        )[0]

    def get_single_object_within_scene_data(
        self, cam_a: str, cam_b: str, traj_idx: int, img_idx: int
    ) -> tuple:
        """
        Simple wrapper around get_within_scene_data(), for the single object
        case.
        """
        if len(self.object_labels) == 0:
            raise ValueError("Found no labels in dataset.")

        if self.dc_config.only_use_first_object_label:
            logger.warning(
                "Using hardcoded object_no instead of sample. " "Is that intended?"
            )
            object_no = 0  # HACK for Lid to always pick lid_label
            object_label = self.object_labels[object_no]
            object_labels = [object_label]
        elif self.dc_config.conflate_so_object_labels:
            logger.warning(
                "Conflating object labels instead of sample.  "
                "Is that intended? "
                "Should only be needed for GT-masks and SO scenes."
            )
            object_no = 0  # HACK for Lid to use same pose for both objects
            object_labels = self.object_labels
        else:
            object_no, object_label = self.get_random_object_label()
            object_labels = [object_label]

        metadata = {
            "type": DcDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE,
            # TODO: add other metadata?
            # "traj_idx": traj_idx,
            # "img_idx": img_idx
        }

        return self.get_within_scene_data(
            traj_idx=traj_idx,
            img_idx=img_idx,
            object_labels=object_labels,
            metadata=metadata,
            cam_a=cam_a,
            cam_b=cam_b,
            object_no=object_no,
        )

    def get_different_object_data(
        self, cam_a: str, cam_b: str, traj_idx: int, img_idx: int
    ) -> tuple:
        """
        Simple wrapper around get_within_scene_data(), for the multi object
        case
        """
        # TODO: need object_no in cross_scene_data?
        obj_no_a, object_label_a = self.get_random_object_label()
        object_label_a = [object_label_a]

        traj_idx_a = traj_idx
        img_idx_a = img_idx

        if self.contrast_set is None:
            raise AttributeError("Need contrast set for different object sample.")

        object_label_b = self.contrast_set.object_labels
        traj_idx_b = self.contrast_set.scene_data.sample_traj_idx()
        img_idx_b = self.contrast_set.scene_data.sample_img_idx(traj_idx_b)

        metadata = {
            "type": DcDatasetDataType.DIFFERENT_OBJECT,
        }

        return self.get_across_scene_data(
            traj_idx_a=traj_idx_a,
            img_idx_a=img_idx_a,
            cam_a=cam_a,
            traj_idx_b=traj_idx_b,
            img_idx_b=img_idx_b,
            cam_b=cam_b,
            object_label_a=object_label_a,
            object_label_b=object_label_b,
            set_b=self.contrast_set.scene_data,
            metadata=metadata,
        )

    def get_multi_object_within_scene_data(
        self, cam_a: str, cam_b: str, traj_idx: int, img_idx: int
    ) -> tuple:
        """
        Simple wrapper around get_within_scene_data(), for the multi object
        case
        """
        if len(self.object_labels) < 2:
            raise ValueError("Found less than two labels in dataset.")

        # HACK: object label is assignment in rlbench is a bit odd.
        # Microwave is composed of multiple labels (84, 87) and I want it to
        # be treated as one object. Phone also has two labels (82, 84) but I
        # consider them two object. I have implemented different ways to handle
        # labels, seee scene_data.get_rgbd_mask_pose_intrinsics. Wrapping the
        # single label here in a list, will differentiate between the labels.
        # Need to verify that this works for other tasks as well and fix if
        # needed.
        object_no, object_label = self.get_random_object_label()
        object_labels = [object_label]

        # traj_idx = traj_idx or self.sample_traj_idx()
        # img_idx = img_idx or self.sample_img_idx(traj_idx)

        metadata = {
            "type": DcDatasetDataType.MULTI_OBJECT,
        }

        # TODO: either trust that enough non-matches are sampled on the other
        # object OR specifically sample from the other object's mask.
        # ATM there's technically no difference between this and single object
        # training besides the number of object masks to sample from.

        return self.get_within_scene_data(
            traj_idx=traj_idx,
            img_idx=img_idx,
            object_labels=object_labels,
            metadata=metadata,
            cam_a=cam_a,
            cam_b=cam_b,
            object_no=object_no,
        )

    def get_dc_obs(
        self,
        traj_idx: int,
        img_idx: int,
        object_labels: Iterable[int] | None = (1,),
        cam: str = "wrist",
        dataset: Dataset | None = None,
    ) -> SingleCamObservation:
        if dataset is None:
            dataset = self.scene_data

        obs = dataset.get_observation(
            traj_idx=traj_idx,
            img_idx=img_idx,
            cam=cam,
            mask_type=self.dc_config.mask_type,
            raw_mask=False,
            labels=object_labels,
            collapse_labels=True,
            get_rgb=True,
            get_int=True,
            get_ext=True,
            get_depth=True,
            get_mask=True,
            get_action=False,
            get_feedback=False,
            get_gripper_pose=False,
            get_object_poses=self.dc_config.use_object_pose,
            get_proprio_obs=False,
            get_wrist_pose=False,
        )

        # stack the needed object poses into one tensor
        if (idcs := self.dc_config.pose_indeces) is not None:
            assert self.dc_config.only_use_labels is not None
            object_poses = torch.stack([v for _, v in obs.object_poses.items()])
            obs.object_poses = object_poses[idcs]

        return obs

    def get_ext(
        self, traj_idx: int, img_idx: int, cam: str = "wrist", dataset: Dataset = None
    ) -> SingleCamObservation:
        if dataset is None:
            dataset = self.scene_data

        obs = dataset.get_observation(
            traj_idx=traj_idx,
            img_idx=img_idx,
            cam=cam,
            mask_type=self.dc_config.mask_type,
            raw_mask=False,
            labels=None,
            collapse_labels=True,
            get_rgb=False,
            get_int=False,
            get_ext=True,
            get_depth=False,
            get_mask=False,
            get_action=False,
            get_feedback=False,
            get_gripper_pose=False,
            get_object_pose=self.dc_config.use_object_pose,
            get_proprio_obs=False,
            get_wrist_pose=False,
        )

        return obs.extr

    def get_within_scene_data(
        self,
        traj_idx: int,
        img_idx: int,
        object_labels: list[int],
        cam_a: str = "wrist",
        cam_b: str = "wrist",
        object_no: int | None = None,
        metadata: dict = None,
        for_synthetic_multi_object: bool = False,
        dataset: Dataset | None = None,
    ) -> tuple:
        if dataset is None:
            dataset = self.scene_data

        obs_a = self.get_dc_obs(
            traj_idx, img_idx, object_labels=object_labels, cam=cam_a, dataset=dataset
        )

        # Get a second frame which is 'different enough' from the first one.
        # Uses the camera pose to determine that.
        # Technically, this only makes sense for the wrist cam, as the other
        # cameras are static. However, when we pretrain on trajectories,
        # 'different' cam also mean that the robot arm is now in the way (or at
        # another position), even though the overhead cam is still at the same
        # position. If cam_a and cam_b are different, even the same index would
        # be ok, so pass cam_b then. Else pass "wrist" and wrist_ext to get a
        # a differnt observation from the trajectory.
        if cam_b == cam_a and cam_a != "wrist":
            pose_for_pose_dif = self.get_ext(traj_idx, img_idx, cam="wrist")
            cam_for_pose_dif = "wrist"
        else:
            pose_for_pose_dif = obs_a.extr
            cam_for_pose_dif = cam_b

        image_b_idx = dataset.get_img_idx_with_different_pose(
            traj_idx, pose_for_pose_dif, num_attempts=50, cam=cam_for_pose_dif
        )

        metadata["image_b_idx"] = image_b_idx

        if image_b_idx is None:
            if self._verbose:
                logger.info(
                    "No frame with sufficiently different pose found," " returning."
                )
            if for_synthetic_multi_object:
                return self.return_empty_data_for_smo(obs_a.rgb, obs_a.rgb)
            else:
                return self.return_empty_data(obs_a.rgb, obs_a.rgb)

        obs_b = self.get_dc_obs(
            traj_idx,
            image_b_idx,
            object_labels=object_labels,
            cam=cam_b,
            dataset=dataset,
        )

        if self.dc_config.sample_matches_only_off_mask:
            correspondence_mask = obs_a.mask
        else:
            correspondence_mask = None

        # find correspondences
        uv_a, uv_b = correspondence_finder.batch_find_pixel_correspondences(
            obs_a.depth,
            obs_a.extr,
            obs_b.depth,
            obs_b.extr,
            img_a_mask=correspondence_mask,
            num_attempts=self.dc_config.num_matching_attempts,
            K_a=obs_a.intr,
            K_b=obs_b.intr,
            obj_pose_a=obs_a.object_poses,
            obj_pose_b=obs_b.object_poses,
            object_no=object_no,
        )

        if for_synthetic_multi_object:
            return (
                obs_a.rgb,
                obs_b.rgb,
                obs_a.depth,
                obs_b.depth,
                obs_a.mask,
                obs_b.mask,
                uv_a,
                uv_b,
            )

        if uv_a is None:
            if self._verbose:
                logger.info("No matches found, returning.")
            return self.return_empty_data(obs_a.rgb, obs_a.rgb)

        (
            image_a_rgb,
            image_a_mask,
            image_a_depth,
            image_b_rgb,
            image_b_mask,
            image_b_depth,
            uv_a,
            uv_b,
        ) = self._image_augment(obs_a, obs_b, uv_a, uv_b)

        if uv_a is None:
            if self._verbose:
                logger.info("No matches left after image augment, returning.")
            return self.return_empty_data(obs_a.rgb, obs_b.rgb)

        # find non_correspondences
        image_b_shape = image_b_depth.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        # calculate the number of fg/bg non-matches depending on mask size
        if self.num_masked_non_matches_per_match == "auto":
            rel_mask_size = image_b_mask.sum() / image_b_mask.numel()
            num_masked_non_matches = int(
                rel_mask_size * self.dc_config.num_non_matches_per_match
            )
            num_background_non_matches = int(
                (1 - rel_mask_size) * self.dc_config.num_non_matches_per_match
            )
        else:
            num_masked_non_matches = self.num_masked_non_matches_per_match
            num_background_non_matches = self.num_background_non_matches_per_match

        uv_b_masked_non_matches = correspondence_finder.create_non_correspondences(
            uv_b,
            image_b_shape,
            num_non_matches_per_match=num_masked_non_matches,
            img_b_mask=image_b_mask,
        )

        if self.dc_config._use_image_b_mask_inv:
            image_b_mask_inv = 1 - image_b_mask
        else:
            image_b_mask_inv = None

        uv_b_background_non_matches = correspondence_finder.create_non_correspondences(
            uv_b,
            image_b_shape,
            num_non_matches_per_match=num_background_non_matches,
            img_b_mask=image_b_mask_inv,
        )

        matches_a = self.flatten_uv_tensor(uv_a, image_width)
        matches_b = self.flatten_uv_tensor(uv_b, image_width)

        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(
            uv_a, uv_b_masked_non_matches, num_masked_non_matches
        )

        masked_non_matches_a = self.flatten_uv_tensor(
            uv_a_masked_long, image_width
        ).squeeze(1)

        if uv_b_masked_non_matches_long is None:
            if self._verbose:
                logger.info("No masked non-matches found, returning.")
            return self.return_empty_data(obs_a.rgb, obs_b.rgb)
            # masked_non_matches_b = None
        else:
            masked_non_matches_b = self.flatten_uv_tensor(
                uv_b_masked_non_matches_long, image_width
            ).squeeze(1)

        # Non-masked non-matches
        (
            uv_a_background_long,
            uv_b_background_non_matches_long,
        ) = self.create_non_matches(
            uv_a, uv_b_background_non_matches, num_background_non_matches
        )

        background_non_matches_a = self.flatten_uv_tensor(
            uv_a_background_long, image_width
        ).squeeze(1)
        background_non_matches_b = self.flatten_uv_tensor(
            uv_b_background_non_matches_long, image_width
        )

        if background_non_matches_b is None:
            if self._verbose:
                logger.info("No background non-matches found, returning.")
            return self.return_empty_data(obs_a.rgb, obs_b.rgb)
        else:
            background_non_matches_b = background_non_matches_b.squeeze(1)

        # make blind non matches
        matches_a_mask = self.mask_image_from_uv_flat_tensor(
            matches_a, image_width, image_height
        )
        mask_a_flat = image_a_mask.long().view(-1, 1).squeeze(1)
        blind_non_matches_a = torch.nonzero(mask_a_flat - matches_a_mask)

        no_blind_matches_found = False

        if len(blind_non_matches_a) == 0:
            no_blind_matches_found = True
        else:
            blind_non_matches_a = blind_non_matches_a.squeeze(1)
            num_blind_samples = blind_non_matches_a.size()[0]

            if num_blind_samples > 0:
                # blind_uv_b is a tuple of torch.LongTensor
                # make sure we check that blind_uv_b is not None and that
                # it is non-empty

                blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(  # noqa 501
                    image_b_mask, num_blind_samples
                )

                if blind_uv_b[0] is None:
                    no_blind_matches_found = True
                elif len(blind_uv_b[0]) == 0:
                    no_blind_matches_found = True
                else:
                    blind_non_matches_b = uv_to_flattened_pixel_locations(
                        blind_uv_b, image_width
                    )

                    if len(blind_non_matches_b) == 0:
                        no_blind_matches_found = True
            else:
                no_blind_matches_found = True

        if no_blind_matches_found:
            blind_non_matches_a = self.empty_tensor()
            blind_non_matches_b = self.empty_tensor()

            blind_uv_b = None

        if self.dc_config.debug:
            debug_plots(
                channel_front2back(image_a_rgb),
                channel_front2back(image_b_rgb),
                image_height,
                image_a_depth,
                image_b_depth,
                image_a_mask,
                image_b_mask,
                matches_a_mask,
                mask_a_flat,
                uv_a,
                uv_b,
                uv_a_masked_long,
                uv_b_masked_non_matches_long,
                uv_a_background_long,
                uv_b_background_non_matches_long,
                blind_non_matches_a,
                image_width,
                blind_uv_b,
            )

        return (
            metadata["type"],
            image_a_rgb,
            image_b_rgb,
            matches_a,
            matches_b,
            masked_non_matches_a,
            masked_non_matches_b,
            background_non_matches_a,
            background_non_matches_b,
            blind_non_matches_a,
            blind_non_matches_b,
            metadata,
        )

    def get_across_scene_data(
        self,
        traj_idx_a,
        img_idx_a,
        object_label_a,
        traj_idx_b,
        img_idx_b,
        object_label_b,
        set_b,
        metadata=None,
        cam_a="wrist",
        cam_b="overhead",
    ):
        """
        Essentially just returns a bunch of samples off the masks from
        scene_name_a, and scene_name_b.

        Since this data is across scene, we can't generate matches.

        Return args are for returning directly from __getitem__

        See get_within_scene_data() for documentation of return args.
        """

        obs_a = self.get_dc_obs(
            traj_idx_a, img_idx_a, object_labels=object_label_a, cam=cam_a
        )

        obs_b = self.get_dc_obs(
            traj_idx_b,
            img_idx_b,
            dataset=set_b,
            object_labels=object_label_b,
            cam=cam_b,
        )

        # sample random indices from masks in both images
        num_samples = self.dc_config.cross_scene_num_samples
        blind_uv_a = correspondence_finder.random_sample_from_masked_image_torch(
            obs_a.mask, num_samples
        )
        blind_uv_b = correspondence_finder.random_sample_from_masked_image_torch(
            obs_b.mask, num_samples
        )

        if (blind_uv_a[0] is None) or (blind_uv_b[0] is None):
            return self.return_empty_data(obs_a.rgb, obs_a.rgb)

        (
            image_a_rgb,
            _,
            image_a_depth,
            image_b_rgb,
            _,
            image_b_depth,
            blind_uv_a,
            blind_uv_b,
        ) = self._image_augment(obs_a, obs_b, blind_uv_a, blind_uv_b)

        image_b_shape = image_b_depth.shape
        image_width = image_b_shape[1]

        blind_uv_a_flat = self.flatten_uv_tensor(blind_uv_a, image_width)
        blind_uv_b_flat = self.flatten_uv_tensor(blind_uv_b, image_width)

        empty_tensor = self.empty_tensor()

        if self.dc_config.debug and (
            (blind_uv_a[0] is not None) and (blind_uv_b[0] is not None)
        ):
            cross_debug_plot(
                channel_front2back(image_a_rgb),
                channel_front2back(image_b_rgb),
                image_a_depth,
                image_b_depth,
                blind_uv_a,
                blind_uv_b,
            )

        return (
            metadata["type"],
            image_a_rgb,
            image_b_rgb,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            empty_tensor,
            blind_uv_a_flat,
            blind_uv_b_flat,
            metadata,
        )

    def get_synthetic_multi_object_within_scene_data(
        self,
        cam_a: str,
        cam_b: str,
        traj_idx: int | None = None,
        img_idx: int | None = None,
    ) -> tuple:
        if self.scene_data.smo_order is None:
            dataset_keys = list(self.scene_data.smo_data.keys())
            dataset_a_idx, dataset_b_idx = random.sample(dataset_keys, 2)
        else:
            dataset_a_idx, dataset_b_idx = self.scene_data.smo_order

        dataset_a = self.scene_data.smo_data[dataset_a_idx]
        dataset_b = self.scene_data.smo_data[dataset_b_idx]

        object_no_a, object_label_a = self.get_random_object_label_for_other_set(
            dataset_a
        )
        object_no_b, object_label_b = self.get_random_object_label_for_other_set(
            dataset_b
        )
        object_label_a = [object_label_a]
        object_label_b = [object_label_b]

        metadata = {
            "type": DcDatasetDataType.SYNTHETIC_MULTI_OBJECT,
        }

        if self.scene_data.smo_order is None or traj_idx is None:
            traj_idx_a = dataset_a.sample_traj_idx()
        else:
            traj_idx_a = traj_idx
        if self.scene_data.smo_order is None or img_idx is None:
            img_idx_a = dataset_a.sample_img_idx(traj_idx_a)
        else:
            img_idx_a = img_idx
        traj_idx_b = dataset_b.sample_traj_idx()
        img_idx_b = dataset_b.sample_img_idx(traj_idx_b)

        (
            image_a1_rgb,
            image_a2_rgb,
            image_a1_depth,
            image_a2_depth,
            image_a1_mask,
            image_a2_mask,
            uv_a1,
            uv_a2,
        ) = self.get_within_scene_data(
            traj_idx_a,
            img_idx_a,
            object_label_a,
            cam_a,
            cam_b,
            object_no_a,
            metadata,
            for_synthetic_multi_object=True,
            dataset=dataset_a,
        )

        if uv_a1 is None:
            logger.info("no matches found, returning")
            return self.return_empty_data(image_a1_rgb, image_a1_rgb)

        (
            image_b1_rgb,
            image_b2_rgb,
            image_b1_depth,
            image_b2_depth,
            image_b1_mask,
            image_b2_mask,
            uv_b1,
            uv_b2,
        ) = self.get_within_scene_data(
            traj_idx_b,
            img_idx_b,
            object_label_b,
            cam_a,
            cam_b,
            object_no_b,
            metadata,
            for_synthetic_multi_object=True,
            dataset=dataset_b,
        )

        if uv_b1 is None:
            logger.info("no matches found, returning")
            return self.return_empty_data(image_b1_rgb, image_b1_rgb)

        uv_a1 = (uv_a1[0].long(), uv_a1[1].long())
        uv_a2 = (uv_a2[0].long(), uv_a2[1].long())
        uv_b1 = (uv_b1[0].long(), uv_b1[1].long())
        uv_b2 = (uv_b2[0].long(), uv_b2[1].long())

        matches_pair_a = (uv_a1, uv_a2)
        matches_pair_b = (uv_b1, uv_b2)
        (
            merged_rgb_1,
            merged_mask_1,
            merged_depth_1,
            uv_a1,
            uv_a2,
            uv_b1,
            uv_b2,
        ) = correspondence_augmentation.merge_images_with_occlusions(
            image_a1_rgb,
            image_b1_rgb,
            image_a1_mask,
            image_b1_mask,
            image_a1_depth,
            image_a2_depth,
            matches_pair_a,
            matches_pair_b,
        )

        if (uv_a1 is None) or (uv_a2 is None) or (uv_b1 is None) or (uv_b2 is None):
            logger.info("some object got fully occluded, returning")
            return self.return_empty_data(image_b1_rgb, image_b1_rgb)

        matches_pair_a = (uv_a2, uv_a1)
        matches_pair_b = (uv_b2, uv_b1)
        (
            merged_rgb_2,
            merged_mask_2,
            merged_depth_2,
            uv_a2,
            uv_a1,
            uv_b2,
            uv_b1,
        ) = correspondence_augmentation.merge_images_with_occlusions(
            image_a2_rgb,
            image_b2_rgb,
            image_a2_mask,
            image_b2_mask,
            image_b1_depth,
            image_b2_depth,
            matches_pair_a,
            matches_pair_b,
        )

        if (uv_a1 is None) or (uv_a2 is None) or (uv_b1 is None) or (uv_b2 is None):
            logger.info("some object got fully occluded, returning")
            return self.return_empty_data(image_b1_rgb, image_b1_rgb)

        matches_1 = correspondence_augmentation.merge_matches(uv_a1, uv_b1)
        matches_2 = correspondence_augmentation.merge_matches(uv_a2, uv_b2)
        matches_2 = (matches_2[0].float(), matches_2[1].float())

        # find non_correspondences
        image_b_shape = merged_mask_2.shape
        image_width = image_b_shape[1]
        image_height = image_b_shape[0]

        # calculate the number of fg/bg non-matches depending on mask size
        if self.num_masked_non_matches_per_match == "auto":
            rel_mask_size = merged_mask_2.sum() / merged_mask_2.numel()
            num_masked_non_matches = int(
                rel_mask_size * self.dc_config.num_non_matches_per_match
            )
            num_background_non_matches = int(
                (1 - rel_mask_size) * self.dc_config.num_non_matches_per_match
            )
        else:
            num_masked_non_matches = self.num_masked_non_matches_per_match
            num_background_non_matches = self.num_background_non_matches_per_match

        matches_2_masked_non_matches = correspondence_finder.create_non_correspondences(
            matches_2,
            image_b_shape,
            num_non_matches_per_match=num_masked_non_matches,
            img_b_mask=merged_mask_2,
        )

        if self.dc_config._use_image_b_mask_inv:
            merged_mask_2_torch_inv = 1 - merged_mask_2
        else:
            merged_mask_2_torch_inv = None

        matches_2_background_non_matches = (
            correspondence_finder.create_non_correspondences(
                matches_2,
                image_b_shape,
                num_non_matches_per_match=num_background_non_matches,
                img_b_mask=merged_mask_2_torch_inv,
            )
        )

        matches_a = self.flatten_uv_tensor(matches_1, image_width)
        matches_b = self.flatten_uv_tensor(matches_2, image_width)

        # Masked non-matches
        uv_a_masked_long, uv_b_masked_non_matches_long = self.create_non_matches(
            matches_1, matches_2_masked_non_matches, num_masked_non_matches
        )

        if uv_b_masked_non_matches_long is None:
            return self.return_empty_data(merged_rgb_1, merged_rgb_2)

        masked_non_matches_a = self.flatten_uv_tensor(
            uv_a_masked_long, image_width
        ).squeeze(1)
        masked_non_matches_b = self.flatten_uv_tensor(
            uv_b_masked_non_matches_long, image_width
        ).squeeze(1)

        # Non-masked non-matches
        (
            uv_a_background_long,
            uv_b_background_non_matches_long,
        ) = self.create_non_matches(
            matches_1, matches_2_background_non_matches, num_background_non_matches
        )

        background_non_matches_a = self.flatten_uv_tensor(
            uv_a_background_long, image_width
        ).squeeze(1)

        if uv_b_background_non_matches_long is None:
            if self._verbose:
                logger.info("No masked non-matches found, returning.")
            return self.return_empty_data(image_a1_rgb, image_b1_rgb)
            # masked_non_matches_b = None
        else:
            background_non_matches_b = self.flatten_uv_tensor(
                uv_b_background_non_matches_long, image_width
            ).squeeze(1)

        blind_non_matches_a = self.empty_tensor()
        blind_non_matches_b = self.empty_tensor()

        if self.dc_config.debug:
            blind_uv_b = None

            matches_a_mask = self.mask_image_from_uv_flat_tensor(
                matches_a, image_width, image_height
            )
            mask_a_flat = merged_mask_1.long().view(-1, 1).squeeze(1)

            debug_plots(
                channel_front2back(merged_rgb_1),
                channel_front2back(merged_rgb_2),
                image_height,
                merged_depth_1,
                merged_depth_2,
                merged_mask_1,
                merged_mask_2,
                matches_a_mask,
                mask_a_flat,
                matches_1,
                matches_2,
                uv_a_masked_long,
                uv_b_masked_non_matches_long,
                uv_a_background_long,
                uv_b_background_non_matches_long,
                blind_non_matches_a,
                image_width,
                blind_uv_b,
            )

        return (
            metadata["type"],
            merged_rgb_1,
            merged_rgb_2,
            matches_a,
            matches_b,
            masked_non_matches_a,
            masked_non_matches_b,
            background_non_matches_a,
            background_non_matches_b,
            blind_non_matches_a,
            blind_non_matches_b,
            metadata,
        )

    def _image_augment(self, obs_a, obs_b, uv_a, uv_b):
        image_a_rgb = obs_a.rgb
        image_a_mask = obs_a.mask
        image_a_depth = obs_a.depth
        image_b_rgb = obs_b.rgb
        image_b_mask = obs_b.mask
        image_b_depth = obs_b.depth

        if self.dc_config.domain_randomize:
            image_a_rgb = (
                correspondence_augmentation.random_domain_randomize_background(
                    image_a_rgb, image_a_mask
                )
            )
            image_b_rgb = (
                correspondence_augmentation.random_domain_randomize_background(
                    image_b_rgb, image_b_mask
                )
            )

        if not self.dc_config.debug:
            if self.dc_config.random_flip:
                [image_a_rgb], uv_a = correspondence_augmentation.random_flip(
                    [image_a_rgb], uv_a
                )
                # mutate mask b too for sampling non-correspondences.
                [
                    image_b_rgb,
                    image_b_mask,
                ], uv_b = correspondence_augmentation.random_flip(
                    [image_b_rgb, image_b_mask], uv_b
                )
            if self.dc_config.random_crop:
                (
                    [image_a_rgb],
                    [image_b_rgb, image_b_mask],
                    uv_a,
                    uv_b,
                ) = correspondence_augmentation.random_apply_random_crop(
                    [image_a_rgb],
                    [image_b_rgb, image_b_mask],
                    uv_a,
                    uv_b,
                    self.dc_config.crop_size,
                    self.dc_config.sample_crop_size,
                )
        else:
            if (
                self.dc_config.random_flip
            ):  # also mutate depth, mask a just for plotting
                [
                    image_a_rgb,
                    image_a_mask,
                    image_a_depth,
                ], uv_a = correspondence_augmentation.random_flip(
                    [image_a_rgb, image_a_mask, image_a_depth], uv_a
                )
                [
                    image_b_rgb,
                    image_b_mask,
                    image_b_depth,
                ], uv_b = correspondence_augmentation.random_flip(
                    [image_b_rgb, image_b_mask, image_b_depth], uv_b
                )
            if self.dc_config.random_crop:
                (
                    [image_a_rgb, image_a_mask, image_a_depth],
                    [image_b_rgb, image_b_mask, image_b_depth],
                    uv_a,
                    uv_b,
                ) = correspondence_augmentation.random_apply_random_crop(
                    [image_a_rgb, image_a_mask, image_a_depth],
                    [image_b_rgb, image_b_mask, image_b_depth],
                    uv_a,
                    uv_b,
                    self.dc_config.crop_size,
                    self.dc_config.sample_crop_size,
                )
        return (
            image_a_rgb,
            image_a_mask,
            image_a_depth,
            image_b_rgb,
            image_b_mask,
            image_b_depth,
            uv_a,
            uv_b,
        )

    @staticmethod
    def empty_tensor():
        """
        Makes a placeholder tensor
        :return:
        :rtype:
        """
        return torch.LongTensor([-1])

    @staticmethod
    def is_empty(tensor):
        """
        Tells if the tensor is the same as that created by empty_tensor()
        """
        return (tuple(tensor.shape) == (1,)) and (tensor[0] == -1)

    @staticmethod
    def return_empty_data(image_a_rgb, image_b_rgb, metadata=None):
        if metadata is None:
            metadata = {}

        empty = DenseCorrespondenceDataset.empty_tensor()
        return (
            -1,
            image_a_rgb,
            image_b_rgb,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            metadata,
        )

    @staticmethod
    def return_empty_data_for_smo(image_a_rgb, image_b_rgb):
        return image_a_rgb, image_b_rgb, None, None, None, None, None, None

    @staticmethod
    def create_non_matches(uv_a, uv_b_non_matches, multiplier):
        """
        Simple wrapper for repeated code
        :param uv_a:
        :type uv_a:
        :param uv_b_non_matches:
        :type uv_b_non_matches:
        :param multiplier:
        :type multiplier:
        :return:
        :rtype:
        """
        uv_a_long = (
            torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
            torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1),
        )

        if uv_b_non_matches is None:
            uv_b_non_matches_long = None

        else:
            uv_b_non_matches_long = (
                uv_b_non_matches[0].view(-1, 1),
                uv_b_non_matches[1].view(-1, 1),
            )

        return uv_a_long, uv_b_non_matches_long

    @staticmethod
    def flatten_uv_tensor(uv_tensor, image_width):
        """
        Flattens a uv_tensor to single dimensional tensor
        :param uv_tensor:
        :type uv_tensor:
        :return:
        :rtype:
        """
        if uv_tensor is None:
            return None
        else:
            return uv_tensor[1].long() * image_width + uv_tensor[0].long()

    @staticmethod
    def mask_image_from_uv_flat_tensor(uv_flat_tensor, image_width, image_height):
        """
        Returns a torch.LongTensor with shape [image_width*image_height]
         It has a 1 exactly at the indices specified by uv_flat_tensor.
        :param uv_flat_tensor:
        :type uv_flat_tensor:
        :param image_width:
        :type image_width:
        :param image_height:
        :type image_height:
        :return:
        :rtype:
        """
        image_flat = torch.zeros(image_width * image_height).long()
        image_flat[uv_flat_tensor] = 1
        return image_flat

    def sample_single_image(
        self,
        cameras: tuple[str] = ("wrist",),
        dataset: Dataset | None = None,
        traj_idx: int | None = None,
        img_idx: int | None = None,
    ) -> SingleCamObservation:
        traj_idx = traj_idx or self.sample_traj_idx()
        img_idx = img_idx or self.scene_data.sample_img_idx(traj_idx)
        cam = random.choice(cameras)

        if dataset is None:
            dataset = self.scene_data

        obs = dataset.get_observation(
            traj_idx=traj_idx,
            img_idx=img_idx,
            cam=cam,
            mask_type=None,
            raw_mask=False,
            labels=None,
            collapse_labels=True,
            get_rgb=True,
            get_int=False,
            get_ext=False,
            get_depth=False,
            get_mask=False,
            get_action=False,
            get_feedback=False,
            get_gripper_pose=False,
            get_proprio_obs=False,
            get_wrist_pose=False,
        )

        return obs.rgb

    def sample_image_tensor(
        self, no_samples: int, cameras: tuple[str] = ("wrist",)
    ) -> torch.FloatTensor:
        imgs = [self.sample_single_image(cameras=cameras) for _ in range(no_samples)]

        return torch.stack(imgs)

    def estimate_image_mean_and_std(
        self, num_image_samples: int = 10, cam: tuple[str] = ("wrist",)
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Estimate the image_mean and std_dev by sampling from scene images.
        Returns two torch.FloatTensor objects, each of size [3]

        Parameters
        ----------
        num_image_samples : int, optional
            Number of samples, by default 10
        cam : tuple[str], optional
            The cameras to sample from, by default ("wrist", )

        Returns
        -------
        tuple[torch.FloatTensor, torch.FloatTensor]
            Sample mean and std_dev.
        """

        # have channel in front now, so keep dim -3
        samples = self.sample_image_tensor(num_image_samples, cameras=cam)
        sample_mean = get_image_tensor_mean(samples, dims_to_keep=(-3,))
        sample_std = get_image_tensor_std(samples, dims_to_keep=(-3,))

        return sample_mean, sample_std

    def sample_data_pair(
        self,
        cross_scene=False,
        contrast_obj=False,
        cam_a="wrist",
        cam_b="wrist",
        contrast_cam="overhead",
        contrast_fraction=0,
        first_image_only=False,
    ):
        # This is used for the live heatmap visualization, which only
        # needs the image, not the masks.
        traj_idx = self.sample_traj_idx()
        image_a_idx = 0 if first_image_only else self.sample_img_idx(traj_idx)

        kwargs = {
            "mask_type": None,
            "raw_mask": True,
            "collapse_labels": False,
            "labels": None,
            "get_rgb": True,
            "get_int": False,
            "get_ext": True,
            "get_depth": False,
            "get_mask": False,
            "get_action": False,
            "get_feedback": False,
            "get_gripper_pose": False,
            "get_proprio_obs": False,
            "get_wrist_pose": False,
        }

        obs_a = self.scene_data.get_observation(
            traj_idx=traj_idx, img_idx=image_a_idx, cam=cam_a, **kwargs
        )

        use_contrast_image = contrast_obj and bool(
            np.random.binomial(1, contrast_fraction)
        )

        if use_contrast_image:
            traj_idx = self.contrast_set.sample_traj_idx()
            image_b_idx = (
                0
                if first_image_only
                else self.contrast_set.scene_data.sample_img_idx(traj_idx)
            )
            obs_b = self.contrast_set.scene_data.get_observation(
                traj_idx=traj_idx, img_idx=image_b_idx, cam=cam_b, **kwargs
            )
        else:
            if cross_scene:
                traj_idx = self.sample_traj_idx()
                image_b_idx = 0 if first_image_only else self.sample_img_idx(traj_idx)
            else:
                image_b_idx = (
                    0
                    if first_image_only
                    else self.scene_data.get_img_idx_with_different_pose(
                        traj_idx, obs_a.extr, num_attempts=50
                    )
                )

            obs_b = self.scene_data.get_observation(
                traj_idx=traj_idx, img_idx=image_b_idx, cam=cam_b, **kwargs
            )

        return obs_a, obs_b
