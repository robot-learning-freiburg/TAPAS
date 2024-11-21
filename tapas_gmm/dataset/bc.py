from dataclasses import dataclass, field

import numpy as np
import torch
from loguru import logger
from omegaconf import MISSING, OmegaConf
from torch.utils.data import Dataset

from tapas_gmm.utils.logging import log_constructor
from tapas_gmm.utils.observation import MaskTypes, SingleCamObservation, collate


@dataclass
class BCDataConfig:
    fragment_length: int

    pre_padding: int = 0
    post_padding: int = 0

    cameras: tuple[str, ...] | None = None
    pre_embedding: bool = False
    kp_pre_encoding: str | None = None
    encoder_name: str | None = None

    mask_type: MaskTypes = MaskTypes.TSDF
    only_use_labels: tuple[int] | None = None

    extra_attr: dict = field(default_factory=dict)

    force_load_raw: bool = False  # only needed for PF when not pre-encoding
    debug_encoding: bool = False
    force_skip_rgb: bool = False  # only needed for GT-KP encoder

    sample_freq: int | None = None

    subsample_to_common_length: bool = False


class BCDataset(Dataset):
    @log_constructor
    def __init__(self, scene_dataset, config: BCDataConfig):
        self.scene_data = scene_dataset
        self.config = config

        self.fragment_length = config.fragment_length
        self.pre_padding = config.pre_padding
        self.post_padding = config.post_padding

        if (T := self.fragment_length) is None:
            logger.info("Training on full trajectories. Padding.")
        else:
            logger.info("Training on fragments of length {}.", format(T))

        self.object_labels = self.get_object_labels()

        self.load_pre_embedding = ["descriptor"] if config.pre_embedding else []
        self.encoder_name = config.encoder_name

        self.pre_encoding_attr = ["kp"] if config.kp_pre_encoding else []
        self.pre_encoding_name = enc if (enc := config.kp_pre_encoding) else None

        self.extra_attr = config.extra_attr

        if self.pre_encoding_name:
            logger.info(
                "Loading attr {} from encoding {} from tapas_gmm.encoder named {}.",
                self.pre_encoding_attr,
                self.pre_encoding_name,
                self.encoder_name,
            )
        elif self.load_pre_embedding:
            logger.info(
                "Loading embedding {} from tapas_gmm.encoder named {}.",
                self.load_pre_embedding,
                self.encoder_name,
            )
        else:
            logger.info("Loading raw data for encoder.")

    @property
    def _full_traj_mode(self) -> bool:
        return self.fragment_length is None or self.fragment_length == -1

    @property
    def _unpadded_traj_lens(self) -> np.ndarray:
        return np.array(self.scene_data._traj_lens)

    @property
    def _traj_lens(self) -> np.ndarray:
        return self._unpadded_traj_lens + self.pre_padding + self.post_padding

    @property
    def _traj_segment_sample_end_idcs(self) -> np.ndarray:
        return np.cumsum(self._traj_lens - self.fragment_length + 1)

    @property
    def _traj_segment_sample_start_idcs(self) -> np.ndarray:
        return np.concatenate([[0], self._traj_segment_sample_end_idcs[:-1]])

    @property
    def n_trajs(self) -> int:
        return len(self.scene_data)

    def __len__(self) -> int:
        if self._full_traj_mode:
            return self.n_trajs
        else:
            return self._traj_segment_sample_end_idcs[-1]

    @property
    def _traj_start_idcs(self) -> np.ndarray:
        return np.cumsum([0] + list(self._traj_lens)[:-1])

    @property
    def _traj_end_idcs(self) -> np.ndarray:
        return np.cumsum(self._traj_lens)

    def _split_idx(self, idx: int) -> tuple[int, int]:
        """
        Given a global index, return the trajectory index and segment index.
        """
        assert 0 <= idx < len(self)

        if self._full_traj_mode:
            traj_idx, obs_idx = idx, 0
        else:
            traj_idx = np.searchsorted(
                self._traj_segment_sample_end_idcs, idx, side="right"
            )
            traj_start = self._traj_segment_sample_start_idcs[traj_idx]
            obs_idx = idx - traj_start

        return traj_idx, obs_idx

    def __getitem__(self, index: int):
        # TODO: the force_load/skip business is very HACK-y
        force_load_raw = self.config.force_load_raw
        debug_encoding = self.config.debug_encoding
        force_skip_rgb = self.config.force_skip_rgb

        assert not force_load_raw or not force_skip_rgb

        traj_idx, obs_idx = self._split_idx(index)

        needed_pre_padding = max(0, self.pre_padding - obs_idx)

        scene_obs_start = obs_idx - self.pre_padding

        if self._full_traj_mode:
            scene_fragment_length = -1

            assert needed_pre_padding == 0
        else:
            scene_obs_end = scene_obs_start + self.fragment_length
            scene_obs_start = max(0, scene_obs_start)
            scene_obs_end = min(self._unpadded_traj_lens[traj_idx], scene_obs_end)
            scene_fragment_length = scene_obs_end - scene_obs_start + 1

        needed_post_padding = (
            self.fragment_length - scene_fragment_length - needed_pre_padding
        )

        # logger.info(
        #     f"Got index {index}, traj_idx {traj_idx}, obs_idx {obs_idx}, scene_obs_start {scene_obs_start}, end {scene_obs_end}, unpadded traj len {self._unpadded_traj_lens[traj_idx]} with padding {self.pre_padding}, {self.post_padding}, fragment len {self.fragment_length}, corrected_fragment_length {scene_fragment_length}, needed pre padding {needed_pre_padding} and post padding {needed_post_padding}, getting scene idcs {scene_obs_start}-{scene_obs_end}."
        # )

        traj = self._get_bc_traj(
            traj_idx,
            fragment_idx=scene_obs_start,
            fragment_length=scene_fragment_length,
            cams=self.config.cameras,
            force_skip_rgb=force_skip_rgb or not (force_load_raw or debug_encoding),
            extra_attr=self.config.extra_attr,
        )
        pre_padding_td = torch.zeros_like(traj[0])
        post_padding_td = torch.zeros_like(traj[0])
        pre_padding_td.feedback = traj[0].feedback
        post_padding_td.feedback = traj[-1].feedback
        # TODO: also pad the other attributes with proper values? Eg unit quaternions.
        pre_padding_td = pre_padding_td.expand(needed_pre_padding)
        post_padding_td = post_padding_td.expand(needed_post_padding)

        padded_traj = torch.cat([pre_padding_td, traj, post_padding_td])

        return padded_traj

    def _get_bc_traj(
        self,
        traj_index,
        fragment_idx: int | None = None,
        cams: tuple["str", ...] = ("wrist",),
        fragment_length: int | None = None,
        force_skip_rgb: bool = True,
        extra_attr: dict | None = None,
    ):
        if fragment_idx is None:
            assert self._full_traj_mode, "Need to specify obs_index for fragment mode."
        # fragment_length: None for self.fl, -1 for full traj
        if fragment_length is None:
            fragment_length = self.fragment_length

        if extra_attr is None:
            extra_attr = {}

        # define the additional attributes to load
        embedding_attr = [
            "cam_{}_{}".format(c[0], a) for c in cams for a in self.load_pre_embedding
        ]

        embedding_attr = {
            c: {
                a: self.scene_data._get_cam_attr_name(c, a)
                for a in self.load_pre_embedding
            }
            for c in cams
        }

        encoding_attr = self.pre_encoding_attr

        # if we load the embeddings directly, we can skip the camera attributes
        skip_rgb = (
            bool(self.load_pre_embedding) or bool(encoding_attr)
        ) or force_skip_rgb

        return self.scene_data._get_bc_traj(
            traj_index,
            cams=cams,
            fragment_idx=fragment_idx,
            mask_type=self.config.mask_type,
            sample_freq=self.config.sample_freq,
            extra_attr=extra_attr,
            fragment_length=fragment_length,
            embedding_attr=embedding_attr,
            encoding_attr=encoding_attr,
            encoding_name=self.pre_encoding_name,
            encoder_name=self.encoder_name,
            skip_rgb=skip_rgb,
        )

    def sample_bc(self, batch_size, cam=("wrist",), idx=None, skip_rgb=False):
        # legacy func for encoder init and viz

        if type(cam) is str:
            cam = (cam,)
        if idx is None:
            idx = self.scene_data.sample_traj_idx(batch_size=batch_size)

        return collate(
            [
                self._get_bc_traj(
                    i, cams=cam, fragment_length=-1, force_skip_rgb=skip_rgb
                )
                for i in idx
            ]
        )

    def sample_data_point_with_object_labels(
        self,
        cam: str = "wrist",
        traj_idx: int | None = None,
        img_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        get_mask = not bool(self.config.mask_type in [None, MaskTypes.NONE])

        obs = self.scene_data.sample_observation(
            traj_idx=traj_idx,
            img_idx=img_idx,
            cam=cam,
            mask_type=self.config.mask_type,
            raw_mask=False,
            labels=self.object_labels,
            collapse_labels=False,
            get_rgb=True,
            get_int=False,
            get_ext=False,
            get_depth=True,
            get_mask=get_mask,
            get_action=False,
            get_feedback=False,
            get_gripper_pose=False,
            get_proprio_obs=False,
            get_wrist_pose=False,
        )

        return obs

    def sample_data_point_with_ground_truth(
        self, cam="wrist", traj_idx=None, img_idx=None
    ) -> SingleCamObservation:
        obs = self.scene_data.sample_observation(
            traj_idx=traj_idx,
            img_idx=img_idx,
            cam=cam,
            mask_type=MaskTypes.GT,
            raw_mask=False,
            labels=self.object_labels,
            collapse_labels=False,
            get_rgb=True,
            get_int=True,
            get_ext=True,
            get_depth=True,
            get_mask=True,
            get_action=False,
            get_feedback=False,
            get_gripper_pose=False,
            get_object_poses=True,
            get_proprio_obs=False,
            get_wrist_pose=False,
        )

        return obs

    def get_object_labels(self):
        if (labels := self.scene_data.object_labels) is not None:
            pass
        elif (mtype := self.config.mask_type) is MaskTypes.GT:
            labels = self.scene_data.object_labels_gt
        elif mtype is MaskTypes.TSDF:
            labels = self.scene_data.object_labels_tsdf
        elif mtype is None or MaskTypes.NONE:
            labels = None
        else:
            raise ValueError("Could not get labels for type {}".format(mtype))

        if label_subset := self.config.only_use_labels:
            for l in label_subset:
                if labels is None or l not in labels:
                    logger.warning(f"Label {l} not in dataset.")
            logger.info(f"Only using labels {label_subset}")
            labels = label_subset

        return labels

    def add_embedding(
        self, traj_idx, obs_idx, cam_name, emb_name, encoding, encoder_name
    ):
        return self.scene_data.add_embedding(
            traj_idx, obs_idx, cam_name, emb_name, encoding, encoder_name
        )

    def load_embedding(self, traj_idx, img_idx, cam, embedding_name):
        return self.scene_data.load_embedding(
            traj_idx, img_idx, cam, embedding_name, self.encoder_name
        )

    def load_embedding_batch(self, traj_idx, img_idx, cam, embedding_name):
        return self.scene_data.load_embedding_batch(
            traj_idx, img_idx, cam, embedding_name, self.encoder_name
        )

    def add_embedding_config(self, encoder_name, config):
        return self.scene_data.add_embedding_config(encoder_name, config)

    def add_encoding(
        self,
        traj_idx,
        obs_idx,
        cam_name,
        attr_name,
        encoding,
        encoder_name,
        encoding_name,
    ):
        return self.scene_data.add_encoding(
            traj_idx,
            obs_idx,
            cam_name,
            attr_name,
            encoding,
            encoder_name,
            encoding_name,
        )

    def add_encoding_fig(
        self,
        traj_idx,
        obs_idx,
        cam_name,
        attr_name,
        fig,
        encoder_name,
        encoding_name,
        bbox,
        channel=None,
    ):
        return self.scene_data.add_encoding_fig(
            traj_idx,
            obs_idx,
            cam_name,
            attr_name,
            fig,
            encoder_name,
            encoding_name,
            bbox,
            channel,
        )

    def add_encoding_config(self, encoder_name, encoding_name, config):
        return self.scene_data.add_encoding_config(encoder_name, encoding_name, config)

    def add_traj_attr(self, traj_idx, obs_idx, attr_name, value):
        return self.scene_data.add_traj_attr(traj_idx, obs_idx, attr_name, value)

    def update_traj_attr(self, traj_idx, obs_idx, attr_name, value):
        return self.scene_data.update_traj_attr(traj_idx, obs_idx, attr_name, value)

    def get_pre_encoding_distribution(self):
        assert len(self.pre_encoding_attr) == 1, "TODO: extend to multiple attributes"
        return self.scene_data.get_pre_encoded_attribute_distribution(
            encoder_name=self.encoder_name,
            encoding_name=self.pre_encoding_name,
            attribute=self.pre_encoding_attr[0],
        )
