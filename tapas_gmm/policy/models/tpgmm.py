import dataclasses
import enum
import itertools
from collections import OrderedDict

# import pickle
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Callable, Sequence

import dill as pickle
import numpy as np
import pandas as pd
import riepybdlib as rbd
import torch
from loguru import logger
from omegaconf import OmegaConf
from scipy.linalg import block_diag
from tqdm.auto import tqdm

import tapas_gmm.utils.geometry_np as geometry_np
from tapas_gmm.dataset.demos import Demos, DemosSegment, PartialFrameViewDemos
from tapas_gmm.utils.config import (
    _SENTINELS,
    COPY_FROM_MAIN_FITTING,
    recursive_compare_dataclass,
)

# TODO: change geometry_torch imports to module import + geometry_torch. prefix in use
from tapas_gmm.utils.geometry_torch import (
    matrix_to_quaternion,
    quaternion_is_standard,
    quaternion_is_unit,
    quaternion_to_euler,
    quaternion_to_matrix,
    reconstruct_quaternion,
    remove_quaternion_dim,
)
from tapas_gmm.utils.gmm import (  # concat_mvn,
    concat_mvn_rbd,
    get_component_mu_sigma,
    hmm_transition_probabilities,
)
from tapas_gmm.utils.logging import indent_logs
from tapas_gmm.utils.manifolds import (
    Manifold_Quat,
    Manifold_R1,
    Manifold_R3,
    Manifold_S1,
    Manifold_S2,
    Manifold_T,
)
from tapas_gmm.utils.misc import multiply_iterable
from tapas_gmm.utils.np import get_indeces_of_duplicate_rows
from tapas_gmm.utils.torch import batched_block_diag
from tapas_gmm.utils.typing import NDArrayOrNDArraySeq
from tapas_gmm.viz.gmm import (
    SingleDimPlotData,
    TPGMMPlotData,
    plot_coordinate_frame_trajs,
    plot_gmm_components,
    plot_gmm_frames,
    plot_gmm_frames_time_based,
    plot_gmm_time_based,
    plot_gmm_trajs_3d,
    plot_gmm_xdx_based,
    plot_hmm_transition_matrix,
    plot_reconstructions_3d,
    plot_reconstructions_time_based,
    plot_state_activation_for_demo,
    plot_state_sequence_for_demo,
    plot_tangent_data,
)
from tapas_gmm.viz.quaternion import plot_quat_components
from tapas_gmm.viz.threed import plot_rotation_basis

# import tapas_gmm.utils.geometry_torch as geometry_torch


# NOTE: Don't reload this module - this breaks enum comparison!

GMM = rbd.statistics.GMM
Gaussian = rbd.statistics.Gaussian
MarkovModel = rbd.statistics.HMM | rbd.statistics.HSMM
Manifold = rbd.manifold.Manifold


class ReprEnum:
    def __repr__(self):
        return f"{self.name}"  # type: ignore


class ModelType(ReprEnum, enum.Enum):
    GMM = 1
    HMM = 2
    HSMM = 3


MarkovTypes = (ModelType.HMM, ModelType.HSMM)


class ReconstructionStrategy(ReprEnum, enum.Enum):
    GMR = 1
    LQR_VITERBI = 2


class FittingStage(ReprEnum, enum.IntEnum):
    # Leveraging the int order in the code, so make sure the values are in
    # order by auto-assigning them.
    NONE = enum.auto()
    INIT = enum.auto()
    EM_GMM = enum.auto()
    EM_HMM = enum.auto()


class InitStrategy(ReprEnum, enum.Enum):
    KMEANS = 1
    RANDOM = 2
    TIME_BASED = 3
    SAMMI = 4


class OAMapping(ReprEnum, enum.Enum):
    TIME2STATE = 1
    STATE2ACTION = 2


# TODO: other configs (EM maxsteps, ...), SAMMI
@dataclass
class RbdEMConfig:
    reg_lambda: float = 1e-3
    reg_lambda2: float = 1e-3
    reg_type: rbd.statistics.RegularizationType = (
        rbd.statistics.RegularizationType.COMBINED
    )
    minsteps: int = 1
    maxsteps: int = 50
    fix_last_component: bool = True
    plot: bool = False


@dataclass
class RbdHMMEmConfig: ...


@dataclass
class TPGMMConfig:
    n_components: int = 4
    model_type: ModelType = ModelType.HMM
    # model_config: Any = ...

    reg_type: rbd.statistics.RegularizationType = (
        rbd.statistics.RegularizationType.COMBINED
    )
    reg_shrink: float = 1e-3
    reg_time_shrink: float = 1e-3
    reg_diag: float = 1e-3
    reg_diag_vel: float = 1e-5
    reg_diag_gripper: float = 1e-1
    reg_diag_magnitudes: float = 1e-5

    reg_em_left_to_right: bool = True

    reg_em_trans: float | None = None

    reg_em_finish_type: rbd.statistics.RegularizationType = (
        rbd.statistics.RegularizationType.COMBINED
    )
    reg_em_finish_shrink: float = 1e-3
    reg_em_finish_diag: float = 1e-3
    reg_em_finish_diag_vel: float = 1e-5
    reg_em_finish_diag_gripper: float = 1e-1
    reg_em_finish_diag_magnitudes: float = 1e-5

    em_steps: int = 50

    reg_init_diag: float = 5e-5
    reg_init_type: rbd.statistics.RegularizationType = (
        rbd.statistics.RegularizationType.DIAGONAL
    )

    trans_cov_mask_t_pos_corr: bool = True
    joint_cov_diag_reg: bool = False

    add_time_component: bool = False
    add_action_component: bool = True
    position_only: bool = False
    use_riemann: bool = True
    action_as_orientation: bool = False
    action_with_magnitude: bool = False
    add_gripper_action: bool = False

    heal_time_variance: bool = False

    fix_first_component: bool = True
    fix_last_component: bool = True
    fixed_first_component_n_steps: int = 2
    fixed_last_component_n_steps: int = 2

    rbd_em: RbdEMConfig = RbdEMConfig()


@dataclass
class FrameSelectionConfig:
    fix_frames: bool | _SENTINELS = COPY_FROM_MAIN_FITTING
    init_strategy: Any = COPY_FROM_MAIN_FITTING
    fitting_actions: Any = COPY_FROM_MAIN_FITTING
    rel_score_threshold: float = 0.75
    use_bic: bool = False
    use_precision: bool = True
    drop_redundant_frames: bool = False
    pose_only: bool = True

    gt_frames: list[list[int]] | None = None


@dataclass
class DemoSegmentationConfig:
    min_len: int = 1
    distance_based: bool = True
    gripper_based: bool = False
    velocity_based: bool = False

    distance_threshold: float = 0.06

    repeat_first_step: int = 0
    repeat_final_step: int = 10

    components_prop_to_len: bool = False

    min_n_components: int = 1
    use_gripper_states: bool = True

    no_segmentation: bool = False

    min_end_distance: int = 10

    velocity_threshold: float = 0.005
    gripper_threshold: float = 0.2
    max_idx_distance: int = 4


@dataclass
class CascadeConfig:
    kl_keep_time_dim: bool = True
    kl_keep_action_dim: bool = True
    kl_keep_rotation_dim: bool = True
    models_are_sequential: bool = True
    kl_sigma_scale: float = 1.0

    min_prob: float | None = None


@dataclass
class AutoTPGMMConfig:
    tpgmm: TPGMMConfig = TPGMMConfig()

    frame_selection: FrameSelectionConfig = FrameSelectionConfig()
    demos_segmentation: DemoSegmentationConfig = DemoSegmentationConfig()
    cascade: CascadeConfig = CascadeConfig()


class TPGMM:
    def __init__(self, config: TPGMMConfig):
        self.config = config

        if not config.add_action_component and not config.add_time_component:
            raise ValueError(
                "Should add time or action dimension. "
                "What else do you want to predict?"
            )

        if not config.use_riemann:
            raise ValueError("Dropping support for pbdlib.")

        self.use_riemann = config.use_riemann
        self.skip_quaternion_dim = None if self.use_riemann else 0

        self.manifold = None
        self.model = None
        self._demos = None
        self._fix_frames = None

        self.fitting_stage = FittingStage.NONE

        self._online_joint_model: GMM | None = None
        self._online_first_step: bool | None = None

    @property
    def add_rotation_component(self) -> bool:
        return not self.config.position_only

    @property
    def add_time_component(self) -> bool:
        return self.config.add_time_component

    @property
    def add_action_component(self) -> bool:
        return self.config.add_action_component

    @property
    def action_as_orientation(self) -> bool:
        return self.config.action_as_orientation

    @property
    def action_with_magnitude(self) -> bool:
        return self.config.action_with_magnitude

    @property
    def add_gripper_action(self) -> bool:
        return self.config.add_gripper_action

    @cached_property
    def _sigma_dim(self) -> int:
        assert self._model_check()
        return self.model.sigma.shape[1]

    @cached_property
    def _full_reg_diag(self) -> float | np.ndarray:
        """
        Get the full diagonal regularization vector for the model.
        Used for decoupling gripper action regularization from the rest.
        """
        if not self.config.action_with_magnitude and not self.config.add_gripper_action:
            return self.config.reg_diag
        else:
            n_gripper_dims = int(self.config.add_gripper_action)
            n_mag_dims = int(self.config.action_with_magnitude) * (
                1 + int(1 - self.config.position_only)
            )
            n_pos_dims = 3 if self.config.position_only else 6
            vel_man_dim = 2 if self.config.action_as_orientation else 3
            n_vel_dims = (
                int(self.config.add_action_component)
                * vel_man_dim
                * (2 - int(self.config.position_only))
            )

            assert self._sigma_dim == self.n_frames * (
                n_pos_dims + n_vel_dims
            ) + n_gripper_dims + n_mag_dims + int(self.config.add_time_component)

            components = [np.ones(n_pos_dims) * self.config.reg_diag]

            if self.config.add_action_component:
                components.append(np.ones(n_vel_dims) * self.config.reg_diag_vel)

            components = components * self.n_frames

            if self.add_time_component:
                components = [np.ones(1) * self.config.reg_diag] + components

            if self.config.action_with_magnitude:
                components.append(np.ones(n_mag_dims) * self.config.reg_diag_magnitudes)

            if self.config.add_gripper_action:
                components.append(
                    np.ones(n_gripper_dims) * self.config.reg_diag_gripper
                )

            full = np.concatenate(components)

            return full

    @cached_property
    def _full_reg_diag_per_frame(self) -> float | np.ndarray:
        """
        Get the full diagonal regularization vector for the per-frame model.
        Used for decoupling gripper action regularization from the rest.
        """
        if not self.config.action_with_magnitude and not self.config.add_gripper_action:
            return self.config.reg_diag
        else:
            n_gripper_dims = int(self.config.add_gripper_action)
            n_mag_dims = int(self.config.action_with_magnitude) * (
                1 + int(1 - self.config.position_only)
            )
            n_pos_dims = 3 if self.config.position_only else 6
            n_vel_dims = (
                0
                if not self.config.add_action_component
                else (
                    3
                    if self.config.position_only
                    else 4 if self.action_as_orientation else 6
                )
            )

            components = []

            if self.add_time_component:
                components.append(np.ones(1) * self.config.reg_diag)

            components = components.append(np.ones(n_pos_dims) * self.config.reg_diag)

            if self.config.add_action_component:
                components.append(np.ones(n_vel_dims) * self.config.reg_diag_vel)

            if self.config.action_with_magnitude:
                components.append(np.ones(n_mag_dims) * self.config.reg_diag_magnitudes)

            if self.config.add_gripper_action:
                components.append(
                    np.ones(n_gripper_dims) * self.config.reg_diag_gripper
                )

            return np.concatenate(components)

    @cached_property
    def _full_reg_em_finish_diag(self) -> float | np.ndarray:
        if not self.config.action_with_magnitude and not self.config.add_gripper_action:
            return self.config.reg_em_finish_diag
        else:
            n_gripper_dims = int(self.config.add_gripper_action)
            n_mag_dims = int(self.config.action_with_magnitude) * (
                1 + int(1 - self.config.position_only)
            )
            n_pos_dims = 3 if self.config.position_only else 6
            vel_man_dim = 2 if self.config.action_as_orientation else 3
            n_vel_dims = (
                int(self.config.add_action_component)
                * vel_man_dim
                * (2 - int(self.config.position_only))
            )

            assert self._sigma_dim == self.n_frames * (
                n_pos_dims + n_vel_dims
            ) + n_gripper_dims + n_mag_dims + int(self.config.add_time_component)

            components = [np.ones(n_pos_dims) * self.config.reg_em_finish_diag]

            if self.config.add_action_component:
                components.append(
                    np.ones(n_vel_dims) * self.config.reg_em_finish_diag_vel
                )

            components = components * self.n_frames

            if self.add_time_component:
                components = [np.ones(1) * self.config.reg_em_finish_diag] + components

            if self.config.action_with_magnitude:
                components.append(
                    np.ones(n_mag_dims) * self.config.reg_em_finish_diag_magnitudes
                )

            if self.config.add_gripper_action:
                components.append(
                    np.ones(n_gripper_dims) * self.config.reg_em_finish_diag_gripper
                )

            full = np.concatenate(components)

            return full

    @cached_property
    def _em_dep_mask(self) -> np.ndarray:
        """
        Get the dependency mask for the EM algorithm.
        NOTE: not using this anymore. Can use it to get more fine-grained control over
        the regularization. The covariance matrices are multiplied by this mask after
        each EM step.
        """
        d = self._sigma_dim

        if self.config.add_time_component:
            mask = np.ones((d, d)) * (1 - self.config.reg_shrink)
            mask[0, 0:d] = self.config.reg_time_shrink
            mask[0:d, 0] = self.config.reg_time_shrink
            np.fill_diagonal(mask, 1)

            self.config.reg_shrink = 0
        else:
            mask = np.ones(self._sigma_dim)

        return mask

    @cached_property
    def _trans_cov_reg_kwargs(self) -> dict | None:
        # reg_kwargs = dict(
        #     reg_lambda=self.config.reg_shrink,
        #     reg_lambda2=self._full_reg_diag_per_frame,
        #     reg_type=self.reg_type,
        # )
        reg_kwargs = None

        return reg_kwargs

    @cached_property
    def _trans_cov_mask(self) -> np.ndarray:
        """
        Using diagonal regularization for the transition covariance matrix.
        Could be used to get more fine-grained control over the regularization.
        """
        dim = (
            self._per_frame_tangent_dim
            + int(self.config.add_time_component)
            + int(self._n_global_dims)
        )

        mask = np.eye(dim)

        if self.config.trans_cov_mask_t_pos_corr:
            mask[:4, 0] = 1
            mask[0, :4] = 1

        return mask

    @cached_property
    def _join_cov_mask(self) -> np.ndarray:
        """
        Additional regularization mask applied to the joint covariance matrix.
        """
        dim = (
            self._per_frame_tangent_dim
            + int(self.config.add_time_component)
            + int(self._n_global_dims)
        )

        if self.config.joint_cov_diag_reg:
            mask = np.eye(dim)
        else:
            mask = np.ones((dim, dim))

        return mask

    @cached_property
    def _t_delta(self) -> float:
        """
        Get the average time step width of the fitted demos.
        Used for time-driven predictions.
        """
        assert self._model_check()
        assert self._demos is not None
        return 1 / self._demos.ss_len

    @cached_property
    def n_frames(self) -> int:
        """
        Number of coordinate frames (task parameters).
        """
        assert self._demos is not None
        return self._demos.n_frames

    @cached_property
    def _per_frame_manifold(self) -> Manifold:
        """
        The manifold of a single frame.
        """
        m_state = (
            Manifold_R3 if self.config.position_only else Manifold_R3 * Manifold_Quat
        )
        m_action = (
            Manifold_S2
            if self.config.action_as_orientation and self.config.position_only
            else (
                Manifold_S2 * Manifold_S2
                if self.config.action_as_orientation
                else m_state
            )
        )
        m_frame = m_state * m_action if self.config.add_action_component else m_state

        return m_frame

    @property
    def _per_frame_manifold_dim(self) -> int:
        """
        Manifold dimensionality of the per-frame manifold.
        """
        return self._per_frame_manifold.n_dimM

    @property
    def _per_frame_tangent_dim(self) -> int:
        """
        Tangent space dimensionality of the per-frame manifold.
        """
        return self._per_frame_manifold.n_dimT

    @cached_property
    def _per_frame_manifold_n_submanis(self) -> int:
        """
        Number of submanifolds in the per-frame manifold.
        """
        return len(self._per_frame_manifold.get_submanifolds())

    @cached_property
    def _action_magnitude_manifold(self) -> Manifold | None:
        """
        Manifold of the action magnitude. None if not configured. Only configured if
        actions are included and are factorized into magnitude and direction.
        """
        if self.config.add_action_component and self.config.action_with_magnitude:
            return (
                Manifold_R1 if self.config.position_only else Manifold_R1 * Manifold_S1
            )
        else:
            return None

    @cached_property
    def _gripper_action_manifold(self) -> Manifold | None:
        """
        Manifold of the gripper action. None if not configured.
        """
        if self.config.add_gripper_action:
            return Manifold_R1
        else:
            return None

    @cached_property
    def _global_dim_manifold(self) -> Manifold | None:
        """
        Manifold of the global dimensions (action magnitudes and gripper action).
        None if not configured.
        """
        manis = []

        if mag := self._action_magnitude_manifold:
            manis.append(mag)
        if grip := self._gripper_action_manifold:
            manis.append(grip)

        return multiply_iterable(manis) if manis else None

    @cached_property
    def _global_mani_full_model_start_idx(self) -> int:
        """
        Start index of the global dimensions (action magnitudes and gripper action)
        in the joint manifold. Indexes the submanifolds (not the data).
        """
        return self._per_frame_manifold_n_submanis * self.n_frames + int(
            self.config.add_time_component
        )

    @cached_property
    def _global_mani_joint_model_start_idx(self) -> int:
        """
        Like _global_mani_start_idx but for the joint model instead of the per-frame
        model.
        """
        return self._per_frame_manifold_n_submanis + int(self.config.add_time_component)

    @cached_property
    def _global_action_dim_manifold(self) -> int:
        """
        Data dimensionality of the global action dims (action magnitudes and gripper
        action) in the joint manifold.
        """
        if self._action_magnitude_manifold:
            dim_mag = self._action_magnitude_manifold.n_dimM
        else:
            dim_mag = 0

        if self._gripper_action_manifold:
            dim_grip = self._gripper_action_manifold.n_dimM
        else:
            dim_grip = 0

        return dim_mag + dim_grip

    @cached_property
    def _global_action_dim_tangent(self) -> int:
        """
        Tangent space dimensionality of the global action dims (action magnitudes and
        gripper action) in the joint manifold.
        """
        if self._action_magnitude_manifold:
            dim_mag = self._action_magnitude_manifold.n_dimT
        else:
            dim_mag = 0

        if self._gripper_action_manifold:
            dim_grip = self._gripper_action_manifold.n_dimT
        else:
            dim_grip = 0

        return dim_mag + dim_grip

    def _setup_manifold(self, n_frames: int) -> None:
        """
        Setup the manifold for the given number of frames.

        For example, for a state-driven model with position + rotation, for each frame,
        the manifold is given by R^3 x S^3 x R^3 x S^3, ie. position, orientation,
        position delta and orientation delta.
        The submanifold can vary:
        - when factorizing the action into magnitude and direction, the position delta
            and rotation delta are S^2 per frame, plus a global magnitude in R^1/S^1.
        - when the model is position-only, the orientation and orientation delta are
            omitted.
        - the model can also be without action (delta dimensions).
        The joint manifold is given by the product of per-frame manifolds.

        Global dimensions (if configured) are added in the following way:
        - time-dimension: time (R^1) is added as the first dimension of the joint mani.
        - action magnitudes: magnitudes (R^1/S^1) are added after the per-frame manis.
        - gripper action: gripper action (R^1) is added as the last dimension.

        Parameters
        ----------
        n_frames : int
            Number of coordinate frames (task parameters).
        """
        if self.use_riemann and self.config.position_only:
            logger.warning("Riemannian GMMs only make sense for rotations.")

        if not self.use_riemann:
            return

        if n_frames != self.n_frames:
            logger.warning(
                f"Number of frames changed from {self.n_frames} to {n_frames}. "
                "Need to reset the model?"
            )

        manis = []

        if self.config.add_time_component:
            manis.append(Manifold_T)

        m_frame = self._per_frame_manifold

        for _ in range(n_frames):
            manis.append(m_frame)

        if m_mag := self._action_magnitude_manifold:
            manis.append(m_mag)

        if m_grip := self._gripper_action_manifold:
            manis.append(m_grip)

        self.manifold = multiply_iterable(manis)

        logger.info(f"Manifold: {self.manifold.name}")

    @cached_property
    def _submani_sizes(self) -> tuple[tuple[int, int], ...]:
        """
        Submanifold and tangent space sizes of all submanifolds in the joint manifold.
        Tuple of tuples of (submanifold size, tangent space size).
        """
        assert self._manifold_check()
        return tuple((m.n_dimM, m.n_dimT) for m in self.manifold.get_submanifolds())

    @cached_property
    def _submani_indices(self) -> tuple[tuple[int, int], ...]:
        """
        Submanifold start and stop indices for all submanifolds in the joint manifold.
        Tuple of tuples of (submanifold start index, submanifold stop index).
        """
        submani_start_idcs = np.cumsum((0,) + tuple(m[0] for m in self._submani_sizes))

        return tuple(
            (i, j) for i, j in zip(submani_start_idcs[:-1], submani_start_idcs[1:])
        )

    @cached_property
    def _submani_tangent_indices(self) -> tuple[tuple[int, int], ...]:
        """
        Submanifold tangent space start and stop indices for all submanifolds in the
        joint manifold.
        Tuple of tuples of (submanifold tangent space start index, stop index).
        """
        submani_tangent_start_idcs = np.cumsum(
            (0,) + tuple(m[1] for m in self._submani_sizes)
        )

        return tuple(
            (i, j)
            for i, j in zip(
                submani_tangent_start_idcs[:-1], submani_tangent_start_idcs[1:]
            )
        )

    @cached_property
    def _n_global_dims(self) -> int:
        n_global_dims = (
            1
            if self.config.add_action_component and self.config.action_with_magnitude
            else 0
        )
        n_global_dims *= 2 if not self.config.position_only else 1
        n_global_dims += 1 if self.config.add_gripper_action else 0

        return n_global_dims

    @cached_property
    def _global_action_full_model_dim_idcs(self) -> tuple[int, ...]:
        """
        Get the submanifold indices of the global dimensions (action magnitudes and
        gripper action) in the joint manifold.

        Used to extract the global dimensions from the joint data and to add it to the
        per-frame data.

        Returns
        -------
        tuple[int, ...]
            The submanifold indices of the global dimensions.
        """
        start = self._global_mani_full_model_start_idx

        return tuple(range(start, start + self._n_global_dims))

    @cached_property
    def _global_action_dim_joint_model_idcs(self) -> tuple[int, ...]:
        """
        Like _global_action_dim_idcs but for the joint model instead of the per-frame
        model.
        """
        start = self._global_mani_joint_model_start_idx

        return tuple(range(start, start + self._n_global_dims))

    @cached_property
    def _global_action_dim_sizes(self) -> tuple[int, ...]:
        """
        Get the dimensionality of the tangent space of the global dimensions (action
        magnitudes and gripper action) in the joint manifold.

        Used to construct the full frame transformation matrix.

        Returns
        -------
        tuple[int, ...]
            The dimensionality of the tangent space of the global dimensions.
        """
        tan_dims = [m.n_dimT for m in self.manifold.get_submanifolds()]

        return tuple(tan_dims[self._global_mani_full_model_start_idx :])

    def _setup_model(self, reset: bool = False) -> None:
        """
        Helper function to set up the actual model.
        Only needs to be called once in the beginning, or if model parameters (number of
        components, number of frames, model type, ...) change.
        Creates the manifold (if needed) and initializes the model class.
        """
        if self.model is not None and not reset:
            return

        assert self._demos is not None
        self._setup_manifold(self._demos.n_frames)

        lib = rbd.statistics
        Model = (
            lib.GMM
            if self.config.model_type is ModelType.GMM
            else lib.HMM if self.config.model_type is ModelType.HMM else lib.HSMM
        )

        if self.use_riemann:
            self.model = Model(self.manifold, n_components=self.config.n_components)
        else:
            self.model = Model(nb_states=self.config.n_components)  # type: ignore

        self._init_borders = None

    def reset_model(self) -> None:
        """
        Reinstantiate the model class using the current class configuration.
        """
        self._setup_model(reset=True)

    def _setup_data(self, demos: Demos) -> None:
        """
        Assign the given demos to self._demos and prepare the data for fitting.
        """
        if self._demos is None:
            self._demos = demos
        elif self._demos != demos:
            logger.warning("Overwriting demos. Need to reset the model?")
            self._demos = demos
        else:
            return

        self.demo_data_flat = self._demos.get_per_frame_data(
            flat=True,
            subsampled=True,
            fixed_frames=self._fix_frames,
            pos_only=self.config.position_only,
            skip_quat_dim=self.skip_quaternion_dim,
            add_action_dim=self.config.add_action_component,
            add_time_dim=self.config.add_time_component,
            action_as_orientation=self.config.action_as_orientation,
            action_with_magnitude=self.config.action_with_magnitude,
            add_gripper_action=self.config.add_gripper_action,
            numpy=True,
        )

    def get_frame_marginals(
        self, time_based: bool, model: GMM | None = None
    ) -> tuple[GMM, ...]:
        """
        Split up the joint GMM into per-frame marginals.
        """
        if not self.use_riemann:
            raise NotImplementedError("Dropping support for pbdlib.")

        if model is None:
            assert self._model_check()
            model = self.model

        manifolds_per_frame = _get_rbd_manifolds_per_frame(
            self.config.position_only, self.config.add_action_component
        )

        global_dim_idcs = self._global_action_full_model_dim_idcs
        model_contains_time = self.config.add_time_component

        if global_dim_idcs is None or len(global_dim_idcs) == 0:
            # No global dims, only per-frame (and time)
            n_frame_manifolds = model.manifold.n_manifolds - int(model_contains_time)
        else:
            # Global dims begin after the per-frame manifolds
            n_frame_manifolds = min(global_dim_idcs) - int(model_contains_time)

        n_frames = n_frame_manifolds // manifolds_per_frame

        # Indeces are manifold-indences
        per_frame_marginals = tuple(
            model.margin(
                _get_rbd_manifold_indices(
                    j,
                    model_contains_time,
                    manifolds_per_frame,
                    with_global_dims=global_dim_idcs,
                    keep_time_dim=time_based,
                )
            )
            for j in range(n_frames)
        )

        return per_frame_marginals

    def patch_frame_transforms(
        self,
        hom_trans: np.ndarray,
        quats: np.ndarray,
        drop_time: bool,
        model: GMM | None = None,
    ) -> tuple[np.ndarray, tuple[Any, ...]]:
        """
        Calculate the manifold transformations from the frame transformations.
        Pass the homogeneous transform matrix AND the corresponding quaternions to avoid
        inconsistencies caused by the conversion from quaternions to rotation matrices.

        hom_trans is a homogeneous transform matrix over pose and velocity (if
        with_action_dim), ie. 7x7 or 4x4.
        quats is the stacked pose and velocity quaternions (if with_action_dim), ie.
        2x4 or 1x4.

        The transform is cast into the proper shape by
        - adding a time dimension if time_based is True
        - adding transformation for rotation dims if position_only is False
        - adding identity transforms for global dims (action magnitude and gripper
            action) if global_dim_sizes is not None
        - adapting the transformation to S2 if action_as_orientation is True
        - creating a block diagonal matrix from the above for A and a vector of
            per-manifold translation vectors for b (that's how riepybdlib needs it)
        """
        if not self.use_riemann:
            raise NotImplementedError("Dropping support for pbdlib.")

        if model is None:
            assert self._model_check()
            model = self.model

        man_id_elem = model.manifold.id_elem
        if model.base != man_id_elem:
            raise ValueError(
                "Parallel_transport to b assumes that the manifold is currently based "
                "at the identity element. This is not the case for the given model."
            )

        assert quats is not None, "Quaternions are required for Riemannian"

        trans_shape = 7 if self.config.add_action_component else 4
        assert hom_trans.shape == (trans_shape, trans_shape)

        r3_dim_size = 3
        s2_tan_size = 2

        if len(quats.shape) == 1:
            assert (
                not self.config.add_action_component
            ), "Need two quaternions for action dim"
            quat_pose = quats
        else:
            quat_pose = quats[:, 0]
            quat_vel = quats[:, 1]

        A_comps = []
        b_comps = []

        if self.config.add_time_component and not drop_time:
            A_time = np.ones(1)
            b_time = np.zeros(1)
            A_comps.append(A_time)
            b_comps.append(b_time)

        pos_slice = slice(0, r3_dim_size)
        vel_slice = slice(r3_dim_size, 2 * r3_dim_size)

        A_pos = hom_trans[pos_slice, pos_slice]
        A_vel = hom_trans[vel_slice, vel_slice]
        b_full = hom_trans[:-1, -1]
        b_pos = b_full[pos_slice]
        b_vel = b_full[vel_slice]

        A_comps.append(A_pos)
        b_comps.append(b_pos)

        position_only = self.config.position_only
        with_action_dim = self.config.add_action_component
        action_as_orientation = self.config.action_as_orientation

        if not position_only:
            A_rot = np.eye(r3_dim_size)
            A_comps.append(A_rot)
            b_quat = rbd.Quaternion.from_nparray(quat_pose)
            b_comps.append(b_quat)

        if action_as_orientation:
            s2_idx = 2 + int(self.add_rotation_component) - int(drop_time)
            s2_id_elem = model.manifold.id_elem[s2_idx]
            b_s2 = geometry_np.rotate_vector_by_quaternion(s2_id_elem, quat_pose)

        if with_action_dim:
            A_pvel = np.eye(s2_tan_size) if action_as_orientation else A_vel
            A_comps.append(A_pvel)
            # b_pvel = b_s2 if action_as_orientation else b_vel
            # b_pvel = rbd.Quaternion.from_nparray(quat_vel)
            b_pvel = (
                rbd.Quaternion.from_nparray(quat_vel)
                if action_as_orientation
                else b_vel
            )
            b_comps.append(b_pvel)

        if with_action_dim and not position_only:
            A_rvel = np.eye(s2_tan_size if action_as_orientation else r3_dim_size)
            A_comps.append(A_rvel)
            # b_rvel = (
            #     b_s2 if action_as_orientation else rbd.Quaternion.from_nparray(quat_vel)
            # )
            b_rvel = rbd.Quaternion.from_nparray(quat_vel)  #  b_pvel
            b_comps.append(b_rvel)

        global_dim_sizes = self._global_action_dim_sizes
        global_dim_indices = self._global_action_dim_joint_model_idcs

        if global_dim_sizes is not None:
            for s, i in zip(global_dim_sizes, global_dim_indices):
                A_comps.append(np.eye(s))
                b_comps.append(man_id_elem[i - int(drop_time)])

        A_joint = block_diag(*A_comps)  # needs to be in tangent space, jointly
        b_joint = tuple(b_comps)  # b needs to be given per sub-manifold

        return A_joint, b_joint

    def _select_fitting_actions(
        self, fitting_actions: Sequence[FittingStage] | None
    ) -> Sequence[FittingStage]:
        if fitting_actions is None:
            logger.info("No fitting actions specified. Auto selecting.")
            default_actions = [FittingStage.INIT, FittingStage.EM_GMM]

            if self.config.model_type in MarkovTypes:
                default_actions.append(FittingStage.EM_HMM)

            fitting_actions = [d for d in default_actions if d > self.fitting_stage]
        elif type(fitting_actions) in (list, tuple):
            fitting_actions = list(fitting_actions)
        elif type(fitting_actions) is fitting_actions:
            fitting_actions = [fitting_actions]  # type: ignore
        else:
            raise ValueError(f"Invalid fitting actions dtype: {type(fitting_actions)}")

        assert fitting_actions == sorted(
            fitting_actions  # type: ignore
        ), "Fitting actions must be sorted."
        assert len(fitting_actions) == len(  # type: ignore
            set(fitting_actions)  # type: ignore
        ), "Fitting actions must be unique."

        logger.info(f"Performing fitting actions: {fitting_actions}")

        return fitting_actions  # type: ignore

    def fit_trajectories(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
        plot_optim: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Fit the model to the given trajectories.

        Parameters
        ----------
        demos : dataset.Demos
            Demonstration class containing the trajectories to fit.

        fix_frames : bool
            If True, fix the coordinate frames to their location in the first
            observation per trajectory. Default: True

        init_strategy : InitStrategy
            Strategy to use for initialization. If None, uses TIME_BASED iff
            add_time_dim is True, else KMEANS. Default: None

        fitting_actions : Sequence[FittingStage]
            List of fitting actions to perform. If None, performs all default
            actions of the model class. Default: None

        plot_optim : bool
            If True, plot the rbd optimization progress. Default: False

        Returns
        -------
        np.ndarray, float
            Likelihoods and average log likelihoods of the fitting.
            Likelihoods have shape (manifold_dim, T) where T is the sum of all
            trajectory lengths.
        """

        if self._fix_frames is None:
            self._fix_frames = fix_frames
        else:
            assert (
                self._fix_frames == fix_frames
            ), "Should use consistent fix_frames for all fitting calls."

        self._setup_data(demos)

        self._setup_model()

        fitting_actions = self._select_fitting_actions(fitting_actions)

        if init_strategy is None and FittingStage.INIT in fitting_actions:
            init_strategy = (
                InitStrategy.TIME_BASED
                if self.config.add_time_component
                else InitStrategy.KMEANS
            )
            logger.info(f"Init strategy not specified. Auto selected {init_strategy}.")

        if self.use_riemann:
            lik, avg_loglik = self._fit_rbd(fitting_actions, init_strategy, plot_optim)
        else:
            raise NotImplementedError("Dropping support for pbdlib.")

        return lik, avg_loglik

    def _fit_rbd(
        self,
        fitting_actions: list[FittingStage],
        init_strategy: InitStrategy,
        plot_optim: bool = False,
    ) -> tuple[np.ndarray, float]:
        for fa in fitting_actions:
            if fa in [FittingStage.INIT, FittingStage.EM_GMM]:
                gmm_input = self._prepare_input_fitting(
                    self.demo_data_flat, model_type=ModelType.GMM
                )

            if fa is FittingStage.INIT:
                lik, avg_loglik = self._fit_rbd_init(
                    init_strategy, gmm_input, plot_optim  # type: ignore
                )

            elif fa is FittingStage.EM_GMM:
                lik, avg_loglik = self._fit_rbd_gmm(gmm_input, plot_optim)  # type: ignore

            elif fa is FittingStage.EM_HMM:
                lik, avg_loglik = self._fit_rbd_hmm()

            else:
                raise ValueError(f"Unknown fitting action: {fa}")

            self.fitting_stage = fa

        return lik, avg_loglik

    def _fit_rbd_init(
        self,
        init_strategy: InitStrategy,
        gmm_input: np.ndarray,
        plot_optim: bool = False,
    ) -> tuple[np.ndarray, float]:
        logger.info("Model init ...")

        assert self._demos is not None
        assert self.model is not None

        if init_strategy is InitStrategy.TIME_BASED:
            if self.config.add_time_component:
                init_input = gmm_input
            else:
                time_based_data_flat = self._demos.get_per_frame_data(
                    flat=True,
                    add_time_dim=True,
                    subsampled=True,
                    fixed_frames=self._fix_frames,
                    pos_only=self.config.position_only,
                    skip_quat_dim=self.skip_quaternion_dim,
                    add_action_dim=self.config.config.add_action_component,
                    action_as_orientation=self.config.action_as_orientation,
                    action_with_magnitude=self.config.action_with_magnitude,
                    add_gripper_action=self.config.add_gripper_action,
                    numpy=True,
                )
                init_input = self._prepare_input_fitting(
                    time_based_data_flat, model_type=ModelType.GMM
                )

            lik, avglik, self._init_borders = self.model.init_time_based_from_np(  # type: ignore
                init_input,
                plot=plot_optim,
                drop_time=not self.config.add_time_component,
                fix_first_component=self.config.fix_first_component,
                fix_last_component=self.config.fix_last_component,
                fixed_first_component_n_steps=self.config.fixed_first_component_n_steps,
                fixed_last_component_n_steps=self.config.fixed_last_component_n_steps,
                reg_type=self.config.reg_init_type,
                reg_lambda=self.config.reg_init_diag,
            )

        elif init_strategy is InitStrategy.KMEANS:
            lik, avglik, self._init_borders = self.model.kmeans_from_np(  # type: ignore
                gmm_input, reg_type=rbd.statistics.RegularizationType.NONE
            )
        elif init_strategy is InitStrategy.SAMMI:
            em_kwargs = dict(
                reg_lambda=self.config.reg_shrink,
                reg_lambda2=self._full_reg_diag,
                reg_type=self.config.reg_type,
                minsteps=1,
                maxsteps=15,
                plot=False,
            )
            kmeans_kwargs = dict(
                reg_type=rbd.statistics.RegularizationType.NONE, plot=False
            )

            lik, avglik, self._init_borders = self.model.sammi_init(  # type: ignore
                self.demo_data_flat,
                includes_time=self.config.add_time_component,
                fixed_component_from_last_step=self.config.fix_last_component,
                fixed_component_from_first_step=self.config.fix_first_component,
                plot_cb=self.plot_model,
                debug_borders=True,
                em_kwargs=em_kwargs,
                kmeans_kwargs=kmeans_kwargs,
            )
        else:
            raise ValueError(f"Unexpected init strategy {init_strategy}")

        # logger.info(
        #     f"Model initizalized with mean {self.model.mu}" # and " \
        #     # f"covariance {self.model.sigma}"
        # )

        return lik, avglik

    def _fit_rbd_gmm(
        self, gmm_input: np.ndarray, plot_optim: bool = False
    ) -> tuple[np.ndarray, float]:
        logger.info("GMM EM ...")

        em_kwargs = dict(
            reg_lambda=self.config.reg_shrink,
            reg_lambda2=self._full_reg_diag,
            reg_type=self.config.reg_type,
            minsteps=1,
            maxsteps=self.config.em_steps,
            fix_last_component=self.config.fix_last_component,
            fix_first_component=self.config.fix_first_component,
            plot=plot_optim,
        )

        if self.config.model_type is ModelType.GMM:
            lik, avg_loglik = self.model.fit_from_np(gmm_input, **em_kwargs)
        else:
            # TODO: other init strategies?
            # TODO: get likelihoods
            lik, avg_loglik = self.model.gmm_init(gmm_input, **em_kwargs)

        return lik, avg_loglik

    def _fit_rbd_hmm(self) -> tuple[np.ndarray, float]:
        logger.info("HMM EM ...")

        # TODO: get and return likelihoods

        hmm_input = self._prepare_input_fitting(
            self.demo_data_flat, model_type=ModelType.HMM
        )

        mle_kwargs = dict(
            reg_lambda=self.config.reg_shrink,
            reg_lambda2=self._full_reg_diag,
            reg_type=self.config.reg_type,
        )

        finish_kwargs = dict(
            reg_lambda=self.config.reg_em_finish_shrink,
            reg_lambda2=self._full_reg_em_finish_diag,
            reg_type=self.config.reg_em_finish_type,
        )
        # finish_kwargs = None

        em_hmm_kwargs = dict(
            nb_max_steps=self.config.em_steps,
            mle_kwargs=mle_kwargs,
            finish_kwargs=finish_kwargs,
            fix_last_component=self.config.fix_last_component,
            fix_first_component=self.config.fix_first_component,
            left_to_right=self.config.reg_em_left_to_right,
            trans_reg=self.config.reg_em_trans,
            # dep_mask=self._em_dep_mask,
        )  # TODO: table/dep?

        assert self.config.model_type in MarkovTypes

        lik, avg_loglik = self.model.em(hmm_input, **em_hmm_kwargs)

        return lik, avg_loglik

    def reconstruct(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
        model: GMM | None = None,
    ) -> tuple[list[list[GMM]], list[list[GMM]], list[GMM], Any]:
        """
        Reconstruct the given demonstrations.

        Parameters
        ----------
        demos : Demos, optional
            Demos to reconstruct, by default self._demos
        use_ss : bool, optional
            Whether to subsample the trajectories to common length,
            by default False
        dbg : bool, optional
            Debug plot switch, by default False
        strategy : ReconstructionStrategy, optional
            by default ReconstructionStrategy.LQR_VITERBI
        time_based : bool, optional
            Whether to use time-based reconstruction, by default None
            If None, decided based on self.add_time_dim.
        model : GMM, optional
            Model to use for reconstruction, by default uses self.model.

        Returns
        -------
        list[list[GMM]], list[list[GMM]], list[GMM], <Reconstrunction>
            Local marginals, global marginals, global model and reconstruction
            artifacts. See respective reconstruction strategy for details.

        """
        if self.fitting_stage < FittingStage.EM_GMM:
            logger.error(
                f"Model not fitted yet. Fitting stage: {self.fitting_stage.name}."
            )
            raise RuntimeError("Model not fitted yet.")

        if strategy is None:
            # strategy = ReconstructionStrategy.GMR if self.model_type \
            #     is ModelType.GMM else ReconstructionStrategy.LQR_VITERBI
            strategy = ReconstructionStrategy.GMR
            logger.info(f"Selected reconstruction strategy {strategy}.")

        if time_based:
            assert (
                self.config.add_time_component
            ), "Need time-based model for time-based reconstruction."

        if time_based is None:
            time_based = self.config.add_time_component
            logger.info(
                f"Time-based reconstruction not specified. Auto selected {time_based}."
            )

        fix_frames = self._fix_frames
        add_action_dim = self.config.add_action_component
        heal_time_variance = time_based and self.config.heal_time_variance

        world_data, frame_trans, frame_quats = self.get_gmr_data(
            use_ss=use_ss,
            time_based=time_based,
            dbg=dbg,
            demos=demos,
        )

        local_marginals, trans_marginals, joint_models = self.get_marginals_and_joint(
            model=model,
            fix_frames=fix_frames,
            heal_time_variance=heal_time_variance,
            frame_trans=frame_trans,
            frame_quats=frame_quats,
            time_based=time_based,
        )

        if strategy is ReconstructionStrategy.GMR:
            ret = self._gmr(
                world_data, joint_models, fix_frames, time_based, add_action_dim
            )
        elif strategy is ReconstructionStrategy.LQR_VITERBI:
            ret = self._lqr_viterbi(
                demos, world_data, joint_models, fix_frames, False, add_action_dim
            )
        else:
            raise NotImplementedError(
                "Unexpected reconstruction strategy: {strategy}".format(
                    strategy=strategy
                )
            )

        return local_marginals, trans_marginals, joint_models, ret

    def get_marginals_and_joint(
        self,
        fix_frames: bool,
        time_based: bool,
        heal_time_variance: bool,
        frame_trans: np.ndarray | list[np.ndarray],
        frame_quats: np.ndarray | list[np.ndarray],
        model: GMM | None = None,
    ):
        """
        Wrapper around get_frame_marginals, transform_marginals and join_marginals.
        """
        local_marginals = self.get_frame_marginals(time_based=time_based, model=model)

        join_cov_mask = self._join_cov_mask
        trans_cov_mask = self._trans_cov_mask
        if not time_based and self.add_time_component:
            join_cov_mask = None if join_cov_mask is None else join_cov_mask[1:, 1:]
            trans_cov_mask = None if trans_cov_mask is None else trans_cov_mask[1:, 1:]

        trans_marginals = transform_marginals(
            fix_frames=fix_frames,
            drop_time=not time_based,
            local_marginals=local_marginals,
            frame_trans=frame_trans,
            frame_quat=frame_quats,
            use_riemann=self.use_riemann,
            patch_func=self.patch_frame_transforms,
            reg_kwargs=self._trans_cov_reg_kwargs,
            cov_mask=trans_cov_mask,
        )

        joint_models = join_marginals(
            marginals=trans_marginals,
            fix_frames=fix_frames,
            heal_time_variance=heal_time_variance and time_based,
            cov_mask=join_cov_mask,
        )

        return local_marginals, trans_marginals, joint_models

    def get_joint_model(
        self,
        fix_frames: bool,
        heal_time_variance: bool,
        frame_trans: np.ndarray | list[np.ndarray],
        frame_quats: np.ndarray | list[np.ndarray],
        model: GMM | None = None,
    ):
        _, _, joint_models = self.get_marginals_and_joint(
            fix_frames=fix_frames,
            heal_time_variance=heal_time_variance,
            frame_trans=frame_trans,
            frame_quats=frame_quats,
            model=model,
        )

        return joint_models

    def get_gmr_data(
        self,
        use_ss: bool = False,
        time_based: bool = False,
        dbg: bool = False,
        demos: Demos | None = None,
    ) -> tuple[NDArrayOrNDArraySeq, NDArrayOrNDArraySeq, NDArrayOrNDArraySeq]:
        if demos is None:
            assert self._data_check()
            demos = self._demos

        world_data, frame_trans, frame_quats = demos.get_gmr_data(  # type: ignore
            use_ss=use_ss,
            fix_frames=self._fix_frames,
            position_only=self.config.position_only,
            skip_quaternion_dim=self.skip_quaternion_dim,
            add_time_dim=time_based,
            add_action_dim=self.config.add_action_component,
            add_gripper_action=self.config.add_gripper_action,
            action_as_orientation=self.config.action_as_orientation,
            action_with_magnitude=self.config.action_with_magnitude,
            numpy=True,
        )

        if dbg:
            plot_coordinate_frame_trajs(frame_trans)

        return world_data, frame_trans, frame_quats

    def _lqr_viterbi(
        self,
        demos,
        world_data,
        joint_models,
        fix_frames=False,
        time_based=False,
        add_action_dim=False,
    ):
        if time_based or add_action_dim:
            raise NotImplementedError(
                "Didn't yet update for time_based, " "add_action_dim."
            )
        # if self.use_riemann:
        #     raise NotImplementedError("Didn't yet update for rbd")

        state_sequences = []
        original_trajectories = []
        reconstructions = []

        for i in tqdm(range(demos.n_trajs), desc="LQR Viterbi"):
            # get the most probable sequence of state for this demonstration
            raise NotImplementedError
            # TODO: need to update viterbi-call. Sofar, gave data in all frames
            # to self.model. Should be able to give world_data to joint model
            # instead.
            # If for some reason fails, can also pass the full all-frame data
            # as well and continue as before.
            # TRY: sq = joint_models[i].viterbi(world_data[i])
            sq = self.model.viterbi(xdx[i])
            state_sequences.append(sq)

            # solving LQR with Product of Gaussian, see notebook on LQR
            lqr_dim = 3 if self.position_only else 6  # pos or pos + rot
            lqr = pbd.PoGLQR(nb_dim=lqr_dim, dt=0.05, horizon=len(sq))

            if fix_frames:
                lqr.mvn_xi = joint_models[i].concatenate_gaussian(sq)
            else:
                # Get the model at each timestep first and then concat again
                xi_per_step = [
                    jm.concatenate_gaussian([s]) for s, jm in zip(sq, joint_models[i])
                ]
                lqr.mvn_xi = concat_mvn(xi_per_step)
            lqr.mvn_u = -4.0
            lqr.x0 = world_data[i][0][: lqr_dim * 2]  # dim is x + dx, so 2*lqr_dim

            original_trajectories.append(world_data[i][:, : lqr_dim * 2])

            xi = lqr.seq_xi
            reconstructions.append(xi)

        return state_sequences, reconstructions, original_trajectories

    def _gmr(
        self,
        trajs: NDArrayOrNDArraySeq,
        joint_models: list[GMM] | list[list[GMM]],
        fix_frames: bool = False,
        time_based: bool = False,
        with_action_dim: bool = False,
        sample: bool = False,
        mu_and_sigma: bool = True,
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        """
        Run Gaussian Mixture Regression on the joint models.

        Returns mean, mean and covariance, or samples from the GMM.

        Parameters
        ----------
        trajs : np.ndarray or list[np.ndarray]
            Prepared demo data returned by _get_data_and_marginals.
        joint_models : list[GMM] or list[list[GMM]]
            GMM per trajectory (and time) - depends on fix_frames.
        fix_frames : bool, optional
            Whether to use fixed coordinate frames per traj, by default False
        sample : bool, optional
            Whether to sample from the GMM, by default False
        mu_and_sigma : bool, optional
            Whether to return mean and covariance, by default True

        Returns
        -------
        list[np.ndarray], list[np.ndarray], dict[str, Any]
            Predicted data per trajectory and time step, original data, and
            extra data (covariance, input data) as dict.
        """

        assert sample is not mu_and_sigma

        first_model = joint_models[0] if fix_frames else joint_models[0][0]
        n_submanis = len(first_model.manifold.get_submanifolds())

        original_trajs_on_mani = []
        reconstructions = []

        extras = {}

        extras["input"] = []

        if mu_and_sigma:
            extras["cov"] = []

        m_in = [0] if time_based or not self.add_rotation_component else [0, 1]
        m_out = list(range(m_in[-1] + 1, n_submanis))

        dim_in = (
            1
            if time_based
            else 3 if not self.add_rotation_component else 7 if self.use_riemann else 6
        )

        n_trajs = len(trajs)

        for i in tqdm(range(n_trajs), desc="GMR"):
            if self.use_riemann:
                traj = trajs[i]

                if fix_frames:
                    cond = [
                        joint_models[i].gmr_from_np(
                            traj[j, :dim_in], i_in=m_in, i_out=m_out, initial_obs=j == 0
                        )
                        for j in range(traj.shape[0])
                    ]

                    inp = [
                        joint_models[i].np_to_manifold_to_np(
                            traj[j, :dim_in], i_in=m_in
                        )
                        for j in range(traj.shape[0])
                    ]
                else:
                    cond = [
                        model.gmr_from_np(
                            traj[j, :dim_in], i_in=m_in, i_out=m_out, initial_obs=j == 0
                        )
                        for j, model in enumerate(joint_models[i])
                    ]

                    inp = np.array(
                        [
                            model.np_to_manifold_to_np(traj[j, :dim_in], i_in=m_in)
                            for j, model in enumerate(joint_models[i])
                        ]
                    )

                # gmr returns a list to work with multiple inputs. We only
                # feed one by one, so take first list element.
                if sample:
                    predictions = np.array([c[0].sample() for c in cond])
                else:
                    mu_sigma = [c[0].get_mu_sigma(as_np=True) for c in cond]
                    mu = np.array([m for m, _ in mu_sigma])
                    sigma = np.array([s for _, s in mu_sigma])

                    predictions = mu

                if fix_frames:
                    orig = joint_models[i].np_to_manifold_to_np(traj)
                else:
                    orig = np.array(
                        [
                            model.np_to_manifold_to_np(traj[j])
                            for j, model in enumerate(joint_models[i])
                        ]
                    )

            else:
                raise NotImplementedError

            reconstructions.append(np.concatenate((inp, predictions), axis=1))
            original_trajs_on_mani.append(orig)

            extras["cov"].append(sigma)
            extras["input"].append(inp)

        return reconstructions, original_trajs_on_mani, extras

    def _online_gmr(
        self,
        input_data: np.ndarray,
        joint_model: GMM,
        fix_frames: bool = False,
        time_based: bool = False,
        first_step: bool = False,
        sample: bool = False,
        mu_and_sigma: bool = True,
        on_tangent: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Online version of Gaussian Mixture Regression on the joint model. Should not be
        called directly, but via online_predict.
        """
        assert sample is not mu_and_sigma

        n_submanis = len(joint_model.manifold.get_submanifolds())
        m_in = [0] if time_based or not self.add_rotation_component else [0, 1]
        m_out = list(range(m_in[-1] + 1, n_submanis))

        extras = {}

        if self.use_riemann:
            if fix_frames:
                cond = joint_model.gmr_from_np(
                    input_data, i_in=m_in, i_out=m_out, initial_obs=first_step
                )[0]
                # inp = joint_model.np_to_manifold_to_np(input_data[:dim], i_in=m_in)
            else:
                # NOTE The translation of marginals and computation of the joint
                # model is already implemented in online_predict. However, for a
                # HMM, the thusly updated joint_model misses the observation history
                # in online_forward_message. Need to fix that.
                raise NotImplementedError

            if sample:
                prediction = cond.sample()
            else:
                mu, sigma = cond.get_mu_sigma(as_np=True, mu_on_tangent=on_tangent)

                prediction = mu
            extras["cov"] = sigma
            extras["input"] = input_data
            extras["mu_tangent"], _ = cond.get_mu_sigma(as_np=True, mu_on_tangent=True)
        else:
            raise NotImplementedError("Dropping support for pbdlib.")

        return prediction, extras

    def _prepare_input_fitting(self, input, model_type=None):
        if model_type is None:
            model_type = self.config.model_type

        if model_type is ModelType.GMM:
            return input.reshape(self._demos.n_trajs * self._demos.ss_len, -1)
        elif model_type is ModelType.HMM:
            return input
        elif model_type is ModelType.HSMM:
            return [i for i in input]
        else:
            raise NotImplementedError

    def _model_check(self):
        if is_none := self.model is None:
            logger.warning("No model fitted yet.")
        return not is_none

    def _manifold_check(self):
        if is_none := self.manifold is None:
            logger.warning("No manifold set up yet.")
        return not is_none

    def _data_check(self):
        if is_none := self._demos is None:
            logger.warning("No data fitted yet.")
        return not is_none

    def copy(self):
        other = TPGMM(self.config)

        other.manifold = copy(self.manifold)  # just in case
        other.model = self.model.copy()

        other._demos = self._demos
        # other.demo_data = self.demo_data
        other.demo_data_flat = self.demo_data_flat
        other._fix_frames = self._fix_frames

        other.fitting_stage = self.fitting_stage

        return other

    def plot_basis(self):
        if self.position_only:
            return

        mu, _ = self.model.get_mu_sigma(mu_on_tangent=False)
        logger.info(f"Basis is \n {mu}")
        mu_rot = tuple(
            tuple(i for i in m if isinstance(i, rbd.angular_representations.Quaternion))
            for m in mu
        )

        # n_rot_mani = len(mu_rot[0])

        # origins = tuple(tuple(
        #     rbd.angular_representations.Quaternion(1, np.zeros(3))
        #     for _ in range(n_rot_mani))
        #     for _ in range(self.n_components))

        plot_rotation_basis(mu_rot)

    def plot_model(
        self,
        plot_traj=True,
        plot_gaussians=True,
        scatter=False,
        rotations_raw=False,
        gaussian_mean_only=False,
        gaussian_cmap="Oranges",
        time_based=None,
        xdx_based=False,
        mean_as_base=False,
        annotate_gaussians=True,
        annotate_trajs=False,
        title=None,
        plot_derivatives=False,
        plot_traj_means=False,
        model: GMM | None = None,
        data: np.ndarray | None = None,
        size=None,
    ):
        """
        Plot the model in all frames. Splits the plot into position, rotation
        and their velocities. The Gaussians are plotted as ellipsoids.
        Overlays the demonstration data.

        Parameters
        ----------
        plot_traj : bool, optional
            Overlay the demo trajectories, by default True
        plot_gaussians : bool, optional
            Plot the Gaussian components, by default True
        scatter : bool, optional
            Use a scatter plot, otherwise standard (line) plot. Scatter plots
            are easier to read if there is little deviation in the data, eg.
            for the rotations. By default False.
        rotations_raw : bool, optional
            Plot the rotations as predicted by the model, ie. the three
            imaginary parts of the quaternion. Otherwise converts to Euler
            angles, which are easier to inspect but can be jumpy.
            By default False.
        gaussian_mean_only : bool, optional
            Only plot the mean of the Gaussians (no cov), by default False
        gaussian_cmap : str, optional
            Colormap for the Gaussians, by default 'Oranges'
            For time_based uses 'tab'
        time_based : bool, optional
            Plot the data as a function of time, otherwise as 3D points.
            By default None, which sets to True if a time dimension is present.
        xdx_based: bool, optional
            Make 2D plots mapping each input dimension to the corresponding
            output dimension (eg. x to dx), by default False
        mean_as_base : bool, optional
            Use the mean of the Gaussians as the base for the tangent space,
            otherwise use the origin. By default True. Only for Riemannian.
        """
        if not self._model_check() and model is None:
            return

        if not self._data_check() and data is None:
            return

        if model is None:
            model = self.model

        if data is None:
            data = self.demo_data_flat
            time_start = self._demos.relative_start_time
            time_stop = self._demos.relative_stop_time
        else:
            logger.warning("Using custom data. Can't infer time start, stop.")
            time_start = 0
            time_stop = 1

        if time_based is None:
            logger.info(
                "Did not specify time_based, deciding automatically.", filter=False
            )
            time_based = self.config.add_time_component

        plot_data = self._split_and_tangent_project_frame_data(
            data=data,
            model=model,
            time_based=time_based,
            xdx_based=xdx_based,
            mean_as_base=mean_as_base,
            rotations_raw=rotations_raw,
        )

        self._plot_data = plot_data

        if title is None:
            title = str(self.fitting_stage)

        if xdx_based:
            raise NotImplementedError("need to update to new plot data format")
            plot_gmm_xdx_based(
                pos_per_frame=pos,
                rot_per_frame=rot,
                pos_delta_per_frame=pos_delta,
                rot_delta_per_frame=rot_delta,
                model=model,
                frame_names=self._demos.frame_names,
                plot_traj=plot_traj,
                plot_gaussians=plot_gaussians,
                scatter=scatter,
                rot_on_tangent=self.use_riemann,
                gaussian_mean_only=gaussian_mean_only,
                cmap=gaussian_cmap,
                annotate_gaussians=annotate_gaussians,
                title=title,
                rot_base=rot_bases_per_frame,
                model_includes_time=self.add_time_dim,
            )
        elif time_based:
            plot_gmm_time_based(
                container=plot_data,
                plot_traj=plot_traj,
                plot_gaussians=plot_gaussians,
                rot_on_tangent=self.use_riemann,
                gaussian_mean_only=gaussian_mean_only,
                annotate_gaussians=annotate_gaussians,
                annotate_trajs=annotate_trajs,
                title=title,
                plot_derivatives=plot_derivatives,
                plot_traj_means=plot_traj_means,
                component_borders=self._init_borders,
                rot_to_degrees=not self.config.action_as_orientation,
                time_start=time_start,
                time_stop=time_stop,
                size=size,
            )
        else:
            plot_gmm_trajs_3d(
                container=plot_data,
                plot_traj=plot_traj,
                plot_gaussians=plot_gaussians,
                scatter=scatter,
                gaussian_mean_only=gaussian_mean_only,
                cmap_gauss=gaussian_cmap,
                annotate_gaussians=annotate_gaussians,
                time_start=time_start,
                time_stop=time_stop,
                title=title,
            )

    def _split_and_tangent_project_frame_data(
        self,
        data: np.ndarray | None,
        model: GMM,
        time_based: bool,
        xdx_based: bool,
        mean_as_base: bool = True,
        rotations_raw: bool = False,
    ) -> TPGMMPlotData:
        """
        Helper function to prepare the fitted data for plotting. Projects the data onto
        the tangent space and splits it into the individual plot components.
        """
        assert self._demos is not None

        if data is None:
            assert self._data_check()
            data = self.demo_data_flat

        if self.use_riemann:
            return self._split_and_tangent_project_data_rbd(
                demo_data_flat=data,
                model=model,
                mean_as_base=mean_as_base,
                time_based=time_based,
                xdx_based=xdx_based,
            )
        else:
            raise NotImplementedError("Dropping support for PBD.")

    def _get_component_mu_sigma_per_frame(
        self,
        mu: np.ndarray,
        mu_tan: np.ndarray,
        sigma: np.ndarray,
        within_frame_mani_idx: int,
        time_based: bool,
        xdx_based: bool,
        mu_on_tangent: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the plot ready mean and covariance of the given per-frame data component (ie
        position, rotation, position delta, rotation delta).
        """
        n_frames = self._demos.n_frames
        n_mani_per_frame = self._per_frame_manifold_n_submanis

        start_idx = int(self.config.add_time_component)  # time dim offset
        start_idx += within_frame_mani_idx

        mus = []
        sigmas = []

        for i in range(n_frames):
            mani_idx = start_idx + i * n_mani_per_frame
            start, stop = self._submani_indices[mani_idx]
            start_tan, stop_tan = self._submani_tangent_indices[mani_idx]

            idx_man = list(range(start, stop))
            idx_tan = list(range(start_tan, stop_tan))

            if time_based:
                idx_man = [0] + idx_man
                idx_tan = [0] + idx_tan
            elif xdx_based:
                if self.config.action_as_orientation:
                    raise NotImplementedError(
                        "XDX-style plotting not implemented for action as orientation. "
                        "Would not make a lot of sense, as R3 is not straight forward "
                        "to plot against S2."
                    )

                raise NotImplementedError
                # TODO: adapt the idx_man and idx_tan below
                # For xdx-based, need to get indieces for x (mani_idx) and dx,
                # which should be at mani_idx + 2

            if mu_on_tangent:
                mus.append(mu_tan[:, idx_tan])
            else:
                mus.append(mu[:, idx_man])
            sigmas.append(sigma[:, idx_tan, :][:, :, idx_tan])

        return np.stack(mus), np.stack(sigmas)

    def _get_component_mu_sigma_global(
        self,
        mu: np.ndarray,
        mu_tan: np.ndarray,
        sigma: np.ndarray,
        global_mani_idx: int,
        time_based: bool,
        xdx_based: bool,
        mu_on_tangent: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the plot ready mean and covariance of the given global component. The global
        components are the action magnitudes and gripper action (if applicable).
        """
        assert (
            mu_on_tangent
        ), "Only implemented for mu on tangent. For R1 that's identical anyway."

        if time_based:
            idx = [0, global_mani_idx]
        elif xdx_based:
            raise NotImplementedError("Not implemented for xdx-based.")
        else:
            raise NotImplementedError(
                "Only implemented global components that are R1/S1, ie not 3D."
            )

        return mu_tan[:, idx], sigma[:, idx, :][:, :, idx]

    def _get_frame_data_idx(self, mani_idx: int, tangent: bool) -> tuple[int, int]:
        """
        Get the indices of the frame data on the full manifold for the given submanifold
        index. The submanifold index indexes into the list of submanifolds per frame, ie
        position, rotation, position delta, rotation delta (if applicable).
        """
        data = self._submani_tangent_indices if tangent else self._submani_indices
        start, stop = data[mani_idx + int(self.config.add_time_component)]

        if self.config.add_time_component:  # frame data does not include time dim
            start -= 1
            stop -= 1

        return start, stop

    def _split_and_tangent_project_data_rbd(
        self,
        demo_data_flat: np.ndarray,
        model: GMM,
        mean_as_base: bool,
        time_based: bool,
        xdx_based: bool,
    ) -> TPGMMPlotData:
        """
        Project the flattened data onto the tangent space and split it into the
        individual plot components (position, rotation, etc.). Then package togtether
        with the respective Gaussian's mean and covariance and manifold information for
        plotting.
        """
        mu, sigma = model.get_mu_sigma(mu_on_tangent=False, as_np=True, stack=True)
        mu_tan, _ = model.get_mu_sigma(mu_on_tangent=True, as_np=True, stack=True)

        t_off = int(self.config.add_time_component)  # time dim offset
        a_idx = demo_data_flat.shape[-1] - self._global_action_dim_manifold

        # Remove time, global action dims and unflatten data over frames
        frame_data_flat: np.ndarray = demo_data_flat[..., t_off:a_idx]
        frame_data = np.stack(
            np.split(frame_data_flat, self._demos.n_frames, axis=-1),  # type: ignore
            axis=2,
        )

        if mean_as_base:
            base, rot_bases_per_frame = self._get_mean_base_for_plots()
        else:
            base = None
            rot_bases_per_frame = None

        # TODO: add rotation bases back to plot data

        tangent_data = np.stack(
            [model.np_to_manifold_to_np(d, base=base) for d in demo_data_flat]
        )

        a_idx_tan = tangent_data.shape[-1] - self._global_action_dim_tangent

        # Remove time, global action dims and unflatten data over frames
        tangent_frame_data_flat: np.ndarray = tangent_data[..., t_off:a_idx_tan]

        # TODO: this will break if the passed data is not self.demo_data_flat
        tangent_frame_data = tangent_frame_data_flat.reshape(
            self._demos.n_trajs, self._demos.ss_len, self._demos.n_frames, -1
        )

        plot_data = []

        # Position data
        man_frame_idx = 0

        pos_mu, pos_sigma = self._get_component_mu_sigma_per_frame(
            mu,
            mu_tan,
            sigma,
            man_frame_idx,
            time_based=time_based,
            xdx_based=xdx_based,
            mu_on_tangent=False,
        )
        data_start, data_stop = self._get_frame_data_idx(man_frame_idx, tangent=False)

        plot_data.append(
            SingleDimPlotData(
                data=frame_data[..., data_start:data_stop],
                name="pos",
                per_frame=True,
                manifold=Manifold_R3,
                mu=pos_mu,
                sigma=pos_sigma,
            )
        )

        man_frame_idx += 1

        # Rotation data
        if not self.config.position_only:
            rot_mu, rot_sigma = self._get_component_mu_sigma_per_frame(
                mu,
                mu_tan,
                sigma,
                man_frame_idx,
                time_based=time_based,
                xdx_based=xdx_based,
                mu_on_tangent=True,
            )
            data_start, data_stop = self._get_frame_data_idx(
                man_frame_idx, tangent=True
            )
            plot_data.append(
                SingleDimPlotData(
                    data=tangent_frame_data[..., data_start:data_stop],
                    name="rot (tangent)",
                    per_frame=True,
                    manifold=Manifold_Quat,
                    on_tangent=True,
                    mu=rot_mu,
                    sigma=rot_sigma,
                )
            )
            man_frame_idx += 1

        # Position delta data
        if self.config.add_action_component:
            pos_delta_mu, pos_delta_sigma = self._get_component_mu_sigma_per_frame(
                mu,
                mu_tan,
                sigma,
                man_frame_idx,
                time_based=time_based,
                xdx_based=xdx_based,
                mu_on_tangent=False,
            )
            data_start, data_stop = self._get_frame_data_idx(
                man_frame_idx, tangent=False
            )
            plot_data.append(
                SingleDimPlotData(
                    data=frame_data[..., data_start:data_stop],
                    name="delta pos"
                    + (" (orient)" if self.config.action_as_orientation else ""),
                    per_frame=True,
                    manifold=(
                        Manifold_S2
                        if self.config.action_as_orientation
                        else Manifold_R3
                    ),
                    mu=pos_delta_mu,
                    sigma=pos_delta_sigma,
                )
            )
            man_frame_idx += 1

            # Rotation delta data
            if not self.config.position_only:
                if self.config.action_as_orientation:
                    rot_delta_on_tangent = False
                    data_start, data_stop = self._get_frame_data_idx(
                        man_frame_idx, tangent=False
                    )
                    rot_delta = frame_data[..., data_start:data_stop]
                else:
                    rot_delta_on_tangent = True
                    data_start, data_stop = self._get_frame_data_idx(
                        man_frame_idx, tangent=True
                    )
                    rot_delta = tangent_frame_data[..., data_start:data_stop]
                rot_delta_mu, rot_delta_sigma = self._get_component_mu_sigma_per_frame(
                    mu,
                    mu_tan,
                    sigma,
                    man_frame_idx,
                    time_based=time_based,
                    xdx_based=xdx_based,
                    mu_on_tangent=rot_delta_on_tangent,
                )
                # data_start, data_stop = self._get_frame_data_idx(man_frame_idx)
                plot_data.append(
                    SingleDimPlotData(
                        data=rot_delta,
                        name="delta rot"
                        + (
                            " (tangent)"
                            if rot_delta_on_tangent
                            else (
                                " (orient)" if self.config.action_as_orientation else ""
                            )
                        ),
                        per_frame=True,
                        manifold=(
                            Manifold_S2
                            if self.config.action_as_orientation
                            else Manifold_Quat
                        ),
                        on_tangent=rot_delta_on_tangent,
                        mu=rot_delta_mu,
                        sigma=rot_delta_sigma,
                    )
                )

        if self._global_action_dim_tangent:
            global_dim_idx = 0
            tangent_global_action_data: np.ndarray = tangent_data[..., a_idx_tan:]

            if self.config.action_with_magnitude:
                pos_mag_mu, pos_mag_sigma = self._get_component_mu_sigma_global(
                    mu,
                    mu_tan,
                    sigma,
                    global_dim_idx + a_idx_tan,
                    time_based=True,  # Can't do 3D plots for R1
                    xdx_based=xdx_based,
                    mu_on_tangent=True,  # Identical for R1
                )
                plot_data.append(
                    SingleDimPlotData(
                        data=tangent_global_action_data[..., global_dim_idx],
                        name="delta pos (mag)",
                        per_frame=False,
                        manifold=Manifold_R1,
                        mu=pos_mag_mu,
                        sigma=pos_mag_sigma,
                    )
                )
                global_dim_idx += 1

            if self.config.action_with_magnitude and not self.config.position_only:
                rot_mag_mu, rot_mag_sigma = self._get_component_mu_sigma_global(
                    mu,
                    mu_tan,
                    sigma,
                    global_dim_idx + a_idx_tan,
                    time_based=True,  # Can't do 3D plots for R1
                    xdx_based=xdx_based,
                    mu_on_tangent=True,
                )
                plot_data.append(
                    SingleDimPlotData(
                        data=tangent_global_action_data[..., global_dim_idx],
                        name="delta rot (mag, tangent)",
                        per_frame=False,
                        manifold=Manifold_S1,
                        on_tangent=True,
                        mu=rot_mag_mu,
                        sigma=rot_mag_sigma,
                    )
                )
                global_dim_idx += 1

            if self.config.add_gripper_action:
                grip_mag_mu, grip_mag_sigma = self._get_component_mu_sigma_global(
                    mu,
                    mu_tan,
                    sigma,
                    global_dim_idx + a_idx_tan,
                    time_based=True,  # Can't do 3D plots for R1
                    xdx_based=xdx_based,
                    mu_on_tangent=True,
                )
                plot_data.append(
                    SingleDimPlotData(
                        data=tangent_global_action_data[..., global_dim_idx],
                        name="gripper action",
                        per_frame=False,
                        manifold=Manifold_R1,
                        mu=grip_mag_mu,
                        sigma=grip_mag_sigma,
                    )
                )

        multiplot_data = TPGMMPlotData(
            frame_names=self._demos.frame_names,
            dims=plot_data,
        )

        return multiplot_data

    def _get_mean_base_for_plots(self):
        mu = self.model.mu_raw
        mu_tangent = self.model.mu_tangent

        if self.config.action_as_orientation:
            logger.warning("Did not adapt mean computation for orientations")

            # average the mean over the components to use uniform base
            # for all components
            # mu = tuple(np.mean(m, axis=0) for m in zip(*mu))
            # NOTE: problem there is averaging the quaternions, so instead
            # just take the mean of the first component
        mu = mu[0]
        mu_tangent = mu_tangent[0]
        base = mu

        start = 1 if self.config.add_time_component else 0
        width = 4 if self.config.add_action_component else 2
        rot_bases_per_frame = [
            mu_tangent[i + 1 : i + 1 + width : 2]
            for i in range(start, len(mu_tangent), width)
        ]

        return base, rot_bases_per_frame

    def plot_model_frames(self, joint_models, trans_marginals, time_based=None):
        if not self._model_check():
            return

        if time_based is None:
            time_based = self.config.add_time_component

        if time_based:
            plot_gmm_frames_time_based(
                self.model,
                joint_models,
                trans_marginals,
                self._demos.frame_names,
                includes_rotations=not self.config.position_only,
            )
        else:
            plot_gmm_frames(
                self.model, joint_models, trans_marginals, self._demos.frame_names
            )

    def plot_model_components(self, joint_models, trans_marginals):
        if not self._model_check():
            return

        plot_gmm_components(
            self.model, joint_models, trans_marginals, self._demos.frame_names
        )

    def plot_tangent_data(self, per_frame, joint):
        plot_tangent_data(per_frame, joint)

    def plot_hmm_transition_matrix(self):
        if not self._model_check():
            return

        if self.fitting_stage < FittingStage.EM_HMM:
            logger.warning("Not yet fitted HMM.")
            return

        assert self.config.model_type in MarkovTypes

        plot_hmm_transition_matrix(self.model)

    def plot_state_sequence_for_demo(self, demo=None, demo_idx=None):
        if not self._model_check():
            return

        # assert self.model_type in MarkovTypes

        if demo is None:
            assert demo_idx is not None
            demo = self.demo_data_flat[demo_idx]

        plot_state_sequence_for_demo(demo, self.model)

    def plot_reconstructions(
        self,
        marginals: tuple[tuple[GMM]],
        joint_models: tuple[GMM],
        reconstructions: list[np.ndarray],
        original_trajectories: list[np.ndarray],
        demos: Demos | None = None,
        time_based: bool | None = None,
        frame_orig_wquats: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if not self._model_check():
            return

        if demos is None:
            assert self._demos is not None
            demos = self._demos

        if time_based is None:
            time_based = self.config.add_time_component

        if time_based:
            if frame_orig_wquats is None:
                frame_orig_wquats = demos._frame_origins_fixed_wquats.numpy()

            frame_origs_log = np.stack(
                [
                    j.np_to_manifold_to_np(f, i_in=[1, 2])
                    for j, f in zip(joint_models, frame_orig_wquats)
                ]
            )

            plot_reconstructions_time_based(
                marginals,
                joint_models,
                reconstructions,
                original_trajectories,
                frame_origs_log,
                demos.frame_names,
                includes_rotations=not self.config.position_only,
                includes_time=self.config.add_time_component,
                includes_actions=self.config.add_action_component,
                includes_action_magnitudes=self.config.action_with_magnitude,
                includes_gripper_actions=self.config.add_gripper_action,
                **kwargs,
            )
        else:
            plot_reconstructions_3d(
                marginals,
                joint_models,
                reconstructions,
                original_trajectories,
                demos._frame_origins_fixed,
                demos.frame_names,
                includes_time=self.config.add_time_component,
                includes_rotations=not self.config.position_only,
                **kwargs,
            )

    def reset_episode(self):
        self._online_first_step = True

    def _online_step_time(
        self, t_curr: float | np.ndarray, time_scale: float = 1.0
    ) -> float | np.ndarray:
        """
        Step the time for online prediction. Managed in the TPGMM instead of the policy
        because for the ATPGMM, the time steps change with the segments, which is
        easiest to manage here.
        """
        return t_curr + time_scale * self._t_delta

    def online_predict(
        self,
        input_data: np.ndarray,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        time_based: bool = False,
        model_contains_time: bool = False,
        local_marginals: tuple[GMM] | None = None,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        heal_time_variance: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Online prediction, ie prediction of the next step given the current
        observation and the sequence of previous prediction.
        Can reset sequence by setting first_step to True.

        Returns the current joint model (world frame) and the prediction.
        """
        fix_frames = self._fix_frames
        position_only = self.config.position_only
        use_riemann = self.use_riemann
        add_action_dim = self.config.add_action_component

        # intial time step or change in frame position
        if frame_trans is not None:
            assert frame_quats is not None and local_marginals is not None

            self._online_joint_model, _ = self.make_joint_model(
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                time_based=time_based,
                local_marginals=local_marginals,
                heal_time_variance=heal_time_variance,
                use_riemann=use_riemann,
            )

        if strategy is ReconstructionStrategy.GMR:
            prediction, extras = self._online_gmr(
                input_data=input_data,
                joint_model=self._online_joint_model,
                fix_frames=fix_frames,
                time_based=time_based,
                first_step=self._online_first_step,
            )
        else:
            raise NotImplementedError

        self._online_first_step = False

        return prediction, extras

    def make_joint_model(
        self,
        frame_trans: list[np.ndarray],
        frame_quats: list[np.ndarray],
        time_based: bool,
        local_marginals: tuple[GMM],
        heal_time_variance: bool,
        use_riemann: bool,
    ) -> tuple[GMM, tuple[GMM, ...]]:
        """
        Create the joint model from the local marginals and the frame transforms.
        For online prediction.
        """
        join_cov_mask = self._join_cov_mask
        trans_cov_mask = self._trans_cov_mask
        if not time_based and self.add_time_component:
            join_cov_mask = None if join_cov_mask is None else join_cov_mask[1:, 1:]
            trans_cov_mask = None if trans_cov_mask is None else trans_cov_mask[1:, 1:]

        trans_marginals = tuple(
            frame_transform_model(
                model=local_marginals[j],
                trans=frame_trans[j],
                quats=frame_quats[j],
                drop_time=not time_based,
                use_riemann=use_riemann,
                patch_func=self.patch_frame_transforms,
                reg_kwargs=self._trans_cov_reg_kwargs,
                cov_mask=trans_cov_mask,
            )
            for j in range(len(local_marginals))
        )

        joint_model = multiply_iterable(trans_marginals)
        if join_cov_mask is not None:
            joint_model.mask_covariance(mask=join_cov_mask)

        if time_based and heal_time_variance:
            joint_model = heal_time_based_model(joint_model, trans_marginals[0])

        return joint_model, trans_marginals

    def from_disk(self, file_name: str, force_config_overwrite: bool = False) -> None:
        logger.info("Loading model")

        with open(file_name, "rb") as f:
            ckpt = pickle.load(f)
        if self.config != ckpt.config:
            diff_string = recursive_compare_dataclass(ckpt.config, self.config)
            logger.error("Config mismatch\n" + diff_string)
            if force_config_overwrite:
                logger.warning(
                    "Overwriting config. This can lead to unexpected errors."
                )
                self.config = ckpt.config
            else:
                raise ValueError("Config mismatch")

        # Set members that are build during fitting, not init.
        self.manifold = ckpt.manifold
        self.model = ckpt.model
        self._demos = ckpt._demos
        self._fix_frames = ckpt._fix_frames
        self.fitting_stage = ckpt.fitting_stage

    def to_disk(self, file_name: str) -> None:
        logger.info("Saving model:")

        with open(file_name, "wb") as f:
            pickle.dump(self, f)


class AutoTPGMM(TPGMM):
    def __init__(self, config: AutoTPGMMConfig):
        self.config = config

        if not self.config.tpgmm.use_riemann:
            raise NotImplementedError("Dropping support for PBD.")

        self.use_riemann = config.tpgmm.use_riemann
        self.skip_quaternion_dim = None if self.use_riemann else 0

        self._fix_frames = None

        self.segment_gmms: list[TPGMM] | None = None
        self._demos: Demos | None = None
        self._demos_segments: tuple[DemosSegment, ...] | None = None
        self.segment_frames: list[tuple[int]] | None = None
        self.segment_frame_views: list[PartialFrameViewDemos] | None = None

        self._online_active_segment: int | None = None
        self._online_joint_models: tuple[GMM, ...] | None = None
        self._online_trans_margs_joint: tuple[GMM, ...] | None = None  # for dbg
        self._online_hmm_cascade: rbd.statistics.HMMCascade | None = None
        self._online_first_step: bool = True

    @property
    def add_rotation_component(self) -> bool:
        return not self.config.tpgmm.position_only

    @property
    def add_time_component(self) -> bool:
        return self.config.tpgmm.add_time_component

    @property
    def add_action_component(self) -> bool:
        return self.config.tpgmm.add_action_component

    @property
    def action_as_orientation(self) -> bool:
        return self.config.tpgmm.action_as_orientation

    @property
    def action_with_magnitude(self) -> bool:
        return self.config.tpgmm.action_with_magnitude

    @property
    def add_gripper_action(self) -> bool:
        return self.config.tpgmm.add_gripper_action

    @property
    def fitting_stage(self) -> FittingStage:
        return (
            self.segment_gmms[0].fitting_stage
            if self.segment_gmms
            else FittingStage.NONE
        )

    @cached_property
    def _used_frames(self) -> int:
        assert self.segment_frames is not None

        return np.unique(np.concatenate([np.array(f) for f in self.segment_frames]))

    @cached_property
    def _segment_lengths(self) -> np.ndarray:
        assert self._demos_segments is not None

        lens = np.array([s.mean_traj_len for s in self._demos_segments])
        return lens / lens.sum()

    def copy(self):
        logger.warning("Copy ATPGMM not properly tested yet.")
        other = AutoTPGMM(self.config)

        other.segment_gmms = (
            [m.copy() for m in self.segment_gmms]
            if self.segment_gmms is not None
            else None
        )

        other._demos_segments = self._demos_segments
        other.segment_frame_views = self.segment_frame_views
        other.segment_frames = self.segment_frames

        other._fix_frames = self._fix_frames

    def _create_tpgmm(self, overwrites: dict[str, Any] | None = None) -> TPGMM:
        config = self.config.tpgmm
        if overwrites is not None:
            config = dataclasses.replace(config, **overwrites)
        return TPGMM(config)

    def _setup_data(self, demos: Demos) -> None:
        if self._demos is None:
            self._demos = demos
        elif self._demos != demos:
            logger.warning("Overwriting demos. Need to reset the model?")
            self._demos = demos
        else:
            return

    def fit_trajectories(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
        plot_optim: bool = False,
        global_frames: bool = False,
    ) -> tuple[tuple[float], tuple[float]]:
        """
        Fit the model to the given trajectories.

        Upon initialization, segments the trajectories and selects the frames for each
        segment. Then fits a TPGMM to each segment.

        Returned liks and average log liks are per segment.
        """
        logger.info("Fitting AutoTPGMM", filter=False)

        if self._fix_frames is None:
            self._fix_frames = fix_frames
        else:
            assert (
                self._fix_frames == fix_frames
            ), "Should use consistent fix_frames for all fitting calls."

        self._setup_data(demos)

        fitting_actions = self._select_fitting_actions(fitting_actions)

        if FittingStage.INIT in fitting_actions:
            self._segment_and_frame_select(
                demos=demos,
                fix_frames=fix_frames,
                init_strategy=init_strategy,
                fitting_actions=fitting_actions,
                plot_optim=plot_optim,
                global_frames=global_frames,
            )

        assert self.segment_gmms is not None and self.segment_frame_views is not None

        liks = []
        avg_logliks = []

        with tqdm(total=len(self.segment_gmms), desc="Fitting segments") as pbar:
            for gmm, data in zip(self.segment_gmms, self.segment_frame_views):
                lik, avg_loglik = gmm.fit_trajectories(
                    data,
                    fix_frames=fix_frames,
                    init_strategy=init_strategy,
                    fitting_actions=fitting_actions,
                    plot_optim=plot_optim,
                )
                liks.append(lik)
                avg_logliks.append(avg_loglik)
                pbar.update(1)

        return tuple(liks), tuple(avg_logliks)

    def _get_segment_tpgmm_overwrites(
        self,
        n_segments: int,
        relative_duration: float | None = None,
        min_n_components: int = 1,
        drop_action_component: bool = False,
    ) -> dict[str, Any]:
        """
        Helper function for parameterizing the segment TPGMMs.
        Most config params are taken from the main TPGMM config, but some are overwritten.
        """

        n_components = (
            int(self.config.tpgmm.n_components * relative_duration)
            if self.config.demos_segmentation.components_prop_to_len
            else int(self.config.tpgmm.n_components / n_segments)
        )
        n_components = max(
            n_components,
            min_n_components,
            self.config.demos_segmentation.min_n_components,
        )

        return {
            "n_components": n_components,
            "fixed_first_component_n_steps": self.config.demos_segmentation.repeat_first_step
            or self.config.tpgmm.fixed_first_component_n_steps,
            "fixed_last_component_n_steps": self.config.demos_segmentation.repeat_final_step
            or self.config.tpgmm.fixed_last_component_n_steps,
            "add_action_component": self.config.tpgmm.add_action_component
            and not drop_action_component,
        }

    def _segment_and_frame_select(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
        plot_optim: bool = False,
        global_frames: bool = False,
    ) -> None:
        """
        Segment the given demos into sub-trajectories and select the relevant frames
        (task parameters) for each segment. If global_frames is True, the frames are
        selected based on the whole demos, otherwise per segment.
        """
        fs_init_strategy = self.config.frame_selection.init_strategy
        if fs_init_strategy is COPY_FROM_MAIN_FITTING:
            fs_init_strategy = init_strategy
        fs_fitting_actions = self.config.frame_selection.fitting_actions
        if fs_fitting_actions is COPY_FROM_MAIN_FITTING:
            fs_fitting_actions = fitting_actions

        self._fix_frames = fix_frames

        if global_frames:
            logger.info("Generating candidate frames", filter=False)
            candidate_ic, rel_candidate_ic, frame_idcs = self._select_frames(
                demos,
                fix_frames,
                fs_init_strategy,
                fs_fitting_actions,
                self.config.frame_selection.rel_score_threshold,
            )
            frame_names = tuple(demos.frame_names[i] for i in frame_idcs)

            logger.info(f"Selected global frames {frame_names}", filter=False)

            candidate_frame_ic = [candidate_ic]
            candidate_frame_rel_ic = [rel_candidate_ic]

        else:
            frame_idcs = None
            candidate_frame_ic = []
            candidate_frame_rel_ic = []

        logger.info("Segmenting trajectories", filter=False)
        if self.config.demos_segmentation.no_segmentation:
            segments = [demos]
        else:
            segments = self._segment_trajectories(demos, fix_frames)
        logger.info(f"... created {len(segments)} segments", filter=False)

        self._demos_segments = segments

        segment_gmms = []
        segement_frames = []
        segment_frame_views = []

        previous_segment_frame_idcs = None

        for s, segment in enumerate(segments):
            if global_frames:
                segment_frame_idcs = frame_idcs
            else:
                (
                    candidate_ic,
                    rel_candidate_ic,
                    segment_frame_idcs,
                ) = self._select_frames(
                    segment,
                    fix_frames,
                    init_strategy,
                    fs_fitting_actions,
                    self.config.frame_selection.rel_score_threshold,
                )
                candidate_frame_ic.append(candidate_ic)
                candidate_frame_rel_ic.append(rel_candidate_ic)

                # if segment_frame_idcs is disjoint for consecutive segments,
                # rank all not selected frames from both and add the best ones to the
                # respective segment where it is missing.
                if previous_segment_frame_idcs is not None and set(
                    previous_segment_frame_idcs
                ).isdisjoint(segment_frame_idcs):
                    logger.warning(
                        f"Selected frames for segments {s - 1} and {s} are disjoint. "
                    )
                    # take the frame from the previous segment that has the highest ic
                    mask = np.zeros_like(candidate_ic)
                    mask[np.array(previous_segment_frame_idcs)] = 1
                    best_frame = np.argmax(rel_candidate_ic * mask)

                    logger.info(f"Adding frame {best_frame} to segment {s}.")

                    segment_frame_idcs = tuple(
                        sorted(list(segment_frame_idcs) + [best_frame])
                    )

            if (gt_frames := self.config.frame_selection.gt_frames) is not None:
                logger.warning("Using manual frame selection for debugging.")
                segment_frame_idcs = gt_frames[s]

            min_n_components = (
                segment._n_gripper_states
                if self.config.demos_segmentation.use_gripper_states
                else 1
            )

            min_n_components += 1 if self.config.tpgmm.fix_first_component else 0
            min_n_components += 1 if self.config.tpgmm.fix_last_component else 0

            segement_frames.append(segment_frame_idcs)
            segment_gmm = self._create_tpgmm(
                overwrites=self._get_segment_tpgmm_overwrites(
                    n_segments=len(self._demos_segments),
                    relative_duration=segment.relative_duration,
                    min_n_components=min_n_components,
                )
            )
            frame_data = PartialFrameViewDemos(segment, list(segment_frame_idcs))

            segment_frame_views.append(frame_data)
            segment_gmms.append(segment_gmm)

            previous_segment_frame_idcs = segment_frame_idcs

        logger.info(f"Segmented trajs into {len(segment_gmms)} segments", filter=False)

        candidate_frame_ic = np.stack(candidate_frame_ic)
        candidate_frame_rel_ic = np.stack(candidate_frame_rel_ic)

        if global_frames:
            logger.info(f"Frame score (abs):\n{candidate_frame_ic}", filter=False)
            logger.info(f"Frame score (rel):\n{candidate_frame_rel_ic}", filter=False)
        else:
            abs_df = pd.DataFrame(
                candidate_frame_ic,
                columns=demos.frame_names,
                index=[f"Segment {i}" for i in range(len(segments))],
            )
            rel_df = pd.DataFrame(
                candidate_frame_rel_ic,
                columns=demos.frame_names,
                index=[f"Segment {i}" for i in range(len(segments))],
            )
            logger.info(f"Frame score (abs):\n{abs_df}", filter=False)
            logger.info(f"Frame score (rel):\n{rel_df}", filter=False)

        self.segment_gmms = segment_gmms
        self.segment_frames = segement_frames
        self.segment_frame_views = segment_frame_views

    def _select_frames(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
        rel_score_threshold: float = 0.6,
    ) -> tuple[np.ndarray, np.ndarray, tuple[int]]:
        """
        Select frames (task parameters) based on how informative they are for
        the trajectories.

        Assumes that the info per frame is independent, thus fits the data in
        all candidate frames, computes score and selects super-threshold ones.
        This is computatinally cheaper than iteratively fitting a joint model.
        """
        candidate_gmms, candidate_ic = self._fit_frames(
            demos, fix_frames, init_strategy, fitting_actions
        )

        if self.config.frame_selection.use_precision:
            # fing candiddates with identical precision across all components
            # are likely redundant, so drop all but the first one of each group
            doublicate_rows = get_indeces_of_duplicate_rows(candidate_ic)
            drop_candidates = (
                None
                if not self.config.frame_selection.drop_redundant_frames
                else (
                    np.concatenate([group[1:] for group in doublicate_rows])
                    if doublicate_rows
                    else None
                )
            )
            # relative precision across frames
            candidate_ic = candidate_ic / candidate_ic.sum(axis=0)
            candidate_ic = candidate_ic.max(axis=1)  # max over gaussians (time)

            candidate_ic = tuple(-candidate_ic)  # for consistency with AIC/BIC

        rel_candidate_ic = np.array(candidate_ic) / np.min(candidate_ic)

        for fr, bic, rel in zip(demos.frame_names, candidate_ic, rel_candidate_ic):
            logger.info(f"{fr:10} score (rel): {bic:6.0f} ({rel:.3f})")
        super_threshold = rel_candidate_ic > rel_score_threshold

        if drop_candidates is not None:
            logger.info(f"Dropping redundant frames {drop_candidates}.")
            super_threshold[drop_candidates] = False

        selected_idcs = np.argwhere(super_threshold)[:, 0]

        return (
            np.array(candidate_ic),
            rel_candidate_ic,
            tuple(selected_idcs),
        )

    def _fit_frames(
        self,
        demos: Demos,
        fix_frames: bool = True,
        init_strategy: InitStrategy | None = None,
        fitting_actions: Sequence[FittingStage] | None = None,
    ) -> tuple[list[TPGMM], list[float]]:
        """
        Fit a TPGMM to each candidate frame of the given demos and return the
        AIC/BIC/precision.
        """
        n_candidate_frames = demos.n_frames
        candidate_frame_names = demos.frame_names

        candidate_gmms = []
        candidate_score = []

        with indent_logs():
            for i in range(n_candidate_frames):
                logger.info(
                    f"Fitting candidate frame {i+1}/{n_candidate_frames}", filter=False
                )
                frame_gmm = self._create_tpgmm(
                    overwrites=self._get_segment_tpgmm_overwrites(
                        n_segments=(
                            1
                            if self._demos_segments is None
                            else len(self._demos_segments)
                        ),
                        relative_duration=demos.relative_duration,
                        drop_action_component=self.config.frame_selection.pose_only,
                    )
                )
                frame_data = PartialFrameViewDemos(demos, [i])

                lik, avg_loglik = frame_gmm.fit_trajectories(
                    frame_data,
                    fix_frames=fix_frames,
                    init_strategy=init_strategy,
                    fitting_actions=fitting_actions,
                )

                score = (
                    frame_gmm.model.precision_det
                    if self.config.frame_selection.use_precision
                    else (
                        frame_gmm.model.bic_from_lik(lik)
                        if self.config.frame_selection.use_bic
                        else frame_gmm.model.aic_from_lik(lik)
                    )
                )

                candidate_gmms.append(frame_gmm)
                candidate_score.append(score)

        if self.config.frame_selection.use_precision:
            # Return stacked instead of taking max here for finding redundant frames
            candidate_score = np.stack(candidate_score)

        return candidate_gmms, candidate_score

    def _segment_trajectories(
        self,
        demos: Demos,
        fix_frames: bool,
    ) -> tuple[DemosSegment, ...]:
        return demos.segment(
            min_len=self.config.demos_segmentation.min_len,
            distance_based=self.config.demos_segmentation.distance_based,
            gripper_based=self.config.demos_segmentation.gripper_based,
            velocity_based=self.config.demos_segmentation.velocity_based,
            distance_threshold=self.config.demos_segmentation.distance_threshold,
            repeat_first_step=self.config.demos_segmentation.repeat_first_step,
            repeat_final_step=self.config.demos_segmentation.repeat_final_step,
            fix_frames=fix_frames,
            min_end_distance=self.config.demos_segmentation.min_end_distance,
            velocity_threshold=self.config.demos_segmentation.velocity_threshold,
            gripper_threshold=self.config.demos_segmentation.gripper_threshold,
            max_idx_dist=self.config.demos_segmentation.max_idx_distance,
        )

    def _model_check(self):
        if is_none := self.segment_gmms is None:
            logger.warning("No model fitted yet.")
        return not is_none

    def calculate_segment_transition_probabilities(
        self,
        keep_time_dim: bool = True,
        keep_action_dim: bool = True,
        keep_rotation_dim: bool = True,
        models_are_sequential: bool = True,
        sigma_scale: float | None = None,
    ) -> tuple[np.ndarray, ...]:
        """
        For the sequence of segment models, calculate the transition probabilities
        between the segment. Assumes that the segment models are sequential, ie. there
        is no branching. (Can easily adapt to that - would need to compute transition
        prob between all pairs of segments).

        For state-action models only! Does not work for time-based models.

        NOTE: to use these probabilities, need to add them to the transition matrix and
        renormalize.
        """
        assert self._model_check()

        # assert self.model_type in MarkovTypes
        assert self.segment_frames is not None
        assert self.segment_gmms is not None

        # Get the common frames between the segments and the corresponding manifold idcs
        seg_frames: list[tuple[int]] = self.segment_frames

        pairwise_common_frames = tuple(
            sorted(set.intersection(set(f1), set(f2)))
            for f1, f2 in zip(seg_frames, seg_frames[1:])
        )

        idcs_of_common_frames_in_first_gmm = tuple(
            tuple(seg_frames[i].index(f) for f in common_frames)
            for i, common_frames in enumerate(pairwise_common_frames)
        )

        idcs_of_common_frames_in_second_gmm = tuple(
            tuple(seg_frames[i + 1].index(f) for f in common_frames)
            for i, common_frames in enumerate(pairwise_common_frames)
        )

        manifolds_per_frame = _get_rbd_manifolds_per_frame(
            self.config.tpgmm.position_only, self.config.tpgmm.add_action_component
        )

        # TODO: add global dims
        first_manifold_idcs = tuple(
            _get_rbd_manifold_indices(
                frames,
                time_based=self.config.tpgmm.add_time_component,
                manifolds_per_frame=manifolds_per_frame,
                keep_time_dim=keep_time_dim,
            )
            for frames in idcs_of_common_frames_in_first_gmm
        )

        second_manifold_idcs = tuple(
            _get_rbd_manifold_indices(
                f,
                time_based=self.config.tpgmm.add_time_component,
                manifolds_per_frame=manifolds_per_frame,
                keep_time_dim=keep_time_dim,
            )
            for f in idcs_of_common_frames_in_second_gmm
        )

        probs = tuple(
            hmm_transition_probabilities(
                g1.model,
                g2.model,
                i1,
                i2,
                drop_action_dim=not keep_action_dim,
                drop_rotation_dim=not keep_rotation_dim,
                includes_time=self.config.tpgmm.add_time_component and keep_time_dim,
                sigma_scale=sigma_scale,
                models_are_sequential=models_are_sequential,
            )
            for g1, g2, i1, i2 in zip(
                self.segment_gmms,
                self.segment_gmms[1:],
                first_manifold_idcs,
                second_manifold_idcs,
            )
        )

        if self.config.cascade.min_prob is not None:
            probs = tuple(np.maximum(p, self.config.cascade.min_prob) for p in probs)

        return probs

    def reconstruct(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
        per_segment: bool = False,
    ) -> tuple[
        tuple[list[list[GMM]]], tuple[list[list[GMM]]], tuple[list[GMM]], tuple[Any]
    ]:
        if demos is not None:
            raise NotImplementedError(
                "Can currently only reconstruct the fitted demos as novel demos would "
                "need to be segmented and the frames selected first."
            )

        # if self.fitting_stage < FittingStage.EM_GMM:
        #     logger.error(
        #         f"Model not fitted yet. Fitting stage: {self.fitting_stage.name}."
        #     )
        #     raise RuntimeError("Model not fitted yet.")

        if strategy is None:
            # strategy = ReconstructionStrategy.GMR if self.model_type \
            #     is ModelType.GMM else ReconstructionStrategy.LQR_VITERBI
            strategy = ReconstructionStrategy.GMR
            logger.info(f"Selected reconstruction strategy {strategy}.")

        if time_based:
            assert (
                self.config.tpgmm.add_time_component
            ), "Need time-based model for time-based reconstruction."

        if time_based is None:
            time_based = self.config.tpgmm.add_time_component
            logger.info(
                f"Time-based reconstruction not specified. Auto selected {time_based}."
            )

        per_segment = per_segment or len(self.segment_gmms) == 1

        if per_segment:
            return self._reconstruct_per_segment(
                demos=demos,
                use_ss=use_ss,
                dbg=dbg,
                strategy=strategy,
                time_based=time_based,
            )
        else:
            # assert self.model_type in MarkovTypes, "No cascading for GMMs."
            if self.fitting_stage == FittingStage.EM_HMM:
                return self._cascade_segment_hmms(
                    demos=demos,
                    use_ss=use_ss,
                    dbg=dbg,
                    strategy=strategy,
                    time_based=time_based,
                )
            else:
                return self._cascade_segment_gmms(
                    demos=demos,
                    use_ss=use_ss,
                    dbg=dbg,
                    strategy=strategy,
                    time_based=time_based,
                )

    def _reconstruct_per_segment(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
    ) -> tuple[
        tuple[list[list[GMM]]], tuple[list[list[GMM]]], tuple[list[GMM]], tuple[Any]
    ]:
        """
        Naive per-segment reconstruction that does not take into account transition
        between segment models. Works somewhat for time-based models.
        """
        if not time_based:
            logger.warning("Trying per-segment reconstruction with state-driven model.")

        local_marginals = []
        trans_marginals = []
        joint_models = []
        reconstructions = []

        for demo_seg, seg_gmm in zip(self.segment_frame_views, self.segment_gmms):
            loc, trans, joint, ret = seg_gmm.reconstruct(
                demos=demo_seg,
                use_ss=use_ss,
                dbg=dbg,
                strategy=strategy,
                time_based=time_based,
            )
            local_marginals.append(loc)
            trans_marginals.append(trans)
            joint_models.append(joint)
            reconstructions.append(ret)

        return (
            tuple(local_marginals),
            tuple(trans_marginals),
            tuple(joint_models),
            tuple((zip(*reconstructions))),
        )

    def _cascade_segment_hmms(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
    ):
        if self.fitting_stage < FittingStage.EM_HMM:
            logger.error(
                f"Transition model not fitted yet. Fitting stage: {self.fitting_stage.name}."
            )
            raise RuntimeError("Model not fitted yet.")

        if demos is None:
            demos = self._demos

        fix_frames = self._fix_frames
        add_action_dim = self.config.tpgmm.add_action_component
        heal_time_variance = time_based and self.config.tpgmm.heal_time_variance

        segment_transition_probs = self.calculate_segment_transition_probabilities(
            keep_time_dim=self.config.cascade.kl_keep_time_dim,
            keep_action_dim=self.config.cascade.kl_keep_action_dim,
            keep_rotation_dim=self.config.cascade.kl_keep_rotation_dim,
            models_are_sequential=self.config.cascade.models_are_sequential,
            sigma_scale=self.config.cascade.kl_sigma_scale,
        )

        logger.info(
            f"Caculated segment transition probabilities: {segment_transition_probs}",
            filter=False,
        )

        if (np.concatenate(segment_transition_probs) < 0.05).any():
            logger.warning(
                "At least one segment transition prob below 5%. Can lead to problems."
                "Consider increasing the diag reg."
            )

        local_marginals = []
        trans_marginals = []
        joint_models = []

        for frame_idcs, segment_gmm in zip(self.segment_frames, self.segment_gmms):
            segment_frame_view = PartialFrameViewDemos(demos, list(frame_idcs))

            world_data, frame_trans, frame_quats = segment_gmm.get_gmr_data(
                use_ss=use_ss,
                time_based=time_based,
                dbg=dbg,
                demos=segment_frame_view,
            )

            loc_marg, trans_marg, joint = segment_gmm.get_marginals_and_joint(
                fix_frames=fix_frames,
                time_based=time_based,
                heal_time_variance=heal_time_variance,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
            )

            local_marginals.append(loc_marg)
            trans_marginals.append(trans_marg)
            joint_models.append(joint)  # per segment and trajectory

        assert (
            fix_frames
        ), "Need to adapt the next two statements (joint_models, cascaded_hmms)"
        joint_models = tuple(zip(*joint_models))  # per trajectory and segment

        trans_marginals = tuple(zip(*trans_marginals))  # per trajectory, segment, frame
        trans_marginals_dict = tuple(dict() for _ in range(len(trans_marginals)))
        for dic, traj in zip(trans_marginals_dict, trans_marginals):
            for f_list, m_list in zip(self.segment_frames, traj):
                for f, m in zip(f_list, m_list):
                    if f not in dic:
                        dic[f] = []
                    dic[f].append(m)

        trans_marginals_joint = tuple(
            tuple(
                rbd.statistics.ModelList(traj[f]) if f in traj else None
                for f in range(self.n_frames)
            )
            for traj in trans_marginals_dict
        )

        cascaded_hmms = tuple(
            rbd.statistics.HMMCascade(
                segment_models=segment_models,
                transition_probs=segment_transition_probs,
            )
            for segment_models in joint_models
        )

        if strategy is ReconstructionStrategy.GMR:
            ret = self._gmr(
                trajs=world_data,
                joint_models=cascaded_hmms,
                fix_frames=fix_frames,
                time_based=time_based,
                with_action_dim=add_action_dim,
            )
        elif strategy is ReconstructionStrategy.LQR_VITERBI:
            raise NotImplementedError("LQR-Viterbi not implemented for ATPGMM yet.")
        else:
            raise NotImplementedError(
                "Unexpected reconstruction strategy: {strategy}".format(
                    strategy=strategy
                )
            )

        return (
            local_marginals,
            trans_marginals,
            trans_marginals_joint,
            joint_models,
            cascaded_hmms,
            ret,
        )

    def _cascade_segment_gmms(
        self,
        demos: Demos | None = None,
        use_ss: bool = False,
        dbg: bool = False,
        strategy: ReconstructionStrategy | None = None,
        time_based: bool | None = None,
    ):
        if self.fitting_stage < FittingStage.EM_GMM:
            raise RuntimeError("Model not fitted yet.")

        if demos is None:
            demos = self._demos

        fix_frames = self._fix_frames
        add_action_dim = self.config.tpgmm.add_action_component
        heal_time_variance = time_based and self.config.tpgmm.heal_time_variance

        local_marginals = []
        trans_marginals = []
        joint_models = []

        for frame_idcs, segment_gmm in zip(self.segment_frames, self.segment_gmms):
            segment_frame_view = PartialFrameViewDemos(demos, list(frame_idcs))

            world_data, frame_trans, frame_quats = segment_gmm.get_gmr_data(
                use_ss=use_ss,
                time_based=time_based,
                dbg=dbg,
                demos=segment_frame_view,
            )

            loc_marg, trans_marg, joint = segment_gmm.get_marginals_and_joint(
                fix_frames=fix_frames,
                time_based=time_based,
                heal_time_variance=heal_time_variance,
                frame_trans=frame_trans,
                frame_quats=frame_quats,
            )

            local_marginals.append(loc_marg)
            trans_marginals.append(trans_marg)
            joint_models.append(joint)  # per segment and trajectory

        assert (
            fix_frames
        ), "Need to adapt the next two statements (joint_models, cascaded_gmms)"
        joint_models = tuple(zip(*joint_models))  # per trajectory and segment

        trans_marginals = tuple(zip(*trans_marginals))  # per trajectory, segment, frame
        trans_marginals_dict = tuple(dict() for _ in range(len(trans_marginals)))
        for dic, traj in zip(trans_marginals_dict, trans_marginals):
            for f_list, m_list in zip(self.segment_frames, traj):
                for f, m in zip(f_list, m_list):
                    if f not in dic:
                        dic[f] = []
                    dic[f].append(m)

        trans_marginals_joint = tuple(
            tuple(
                rbd.statistics.ModelList(traj[f]) if f in traj else None
                for f in range(self.n_frames)
            )
            for traj in trans_marginals_dict
        )

        cascaded_gmms = tuple(
            rbd.statistics.GMMCascade(
                segment_models=segment_models,
                prior_weights=self._segment_lengths,
            )
            for segment_models in joint_models
        )

        if strategy is ReconstructionStrategy.GMR:
            ret = self._gmr(
                trajs=world_data,
                joint_models=cascaded_gmms,
                fix_frames=fix_frames,
                time_based=time_based,
                with_action_dim=add_action_dim,
            )
        elif strategy is ReconstructionStrategy.LQR_VITERBI:
            raise NotImplementedError("LQR-Viterbi not implemented for ATPGMM yet.")
        else:
            raise NotImplementedError(
                "Unexpected reconstruction strategy: {strategy}".format(
                    strategy=strategy
                )
            )

        return (
            local_marginals,
            trans_marginals,
            trans_marginals_joint,
            joint_models,
            cascaded_gmms,
            ret,
        )

    def reset_episode(self):
        """
        Reset the episode for online prediction. Ie. next prediction is treated as the
        first step of a new episode.
        """
        self._online_first_step = True
        self._online_active_segment = 0

    def get_frame_marginals(
        self, time_based: bool, models: tuple[GMM, ...] | None = None
    ) -> tuple[tuple[GMM, ...], ...]:
        """
        Split up the joint segment GMMs into per-frame marginals each.
        To be used for online prediction. In reconstruction, the segment GMMs should be
        used directly.
        """
        if models is None:
            assert self._model_check()
            models = self.segment_gmms

        return tuple(
            model.get_frame_marginals(time_based=time_based) for model in models
        )

    @cached_property
    def _segment_t_delta(self) -> tuple[float, ...]:
        return tuple(
            s._demos.relative_duration / s._demos.ss_len for s in self.segment_gmms
        )

    @cached_property
    def _segment_start_relative(self) -> tuple[float, ...]:
        return tuple(s.relative_start_time for s in self._demos_segments)

    @cached_property
    def _segment_stop_relative(self) -> tuple[float, ...]:
        return tuple(s.relative_stop_time for s in self._demos_segments)

    @property
    def _t_delta(self) -> float:
        if self._online_hmm_cascade is None:  # naive time sequencing
            idx = self._online_active_segment
            assert self.segment_gmms[idx]._demos is not None

            return 1 / self.segment_gmms[idx]._demos.ss_len
        else:
            if self.fitting_stage == FittingStage.EM_HMM:
                # Using the HMM cascade: weight sum of segment time delta by activation
                activation = self._online_hmm_cascade._alpha_tmp
                start_idcs = self._online_hmm_cascade.segment_start_idcs

                segment_activations = tuple(
                    np.sum(a) for a in np.split(activation, start_idcs)
                )
            else:
                # GMM cascade: weight by the relative segment length for now
                # TODO: consider using the segment activations?
                segment_activations = self._segment_lengths

            weighted_sum = sum(
                (a * d for a, d in zip(segment_activations, self._segment_t_delta))
            )

            return weighted_sum

    def _online_step_time(
        self, t_curr: float | np.ndarray, time_scale: float = 1.0
    ) -> float | np.ndarray:
        t_next = t_curr + time_scale * self._t_delta

        idx = self._online_active_segment

        # Naive time sequencing: just switch to the next segment when the time is up.
        if (
            self._online_hmm_cascade is None
            and t_next >= self._segment_stop_relative[idx]
            and idx < len(self.segment_gmms) - 1
        ):
            self._online_active_segment += 1
            self._online_first_step = True

        return t_next

    def online_predict(
        self,
        input_data: np.ndarray,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        time_based: bool = False,
        local_marginals: tuple[tuple[GMM]] | None = None,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        heal_time_variance: bool = False,
        per_segment: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._fix_frames:
            raise NotImplementedError(
                "Online prediction of ATPGMM only implemented for fixed frames. "
                "Currently batch-computing the joint models in the beginning to "
                "prevent lags due to computing the joint model of the current "
                "segment on the fly. "
                "Could store the marginals and transform them on-demand (maybe "
                "in a separate thread to prevent lags, or pause the execution)."
            )
        assert self._fix_frames is not None

        per_segment = per_segment or len(self.segment_gmms) == 1

        if frame_trans is not None:  # intial time step or change in frame position
            assert frame_quats is not None and local_marginals is not None

            (
                self._online_joint_models,
                self._online_trans_margs_joint,
            ) = self._make_segment_joint_models(
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                local_marginals=local_marginals,
                heal_time_variance=heal_time_variance,
                time_based=time_based,
            )

            if not per_segment:
                if self.fitting_stage == FittingStage.EM_HMM:
                    transition_probs = self.calculate_segment_transition_probabilities(
                        keep_time_dim=self.config.cascade.kl_keep_time_dim,
                        keep_action_dim=self.config.cascade.kl_keep_action_dim,
                        keep_rotation_dim=self.config.cascade.kl_keep_rotation_dim,
                        models_are_sequential=self.config.cascade.models_are_sequential,
                    )
                    logger.info(
                        f"Calculated segment transition probabilities: {transition_probs}",
                        filter=False,
                    )
                    self._online_hmm_cascade = rbd.statistics.HMMCascade(
                        segment_models=self._online_joint_models,
                        transition_probs=transition_probs,
                    )
                else:
                    self._online_hmm_cascade = rbd.statistics.GMMCascade(
                        segment_models=self._online_joint_models,
                        prior_weights=self._segment_lengths,
                    )
        if per_segment:
            assert time_based, "For State-Action need to use sequencing of models."
            ret = self._online_predict_per_segment(
                input_data=input_data,
                strategy=strategy,
                time_based=time_based,
            )
        else:
            ret = self._online_cascade_segment_hmms(
                input_data=input_data,
                strategy=strategy,
                time_based=time_based,
            )

        self._online_first_step = False

        return ret

    def batch_predict(
        self,
        input_data: np.ndarray,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        time_based: bool = False,
        local_marginals: tuple[tuple[GMM]] | None = None,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        heal_time_variance: bool = True,
        per_segment: bool = False,
    ) -> np.ndarray:
        """
        Batch predict the given input data using the segment models. Akin to
        online_predict, but over a full trajectory.
        Assumes that frames are fixed -> given in the same format as in online_predict.
        """

        predictions = tuple(
            self.online_predict(
                input_data=input_data[i],
                frame_trans=frame_trans,
                frame_quats=frame_quats,
                time_based=time_based,
                local_marginals=local_marginals,
                strategy=strategy,
                heal_time_variance=heal_time_variance,
                per_segment=per_segment,
            )
            for i in range(input_data.shape[0])
        )

        return np.stack(predictions)

    def _make_segment_joint_models(
        self,
        frame_trans: list[np.ndarray] | None,
        frame_quats: list[np.ndarray] | None,
        local_marginals: tuple[tuple[GMM]] | None = None,
        time_based: bool = True,
        heal_time_variance: bool = True,
    ) -> tuple[tuple[GMM, ...], tuple[rbd.statistics.ModelList, ...]]:
        joint_models = []
        trans_marginals = []

        for i, margs in enumerate(local_marginals):
            selected_trans = [frame_trans[j] for j in self.segment_frames[i]]
            selected_quats = [frame_quats[j] for j in self.segment_frames[i]]

            joint_model, trans_margs = self.segment_gmms[i].make_joint_model(
                frame_trans=selected_trans,
                frame_quats=selected_quats,
                time_based=time_based,
                local_marginals=margs,
                heal_time_variance=heal_time_variance,
                use_riemann=self.use_riemann,
            )
            joint_models.append(joint_model)
            trans_marginals.append(trans_margs)

        trans_marg_dict = dict()
        for f_list, m_list in zip(self.segment_frames, trans_marginals):
            for f, m in zip(f_list, m_list):
                if f not in trans_marg_dict:
                    trans_marg_dict[f] = []
                trans_marg_dict[f].append(m)

        trans_marg_joint = tuple(
            (
                rbd.statistics.ModelList(trans_marg_dict[f])
                if f in trans_marg_dict
                else None
            )
            for f in range(self.n_frames)
        )

        return tuple(joint_models), trans_marg_joint

    def _online_predict_per_segment(
        self,
        input_data: np.ndarray,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        time_based: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not time_based:
            raise NotImplementedError("No state-based segment-wise prediction.")

        idx = self._online_active_segment

        if strategy is ReconstructionStrategy.GMR:
            prediction, extras = self.segment_gmms[idx]._online_gmr(
                input_data=input_data,
                joint_model=self._online_joint_models[idx],
                fix_frames=self._fix_frames,
                time_based=True,
                first_step=self._online_first_step,
            )
        else:
            raise NotImplementedError

        return prediction, extras

    def _online_cascade_segment_hmms(
        self,
        input_data: np.ndarray,
        strategy: ReconstructionStrategy = ReconstructionStrategy.GMR,
        time_based: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if strategy is ReconstructionStrategy.GMR:
            prediction, extras = self._online_gmr(
                input_data=input_data,
                joint_model=self._online_hmm_cascade,
                fix_frames=self._fix_frames,
                time_based=time_based,
                first_step=self._online_first_step,
            )
        else:
            raise NotImplementedError

        return prediction, extras

    def plot_model(
        self,
        plot_traj=True,
        plot_gaussians=True,
        scatter=False,
        rotations_raw=False,
        gaussian_mean_only=False,
        gaussian_cmap="Oranges",
        time_based=None,
        xdx_based=False,
        mean_as_base=True,
        annotate_gaussians=True,
        annotate_trajs=False,
        title=None,
        plot_derivatives=False,
        plot_traj_means=False,
        per_segment: bool = False,
        size=None,
    ):
        if per_segment:
            for seg_model in self.segment_gmms:
                seg_model.plot_model(
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    rotations_raw=rotations_raw,
                    gaussian_mean_only=gaussian_mean_only,
                    gaussian_cmap=gaussian_cmap,
                    time_based=time_based,
                    xdx_based=xdx_based,
                    mean_as_base=mean_as_base,
                    annotate_gaussians=annotate_gaussians,
                    annotate_trajs=annotate_trajs,
                    title=title,
                    plot_derivatives=plot_derivatives,
                    plot_traj_means=plot_traj_means,
                    model=None,
                )
        else:
            if not self._model_check():
                return

            assert self._demos is not None

            if time_based is None:
                logger.info(
                    "Did not specify time_based, deciding automatically.", filter=False
                )
                time_based = self.config.tpgmm.add_time_component

            segment_data = [
                m._split_and_tangent_project_frame_data(
                    model=m.model,
                    data=None,
                    time_based=time_based,
                    xdx_based=xdx_based,
                    mean_as_base=mean_as_base,
                    rotations_raw=rotations_raw,
                )
                for m in self.segment_gmms
            ]

            all_frames = sorted(set.union(*[set(s) for s in self.segment_frames]))
            n_frames = len(all_frames)

            joint_data = []

            n_plot_rows = len(segment_data[0].dims)

            for i in range(n_plot_rows):
                dim_segments = [seg.dims[i] for seg in segment_data]

                n_time_steps = [d.data.shape[1] for d in dim_segments]
                n_time_total = sum(n_time_steps)
                time_starts = np.cumsum([0] + n_time_steps[:-1])
                time_stops = np.cumsum(n_time_steps)

                n_gaus_per_component = [d.mu.shape[-2] for d in dim_segments]
                n_gaus_total = sum(n_gaus_per_component)

                # labels to which segment each gaussian belongs
                segment_per_gaussian = np.concatenate(
                    [np.repeat(g, n) for g, n in enumerate(n_gaus_per_component)]
                )

                if len(dim_segments[0].data.shape) == 2:  # global data
                    dim_data = np.concatenate([d.data for d in dim_segments], axis=1)
                    per_frame = False

                    mus = np.concatenate([d.mu for d in dim_segments], axis=0)
                    sigmas = np.concatenate([d.sigma for d in dim_segments], axis=0)

                else:  # per frame data
                    n_trajs = dim_segments[0].data.shape[0]
                    n_dims = dim_segments[0].data.shape[3]

                    # initialize stacked data with NaNs to account for missing frames
                    dim_data = np.empty((n_trajs, n_time_total, n_frames, n_dims))
                    dim_data.fill(np.nan)

                    for s, t, (j, d) in zip(
                        time_starts, time_stops, enumerate(dim_segments)
                    ):
                        for k, frame_no in enumerate(self.segment_frames[j]):
                            global_frame_no = all_frames.index(frame_no)
                            dim_data[:, s:t, global_frame_no, :] = d.data[:, :, k, :]

                    per_frame = True

                    man_dim = dim_segments[0].mu.shape[-1]  # same as n_dims
                    tan_dim = dim_segments[0].sigma.shape[-1]

                    gaus_seg_starts = np.cumsum([0] + n_gaus_per_component[:-1])
                    gaus_seg_stops = np.cumsum(n_gaus_per_component)

                    mus = np.empty((n_frames, n_gaus_total, man_dim))
                    mus.fill(np.nan)
                    sigmas = np.empty((n_frames, n_gaus_total, tan_dim, tan_dim))
                    sigmas.fill(np.nan)

                    for s, t, (j, d) in zip(
                        gaus_seg_starts, gaus_seg_stops, enumerate(dim_segments)
                    ):
                        for k, frame_no in enumerate(self.segment_frames[j]):
                            global_frame_no = all_frames.index(frame_no)
                            local_mu = d.mu[k, ...]
                            local_sigma = d.sigma[k, ...]
                            mus[global_frame_no, s:t, :] = local_mu
                            sigmas[global_frame_no, s:t, ...] = local_sigma

                # TODO: fix the data. Assignement looks wrong.

                joint_data.append(
                    SingleDimPlotData(
                        data=dim_data,
                        name=segment_data[0].dims[i].name,
                        per_frame=per_frame,
                        manifold=segment_data[0].dims[i].manifold,
                        mu=mus,
                        sigma=sigmas,
                        gauss_labels=segment_per_gaussian,
                        base=segment_data[0].dims[i].base,
                        on_tangent=segment_data[0].dims[i].on_tangent,
                    )
                )

            frame_names = tuple(self._demos.frame_names[i] for i in all_frames)

            joint_container = TPGMMPlotData(frame_names=frame_names, dims=joint_data)

            segment_borders = [n / n_time_total for n in time_starts]

            if xdx_based:
                plot_gmm_xdx_based(
                    pos_per_frame=pos,
                    rot_per_frame=rot,
                    pos_delta_per_frame=pos_delta,
                    rot_delta_per_frame=rot_delta,
                    model=model,
                    frame_names=self._demos.frame_names,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    rot_on_tangent=self.use_riemann,
                    gaussian_mean_only=gaussian_mean_only,
                    cmap=gaussian_cmap,
                    annotate_gaussians=annotate_gaussians,
                    title=title,
                    rot_base=rot_bases_per_frame,
                    model_includes_time=self.config.tpgmm.add_time_component,
                    size=size,
                )
            elif time_based:
                plot_gmm_time_based(
                    container=joint_container,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    rot_on_tangent=self.use_riemann,
                    gaussian_mean_only=gaussian_mean_only,
                    annotate_gaussians=annotate_gaussians,
                    annotate_trajs=annotate_trajs,
                    title=title,
                    plot_derivatives=plot_derivatives,
                    plot_traj_means=plot_traj_means,
                    component_borders=segment_borders,
                    size=size,
                )
            else:
                plot_gmm_trajs_3d(
                    container=joint_container,
                    plot_traj=plot_traj,
                    plot_gaussians=plot_gaussians,
                    scatter=scatter,
                    gaussian_mean_only=gaussian_mean_only,
                    cmap_gauss=gaussian_cmap,
                    annotate_gaussians=annotate_gaussians,
                    title=title,
                    size=size,
                )

    def plot_reconstructions(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        per_segment: bool = False,
        **kwargs,
    ) -> None:
        if per_segment:
            assert (
                type(joint_models[0]) is not rbd.statistics.HMMCascade
            ), "Use joint reconstruction for cascaded models."
            self._plot_reconstructions_per_segment(
                marginals=marginals,
                joint_models=joint_models,
                reconstructions=reconstructions,
                original_trajectories=original_trajectories,
                time_based=time_based,
                **kwargs,
            )
        else:
            self._plot_reconstructions_jointly(
                marginals=marginals,
                joint_models=joint_models,
                reconstructions=reconstructions,
                original_trajectories=original_trajectories,
                time_based=time_based,
                **kwargs,
            )

    def _plot_reconstructions_per_segment(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        **kwargs,
    ) -> None:
        for model, marg, joint, rec, orig, data in zip(
            self.segment_gmms,
            marginals,
            joint_models,
            reconstructions,
            original_trajectories,
            self.segment_frame_views,
        ):
            model.plot_reconstructions(
                marginals=marg,
                joint_models=joint,
                reconstructions=rec,
                original_trajectories=orig,
                demos=data,
                time_based=time_based,
                **kwargs,
            )

    def _plot_reconstructions_jointly(
        self,
        marginals: tuple[tuple[tuple[GMM]]],
        joint_models: tuple[tuple[GMM]] | tuple[rbd.statistics.HMMCascade],
        reconstructions: tuple[list[np.ndarray]],
        original_trajectories: tuple[list[np.ndarray]],
        demos: Demos | None = None,
        time_based: bool | None = None,
        frame_orig_wquats: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if time_based is None:
            logger.info(
                "Did not specify time_based, deciding automatically.", filter=False
            )
            time_based = self.config.tpgmm.add_time_component

        if demos is None:
            demos = self._demos
        else:
            assert demos == self._demos

        reconstruction_is_per_segment = not isinstance(reconstructions[0], np.ndarray)

        if reconstruction_is_per_segment:
            joint_models = joint_models[0]

        marginals = tuple(
            tuple(traj[f] for f in self._used_frames) for traj in marginals
        )

        if time_based:
            if frame_orig_wquats is None:
                logger.info("Taking frame origins from demos.", filter=False)
                frame_orig_wquats = demos._frame_origins_fixed_wquats.numpy()

            frame_orig_wquats = frame_orig_wquats[:, self._used_frames]

            frame_names = [demos.frame_names[i] for i in self._used_frames]

            frame_origs_log = np.stack(
                [
                    j.np_to_manifold_to_np(f, i_in=[1, 2])
                    for j, f in zip(joint_models, frame_orig_wquats)
                ]
            )

            if reconstruction_is_per_segment:
                seg_lens = tuple(
                    tuple(len(r) for r in recs) for recs in zip(*reconstructions)
                )
            else:
                seg_lens = tuple(s.traj_lens for s in self._demos_segments)
                seg_lens = tuple(tuple(lens) for lens in zip(*seg_lens))
            seg_starts = tuple(np.cumsum([0] + list(lens[:-1])) for lens in seg_lens)
            traj_lens = tuple(np.sum(lens) for lens in seg_lens)
            seg_borders = tuple(
                tuple(starts / tlen) for starts, tlen in zip(seg_starts, traj_lens)
            )

        else:
            frame_origins = demos._frame_origins_fixed

            frame_origins = frame_origins[:, self._used_frames]

            frame_names = [demos.frame_names[i] for i in self._used_frames]

        if (
            "plot_gaussians" in kwargs
            and kwargs["plot_gaussians"]
            and reconstruction_is_per_segment
        ):
            logger.info(
                "Cannot plot gaussians for per segment reconstruction yet.",
                filter=False,
            )
            kwargs["plot_gaussians"] = False

        if reconstruction_is_per_segment:
            # Flatten over segments
            reconstructions_cat = tuple(
                np.concatenate([r for r in rec], axis=0)
                for rec in zip(*reconstructions)
            )
            original_trajectories_cat = tuple(
                np.concatenate([t for t in traj], axis=0)
                for traj in zip(*original_trajectories)
            )
        else:
            reconstructions_cat = reconstructions
            original_trajectories_cat = original_trajectories

        if time_based:
            plot_reconstructions_time_based(
                marginals,
                joint_models,
                reconstructions_cat,
                original_trajectories_cat,
                frame_origs_log,
                frame_names,
                includes_rotations=not self.config.tpgmm.position_only,
                includes_time=self.config.tpgmm.add_time_component,
                includes_actions=self.config.tpgmm.add_action_component,
                includes_action_magnitudes=self.config.tpgmm.action_with_magnitude,
                includes_gripper_actions=self.config.tpgmm.add_gripper_action,
                component_borders=seg_borders,
                **kwargs,
            )
        else:
            plot_reconstructions_3d(
                marginals,
                joint_models,
                reconstructions_cat,
                original_trajectories_cat,
                frame_origins,
                frame_names,
                includes_time=self.config.tpgmm.add_time_component,
                includes_rotations=not self.config.tpgmm.position_only,
                **kwargs,
            )

    def plot_hmm_transition_matrix(self):
        for seg_model in self.segment_gmms:
            seg_model.plot_hmm_transition_matrix()

    def from_disk(self, file_name: str, force_config_overwrite: bool = False) -> None:
        logger.info("Loading model:")

        with open(file_name, "rb") as f:
            ckpt = pickle.load(f)

        if self.config != ckpt.config:
            diff_string = recursive_compare_dataclass(ckpt.config, self.config)
            logger.error("Config mismatch\n" + diff_string)
            if force_config_overwrite:
                logger.warning(
                    "Overwriting config. This can lead to unexpected errors."
                )
                self.config = ckpt.config
            else:
                raise ValueError("Config mismatch")

        self._fix_frames = ckpt._fix_frames

        self.segment_gmms = ckpt.segment_gmms
        self._demos = ckpt._demos
        self._demos_segments = ckpt._demos_segments
        self.segment_frames = ckpt.segment_frames
        self.segment_frame_views = ckpt.segment_frame_views


def _get_rbd_manifolds_per_frame(position_only: bool, add_action_dim: bool) -> int:
    manifolds_per_frame = 1 if position_only else 2
    manifolds_per_frame *= 2 if add_action_dim else 1

    return manifolds_per_frame


def _get_rbd_manifold_indices(
    frame_idx: int | Sequence[int],
    time_based: bool,
    manifolds_per_frame: int,
    keep_time_dim: bool = True,
    with_global_dims: list[int] | None = None,
) -> list[int]:
    """
    Get indices of a frame's sub-manifolds in a Riemannian GMM.
    Also works for a sequence of frame indices. In that cases concatenates the
    indices.
    """
    if time_based:
        idx = [0] if keep_time_dim else []
        offset = 1
    else:
        idx = []
        offset = 0

    if isinstance(frame_idx, int):
        frame_idx = [frame_idx]

    for f in frame_idx:
        idx.extend(
            range(
                f * manifolds_per_frame + offset,
                (f + 1) * manifolds_per_frame + offset,
            )
        )
    if with_global_dims is not None:
        idx.extend(list(with_global_dims))

    return idx


def transform_marginals(
    fix_frames: bool,
    drop_time: bool,
    local_marginals: tuple[GMM],
    frame_trans: np.ndarray | list[np.ndarray],
    frame_quat: np.ndarray | list[np.ndarray],
    use_riemann: bool,
    patch_func: Callable,
    reg_kwargs: dict | None = None,
    cov_mask: np.ndarray | None = None,
) -> tuple[tuple[GMM]] | tuple[tuple[tuple[GMM]]]:
    """
    Transform the local marginal distributions to the global frame.

    Parameters
    ----------
    fix_frames : bool
        Whether to fix frame transformations across one trajectory.
    local_marginals : list[GMM]
        GMM per frame.
    frame_trans : np.ndarray or list[np.ndarray]
        Frame transformations per frame and trajectory - and time step if
        not fixed_frame.
    frame_quat : np.ndarray or list[np.ndarray]
        Frame quaternions per frame and trajectory - and time step if
        not fixed_frame. Used for Quaternion-based transformations to avoid
        conversions.
    use_riemann : bool
        Whether the GMMs are Riemannian or Euclidean.
    patch_func : Callable
        Function that patches the frame transformations to the manifold of the model.

    Returns
    -------
    tuple[tuple[GMM]]
        GMM per frame (and timestep if not fix_frames) and trajectory in global frame.
    """

    transformed_marginals = []

    n_trajs = len(frame_trans)
    n_frames = len(local_marginals)

    for i in tqdm(range(n_trajs), desc="Transforming marginals"):
        if fix_frames:
            trans_marginals = tuple(
                frame_transform_model(
                    model=local_marginals[j],
                    trans=frame_trans[i][j, 0],
                    quats=frame_quat[i][j, 0],
                    use_riemann=use_riemann,
                    drop_time=drop_time,
                    patch_func=patch_func,
                    reg_kwargs=reg_kwargs,
                    cov_mask=cov_mask,
                )
                for j in range(n_frames)
            )

        else:
            trans_marginals = tuple(
                tuple(
                    frame_transform_model(
                        model=local_marginals[j],
                        trans=frame_trans[i][j, k],
                        quats=frame_quat[i][j, k],
                        use_riemann=use_riemann,
                        drop_time=drop_time,
                        patch_func=patch_func,
                        reg_kwargs=reg_kwargs,
                        cov_mask=cov_mask,
                    )
                    for j in range(n_frames)
                )
                for k in range(frame_trans[i].shape[1])
            )

        transformed_marginals.append(trans_marginals)

    return tuple(transformed_marginals)


def frame_transform_model(
    model: GMM,
    trans: np.ndarray,
    drop_time: bool,
    quats: np.ndarray | None = None,
    use_riemann: bool = True,
    patch_func: Callable | None = None,
    reg_kwargs: dict | None = None,
    cov_mask: np.ndarray | None = None,
) -> GMM:
    """
    Transform a GMM by a homogeneous transformation. Pass the homogeneous transform
    matrix AND the corresponding quaternions to avoid inconsistencies caused by the
    conversion from quaternions to rotation matrices.

    The patch_func is a method of the TPGMM that maps the frame transformations to
    the transformations appropriate for the manifold of the model.

    reg_kwargs and cov_mask can be used to regularize the transformed covariances.
    reg_kwargs are passed to the regularize method of the GMM (see EM kwargs).
    cov_mask is a mask for the covariance matrix that is multiplied element-wise.
    """
    if use_riemann:
        A_joint, b_joint = patch_func(
            hom_trans=trans, quats=quats, model=model, drop_time=drop_time
        )

        return model.homogeneous_trans(
            A=A_joint, b=b_joint, reg_kwargs=reg_kwargs, mask=cov_mask
        )
    else:
        raise NotImplementedError


def join_marginals(
    marginals: tuple[tuple[GMM]] | tuple[tuple[tuple[GMM]]],
    fix_frames: bool,
    heal_time_variance: bool,
    cov_mask: np.ndarray,
) -> tuple[GMM] | tuple[tuple[GMM]]:
    """
    Join a tuple of marginal distributions in the same frame.

    Parameters
    ----------
    marginals : tuple[tuple[GMM]]
        Marginal distributions per frame and trajectory in global/identical
        frame.
    fix_frames : bool
        Whether frame transformations are fixed across one trajectory.

    Returns
    -------
    tuple[GMM]
        Joint GMM per trajectory.
    """
    joint_models = []

    # for m in marginals:
    #     for n in m:
    #         print(n.mu)

    for i in tqdm(range(len(marginals)), desc="Joining marginals"):
        if fix_frames:
            # print([m.mu for m in marginals[i]])
            joint = multiply_iterable(marginals[i])
            # print(joint.mu)
            if heal_time_variance and len(marginals[i]) > 1:
                joint = heal_time_based_model(joint, marginals[i][0])
            if cov_mask is not None:
                joint.mask_covariance(mask=cov_mask)
        else:
            joint = tuple(
                multiply_iterable(t)
                for t in tqdm(marginals[i], desc=f"Trajectory {i}", leave=False)
            )
            if heal_time_variance and len(marginals[i]) > 1:
                joint = tuple(
                    heal_time_based_model(j, t[0]) for j, t in zip(joint, marginals[i])
                )
            if cov_mask is not None:
                for j in joint:
                    j.mask_covariance(mask=cov_mask)
        joint_models.append(joint)

    return tuple(joint_models)


def _xdx_to_tangent(xdx, use_riemann, position_only, fix_frames):
    per_frame = []

    md = 6 if position_only else 14 if use_riemann else 12

    n_frames = xdx[0].shape[1] // md
    assert xdx[0].shape[1] % md == 0

    for i in tqdm(range(len(xdx)), desc="Local"):
        traj = []
        for f in range(n_frames):
            if use_riemann:
                if fix_frames:
                    man = [
                        geometry_np.log_e(xdx[i][j, f * md : (f + 1) * md])
                        for j in range(xdx[i].shape[0])
                    ]

                else:
                    man = np.array(
                        [
                            geometry_np.log_e(xdx[i][j, f * md : (f + 1) * md])
                            for j in range(xdx[i].shape[0])
                        ]
                    )

            else:
                raise NotImplementedError

            traj.append(man)

        per_frame.append(np.concatenate(traj, axis=1))

    return tuple(per_frame)


def heal_time_based_model(
    model: GMM,
    reference: GMM,
    healing_factor: float = 1,
    via_eigenvalues: bool = True,
) -> GMM:
    """
    Time-based models can have a small time-variance after multiplication of the marginals.
    This can lead to numerical issues in GMR.
    To fix this, this function scales either
    - the first eigenvalue or the whole covariance matrix.
    - OR the whole covariance matrix.
    to elongate the Gaussian in time without changing correlations.
    Scaling the first eigenvalue proportonaly scales the variance along the other dimensions, while
    naive scaling increases the variance along the other dimensions as well.

    In both cases, the scaling factor is computed by computing the quotient of the time-variance of
    joint gaussian and the reference gaussian (which should be one of the marginals). This quotient
    is then scaled by the healing factor, as complete healing might lead too elongated gaussians to
    properly fit the data. (Although the current default makes it even longer, but that
    smooths data better.)
    """

    for i, gaussian in enumerate(model.gaussians):
        t_var_orig = gaussian.sigma[0, 0]
        ref_t_var = reference.gaussians[i].sigma[0, 0]
        scale = ref_t_var / t_var_orig * healing_factor

        model.gaussians[i] = heal_gaussian_time_variance(
            gaussian, scale, via_eigenvalues
        )

    return model


def heal_gaussian_time_variance(
    gaussian: Gaussian, scale: float = 0.5, via_eigenvalues: bool = True
) -> Gaussian:
    if via_eigenvalues:
        eigenvalues, eigenvectors = np.linalg.eig(gaussian.sigma)
        eigenvalues[0] *= scale
        sigma = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)
    else:
        sigma = gaussian.sigma * scale

    return Gaussian(gaussian.manifold, gaussian.mu, sigma)
