from dataclasses import dataclass

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

# import pbdlib
import riepybdlib

# from mpl_toolkits.mplot3d import axes3d
import torch
from loguru import logger
from matplotlib import cm
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from tapas_gmm.utils.manifolds import (
    Manifold_Quat,
    Manifold_R1,
    Manifold_R3,
    Manifold_S1,
    Manifold_S2,
    Manifold_T,
)
from tapas_gmm.viz.threed import (
    colorline_3d,
    plot_coordindate_frame,
    plot_ellipsoid_from_gaussian3d,
    plot_rot_from_quaternion,
)
from tapas_gmm.viz.utils import set_ax_border

dim_names = tuple(("x", "y", "z"))

def_plot_titles = tuple(
    ("Position [m]", "Rotation (Tangent) [degr.]", "pos delta", "rot delta")
)

tab_colors = tuple(
    (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )
)

joint_color = "tab:orange"

dim_colors = tuple(("tab:red", "tab:green", "tab:blue"))

quat_colors = tuple(("tab:red", "tab:green", "tab:blue", "tab:brown"))

frame_colors = tuple(
    (
        "tab:purple",
        "tab:brown",
        "tab:olive",
        "tab:cyan",
        "tab:pink",
        "tab:gray",
    )
)


GMM = riepybdlib.statistics.GMM
HMM = riepybdlib.statistics.HMM
HSMM = riepybdlib.statistics.HSMM
Manifold = riepybdlib.manifold.Manifold


@dataclass
class SingleDimPlotData:
    data: np.ndarray
    name: str
    per_frame: bool
    manifold: Manifold

    mu: np.ndarray | None = None
    sigma: np.ndarray | None = None
    gauss_labels: np.ndarray | None = None

    # manifold_indeces: list[int] | None = None
    base: np.ndarray | None = None
    on_tangent: bool = False


@dataclass
class TPGMMPlotData:
    frame_names: list[str]
    dims: list[SingleDimPlotData]


def scatter_3d_traj(ax, traj, **kwargs):
    ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], **kwargs)


def scatter_3d_point(ax, point, **kwargs):
    ax.scatter(point[0], point[1], point[2], **kwargs)


def lineplot_3d_traj(ax, traj, c, **kwargs):
    line_collection = colorline_3d(traj[:, 0], traj[:, 1], traj[:, 2], c, **kwargs)
    ax.add_collection(line_collection)


class HandlerEllipse(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent, height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plot_ellipsoid_from_gaussian2d(ax, mean, cov, n_std=1, scale=1.96, **kwargs):
    try:
        U, s, rotation = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed for covariance matrix. Skipping Gaussian.")
        return
    else:
        radii = np.sqrt(s) * n_std
        plot_ellipse2d(ax, mean, rotation, radii, scale, **kwargs)


def plot_ellipse2d(ax, mean, rotation, radii, scale=1, **kwargs):
    width = radii[0] * scale
    height = radii[1] * scale

    angle1 = np.arccos(rotation[0, 0])
    # angle2 = np.arcsin(rotation[1, 0])
    # angle3 = - np.arcsin(rotation[0, 1])
    # angle4 = np.arccos(rotation[1, 1])
    angle = np.degrees(angle1)

    if np.sign(rotation[1, 0]) == -1:  # arccos maps to positive values only
        angle = -angle

    ellipse = Ellipse(mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)

    outer_ellipse = Ellipse(
        mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=ellipse.get_edgecolor(),
        alpha=1,
        fill=False,
    )
    ax.add_patch(outer_ellipse)


def plot_gmm_trajs_3d(
    container: TPGMMPlotData,
    plot_traj=True,
    plot_gaussians=True,
    size=(16, None),
    gaussian_mean_only=False,
    cmap_gauss="Dark2",
    plot_coord_origin=True,
    scatter=False,
    annotate_gaussians=False,
    annotate_trajs=False,
    title=None,
    s2_auto_rotate=True,
    s2_show_tangent=False,
    time_start=0,
    time_stop=1,
):
    """
    Plot a set of trajectories in all given coordinate frames, overlayed with
    the GMM components.

    Parameters
    ----------
    data: TPGMMPlotData
        Bundled data for plotting. Contains the trajectories, the model's mean
        and covariance, and infos about the manifold.
    frame_names: list[str]
        Names of the coordinate frames.
    size: tuple[float/int]
        Size of the figure.
        If size[1] is None, the height is automatically determined based on the
        width and the number of trajectories.
    cmap: str
        Name of the colormap to use for the GMM components.
    plot_coord_origin: bool
        Whether to plot the coordinate origin in each frame.
    scatter: bool
        Whether to plot the trajectories as scatter plot or as line plot.
        Scatter can help when values are very close to each other - line plots
        can fall apart in that case.
    rot_on_tangent: bool
        Whether the rotation is on the tangent space (Riemann), else: Euler.
    """
    # if rot_base is not None:
    #     logger.warning("rot_base is not implemented yet.")

    data = container.dims
    frame_names = container.frame_names

    n_rows = len(data)
    n_frames = len(frame_names)
    n_trajs = data[0].data.shape[0]
    n_states = data[0].mu.shape[1]

    fig = plt.figure(figsize=plt.figaspect(n_rows / n_frames))

    if size is None:
        size = (16, None)

    if size[1] is None and size[0] is not None:
        size = (size[0], size[0] / n_frames * n_rows)
    fig.set_size_inches(*size)

    alphas = [0.2 for _ in range(n_states)]  # np.linspace(0.4, 0.6, n_states)
    cmap_gauss = cm.get_cmap(cmap_gauss, n_states)(np.linspace(0, 1, n_states))
    cmap_trajs = cm.get_cmap("tab20" if n_trajs > 10 else "tab10")

    for i, row_data in enumerate(data):
        ax_is_3d = row_data.data.shape[-1] == 3
        ax_is_s2 = row_data.manifold is Manifold_S2
        projection = "3d" if ax_is_3d else None

        mu = row_data.mu
        sigma = row_data.sigma

        per_frame = row_data.per_frame

        n_time_steps = row_data.data.shape[1]
        time_dim = np.linspace(time_start, time_stop, n_time_steps)

        for j in range(n_frames if per_frame else 1):
            ax_idx = n_frames * i + j + 1
            ax = fig.add_subplot(n_rows, n_frames, ax_idx, projection=projection)

            ax.set_title(
                f"{frame_names[j]} frame: {row_data.name}"
                if per_frame
                else row_data.name
            )

            ax_data = row_data.data[:, :, j] if per_frame else row_data.data

            if len(ax_data.shape) == 2:  # single-dimensional data
                ax_data = ax_data[:, :, np.newaxis]

            # ax_zoom = 1.4 if ax_is_s2 else 1.0
            ax_zoom = 1.0

            if ax_is_s2:
                riepybdlib.s2_fcts.plot_manifold(ax)

                if s2_auto_rotate:
                    valid_data = ax_data[~np.isnan(ax_data).any(axis=2)]
                    ax_dim_mean = valid_data.mean(axis=1)
                    xy_angle = 90 - np.arctan2(ax_dim_mean[0], ax_dim_mean[1]) * 360 / (
                        2 * np.pi
                    )
                    z_angle = 45 - np.arccos(ax_dim_mean[2]) * 360 / (2 * np.pi)

                    if np.isnan(ax_dim_mean).any():
                        logger.warning("Could not auto-rotate S2 plot due to NaN data.")
                        z_angle = 45
                        xy_angle = 45

                    ax.view_init(elev=z_angle, azim=xy_angle, roll=0)

            if plot_traj:
                for c, traj in enumerate(ax_data):
                    if ax_is_3d:
                        if scatter:
                            scatter_3d_traj(ax, traj, s=3, color=cmap_trajs(c))
                            scatter_3d_point(ax, traj[0], s=10, marker="D", c="k")
                        else:
                            lineplot_3d_traj(ax, traj, c, cmap=cmap_trajs, linewidth=3)
                        # ax[0, j].set_aspect('equal', 'datalim')
                        if plot_traj and annotate_trajs:
                            ax.text(
                                traj[0, 0] * 1.1,
                                traj[0, 1],
                                traj[0, 2],
                                f"{c}",
                            )
                        ax.set_box_aspect(None, zoom=ax_zoom)
                    else:
                        assert not per_frame
                        ax.set_box_aspect(1)
                        ax.plot(time_dim, traj, color=cmap_trajs(c))

            if plot_gaussians:
                ax_mu = mu[j] if per_frame else mu
                for s in range(n_states):
                    state_mu = ax_mu[s]
                    if np.isnan(state_mu).any():
                        logger.warning(
                            # f"Skipping NaN Gaussians at row {i} col {j} dim {d} state {s}"
                            "Skippig NaN Gaussians."
                        )
                        continue
                    if gaussian_mean_only:
                        ax.scatter(
                            state_mu[0],
                            state_mu[1],
                            state_mu[2],
                            #  color=cmap[s], alpha=1, s=250)
                            color="k",
                            alpha=1,
                            s=1000,
                        )
                    else:
                        if ax_is_s2:
                            riepybdlib.s2_fcts.plot_gaussian(
                                ax,
                                state_mu,
                                sigma[j, s],
                                showtangent=s2_show_tangent,
                                color=cmap_gauss[s],
                                linealpha=alphas[s],
                            )
                        elif ax_is_3d:
                            plot_ellipsoid_from_gaussian3d(
                                ax,
                                state_mu,
                                sigma[j, s],
                                color=cmap_gauss[s],
                                alpha=alphas[s],
                            )
                        else:
                            plot_ellipsoid_from_gaussian2d(
                                ax,
                                state_mu,
                                sigma[s],
                                color=cmap_gauss[s],
                                alpha=alphas[s],
                            )
                    if plot_gaussians and annotate_gaussians and ax_is_3d:
                        ax.text(
                            mu[j, s, 0], mu[j, s, 1], mu[j, s, 2], str(s), fontsize=12
                        )
                    elif plot_gaussians and annotate_gaussians:
                        ax.text(mu[s, 0], mu[s, 1], str(s), fontsize=12)
                # ax.view_init(elev=120, azim=45, roll=15)

            if plot_coord_origin and ax_is_3d and not ax_is_s2:
                ax_min, ax_max = ax.get_xlim()
                ax_range = ax_max - ax_min
                plot_coordindate_frame(
                    ax=ax,
                    origin=np.zeros(3),
                    rotation=np.eye(3),
                    linewidth=2,
                    arrow_length=0.4 * ax_range,
                )

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()


def plot_gmm_time_based(
    container: TPGMMPlotData,
    plot_traj=True,
    plot_gaussians=True,
    size=(16, None),
    gaussian_mean_only=False,
    rot_on_tangent=False,
    rot_to_degrees=True,
    gaussian_scale=1.96,
    annotate_gaussians=False,
    annotate_trajs=False,
    title=None,
    rot_base=None,
    plot_derivatives=False,
    plot_traj_means=False,
    component_borders=None,
    time_start=0,
    time_stop=1,
):
    """
    Plot a set of trajectories in all given coordinate frames, overlayed with
    the GMM components. All plots per time.

    Parameters
    ----------
    data: TPGMMPlotData
        Bundled data for plotting. Contains the trajectories, the model's mean
        and covariance, and infos about the manifold.
    size: tuple[float/int]
        Size of the figure.
        If size[1] is None, the height is automatically determined based on the
        width and the number of trajectories.
    rot_on_tangent: bool
        Whether the rotation is on the tangent space (Riemann), else: Euler.
    """
    if plot_derivatives and plot_gaussians:
        logger.warning(
            "Plotting derivatives of data. Gaussians are of the original data."
        )

    data = container.dims
    frame_names = container.frame_names

    # NOTE: using 360 instead of 180 (the usual radians to degrees factor)
    # as in the tangent space, the rotations are only scaled by HALF the
    # rotation angle - I think to avoid the double covering.
    # logseq://graph/.logseqlib?block-id=6454c121-b29d-47a9-8f66-89a77ca71500
    angle_scale = 360 / np.pi

    n_rows = len(data)
    n_frames = len(frame_names)
    n_states = data[0].mu.shape[1]
    n_time_steps = data[0].data.shape[1]

    time_dim = np.linspace(time_start, time_stop, n_time_steps)

    fig = plt.figure(figsize=plt.figaspect(n_rows / n_frames))

    if size is None:
        size = (16, None)

    if size[1] is None and size[0] is not None:
        size = (size[0], size[0] / n_frames * n_rows)
    fig.set_size_inches(*size)

    alphas = [0.4 for _ in range(n_states)]  # np.linspace(0.4, 0.6, n_states)

    for i, row_data in enumerate(data):
        mu = row_data.mu
        sigma = row_data.sigma

        ax_is_s2 = row_data.manifold is Manifold_S2

        # Only scale rotation data if it contains a rotation magnitude.
        # Ie, it's in the quaternion tangent space or it is the rotation magnitude.
        ax_is_rot_with_mag = row_data.manifold in (Manifold_Quat, Manifold_S1)
        ax_scale_rot = rot_to_degrees and ax_is_rot_with_mag

        per_frame = row_data.per_frame

        for j in range(n_frames if row_data.per_frame else 1):
            ax_idx = i * n_frames + j + 1
            ax = fig.add_subplot(n_rows, n_frames, ax_idx)

            ax.set_title(
                f"{frame_names[j]} frame: {row_data.name}"
                if per_frame
                else row_data.name
            )

            # TODO: update this part
            # axis_min = plot_order[i][:, :, j].min() * 1.05
            # axis_max = plot_order[i][:, :, j].max() * 1.05

            # default_min = -180 if ax_scale_rot else -1
            # default_max = 180 if ax_scale_rot else -1
            # axis_min = default_min if np.isnan(axis_min) else axis_min
            # axis_max = default_max if np.isnan(axis_max) else axis_max

            # ax[i][j].set_xlim(0, 1)
            # ax[i][j].set_ylim(axis_min, axis_max)

            ax_data = row_data.data[:, :, j] if per_frame else row_data.data

            if len(ax_data.shape) == 2:  # single-dimensional data
                ax_data = ax_data[:, :, np.newaxis]

            if ax_scale_rot:
                ax_data = ax_data * angle_scale
                scale_D = np.diag((1, angle_scale))
            else:
                scale_D = np.eye(2)

            if plot_derivatives:
                ax_data = np.gradient(ax_data, 1 / n_time_steps, axis=1)

            ax_mean = np.mean(ax_data, axis=0)

            n_dims = ax_data.shape[-1]

            gauss_labels = row_data.gauss_labels
            if gauss_labels is not None:
                cmap_gauss = cm.get_cmap("tab20" if max(gauss_labels) > 10 else "tab10")

            if plot_traj:
                for t, traj in enumerate(ax_data):
                    for d in range(n_dims):
                        dim_traj = traj[:, d]
                        # HACK
                        # time_dim = np.arange(0, 1, 1 / (dim_traj.shape[0] - 1))
                        ax.plot(
                            time_dim,
                            dim_traj,
                            c=dim_colors[d] if n_dims > 1 else None,
                            alpha=0.15 if plot_traj_means else 0.3,
                        )
                        if annotate_trajs:
                            n_steps = traj.shape[0]
                            x_pos = 0.7
                            ax.annotate(f"{t}", (x_pos, dim_traj[int(n_steps * x_pos)]))

            if plot_traj_means:
                for d in range(n_dims):
                    ax.plot(
                        time_dim, ax_mean[:, d], c=dim_colors[d], alpha=1, linewidth=2
                    )

            if plot_gaussians:
                ax_mu = mu[j] if per_frame else mu
                ax_sigma = sigma[j] if per_frame else sigma

                for d in range(1, ax_mu.shape[-1]):
                    dim_mu = ax_mu[..., [0, d]]
                    if ax_is_s2:
                        logger.warning(
                            "Can't plot S2 Gaussian Cov with Mani data in time-based plot."
                        )
                        dim_sigma = None
                    else:
                        dim_sigma = ax_sigma[..., [0, d]][..., [0, d], :]

                    for s in range(n_states):
                        state_mu = dim_mu[s]
                        if np.isnan(state_mu).any():
                            logger.warning(
                                # f"Skipping NaN Gaussians at row {i} col {j} dim {d} state {s}"
                                "Skippig NaN Gaussians."
                            )
                            continue
                        edge_color = (
                            "k" if gauss_labels is None else cmap_gauss(gauss_labels[s])
                        )
                        if ax_scale_rot and rot_base is not None:
                            state_mu[1] = state_mu[1] - rot_base[j][i // 2][d]
                        scaled_mu = scale_D @ state_mu
                        if gaussian_mean_only or ax_is_s2:
                            ax.scatter(
                                scaled_mu[0],
                                scaled_mu[1],
                                edgecolors=edge_color,
                                c=dim_colors[d - 1],
                                alpha=0.7,
                                s=1000,
                            )
                        else:
                            scaled_sigma = scale_D.T @ dim_sigma[s] @ scale_D
                            plot_ellipsoid_from_gaussian2d(
                                ax,
                                scaled_mu,
                                scaled_sigma,
                                alpha=alphas[s],
                                facecolor=dim_colors[d - 1],
                                edgecolor=edge_color,
                                scale=gaussian_scale,
                            )

                        if annotate_gaussians:
                            ax.annotate(f"{s}", scaled_mu)

            if component_borders is not None:
                for x in component_borders:
                    ax.axvline(x=x, linestyle=":", color="k", alpha=0.5)

    patches = [(Line2D([0], [0], color=c, lw=4)) for c in dim_colors]
    labels = ["x", "y", "z"]

    fig.axes[0].legend(patches, labels)

    if title is not None:
        title = title + (" (1st derivative)" if plot_derivatives else "")
        plt.suptitle(title)

    plt.tight_layout()

    plt.show()


def plot_gmm_xdx_based(
    pos_per_frame,
    model,
    frame_names,
    rot_per_frame=None,
    pos_delta_per_frame=None,
    rot_delta_per_frame=None,
    plot_traj=True,
    plot_gaussians=True,
    size=(16, 32),
    gaussian_mean_only=False,
    cmap="Oranges",
    plot_coord_origin=True,
    scatter=False,
    rot_on_tangent=False,
    annotate_gaussians=False,
    annotate_trajs=False,
    title=None,
    rot_base=None,
    model_includes_time=False,
):
    """
    Plot GMMS by mapping X to DX.

    Parameters
    ----------
    pos_per_frame: torch.Tensor
        Position trajectories in all coordinate frames.
        Shape: (n_traj, n_obs, n_frames, 3)
    model: GMM
        GMM model (or any model with mu and sigma attributes).
        Assumes that the model has 6 components per coordinate frame (x, y, z,
        vx, vy, vz) and then stacks these frame-wise submodels.
    frame_names: list[str]
        Names of the coordinate frames.
    rot_per_frame: torch.Tensor
        Rotation trajectories in all coordinate frames as Euler angles.
        Shape: (n_traj, n_obs, n_frames, 3)
    pos_delta_per_frame: torch.Tensor
        Position delta trajectories in all coordinate frames.
        Shape: (n_traj, n_obs, n_frames, 3)
    rot_delta_per_frame: torch.Tensor
        Rotation delta trajectories in all coordinate frames as Euler angles.
        Shape: (n_traj, n_obs, n_frames, 3)
    size: tuple[float/int]
        Size of the figure.
        If size[1] is None, the height is automatically determined based on the
        width and the number of trajectories.
    cmap: str
        Name of the colormap to use for the GMM components.
    plot_coord_origin: bool
        Whether to plot the coordinate origin in each frame.
    scatter: bool
        Whether to plot the trajectories as scatter plot or as line plot.
        Scatter can help when values are very close to each other - line plots
        can fall apart in that case.
    rot_on_tangent: bool
        Whether the rotation is on the tangent space (Riemann), else: Euler.
    """
    if rot_base is not None:
        logger.warning("rot_base is not implemented yet.")

    n_rows = 1 if rot_per_frame is None else 2
    assert pos_delta_per_frame is not None

    n_frames = len(frame_names)

    fig, ax = plt.subplots(
        ncols=n_frames,
        nrows=n_rows * 3,
    )
    if n_rows == 1:
        ax = ax[np.newaxis, :]

    if n_frames == 1:
        ax = ax[:, np.newaxis]

    if size[1] is None and size[0] is not None:
        size = (size[0], size[0] / n_frames * n_rows)
    fig.set_size_inches(*size)

    n_states = model.nb_states

    alphas = [0.4 for _ in range(n_states)]  # np.linspace(0.4, 0.6, n_states)
    cmap = cm.get_cmap(cmap, n_states)(np.linspace(0, 1, n_states))

    plot_order = [
        [pos_per_frame, pos_delta_per_frame],
        [rot_per_frame, rot_delta_per_frame],
    ]

    plot_titles = ["pos", "rot"]

    if rot_on_tangent:
        plot_titles[1] = plot_titles[1] + " (tangent)"
    else:
        plot_titles[1] = plot_titles[1] + " (Euler)"

    plot_order = [p for p in plot_order if p[0] is not None]

    dims = tuple(("x", "y", "z"))

    for j in range(n_frames):
        # position plotting
        for r, t in enumerate(plot_titles):
            x_traj = plot_order[r][0]
            dx_traj = plot_order[r][1]
            for d in range(3):
                i = r * 3 + d
                ax[i][j].set_title(f"{frame_names[j]} frame: {t}")

                # x_min = x_traj[:, :, j, d].min()
                # x_max = x_traj[:, :, j, d].max()
                # dx_min = dx_traj[:, :, j, d].min()
                # dx_max = dx_traj[:, :, j, d].max()

                ax[i][j].set_xlabel(dims[d])
                ax[i][j].set_ylabel(f"d{dims[d]}")

                if plot_traj:
                    for c in range(x_traj.shape[0]):
                        ax[i][j].plot(x_traj[c, :, j, d], dx_traj[c, :, j, d])
                        ax[i][j].scatter(
                            x_traj[c, 0, j, d],
                            dx_traj[c, 0, j, d],
                            c="k",
                            marker="D",
                            s=10,
                        )
                if plot_gaussians:
                    k = j * n_rows * 6 + r * 3 + d + (1 if model_includes_time else 0)
                    # print(j, m, r, k, k+ 3 * n_rows)
                    # print(model.mu.shape)
                    if rot_on_tangent:
                        mu, sigma = model.get_mu_sigma(
                            idx=[k, k + 3 * n_rows], stack=True, as_np=True
                        )
                        # print(mu.shape)
                    else:
                        raise NotImplementedError

                    for s in range(n_states):
                        if gaussian_mean_only:
                            ax[i][j].scatter(
                                mu[s, 0],
                                mu[s, 1],
                                #  color=cmap[s], alpha=1, s=250)
                                color="k",
                                alpha=1,
                                s=1000,
                            )
                        else:
                            plot_ellipsoid_from_gaussian2d(
                                ax[i][j],
                                mu[s],
                                sigma[s],
                                color=cmap[s],
                                alpha=alphas[s],
                            )
                        if annotate_gaussians:
                            ax[i][j].text(
                                mu[s, 0], mu[s, 1], mu[s, 2], str(s), fontsize=12
                            )

    if title is not None:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()


def plot_gmm_frames_time_based(
    model,
    joint_models,
    models_transformed,
    frame_names,
    plot_trajectory=0,
    size=(21, None),
    equal_aspect=False,
    plot_joint=True,
    plot_coord_origin=False,
    axis_range=(-0.7, 0.7),
    rotation_range=(-1.7, 1.7),
    joint_plot_per_component=True,
    includes_rotations=True,
    gaussian_scale=1.96,
):
    border_width = 1.5

    n_frames = len(frame_names)

    n_states = model.nb_states

    m_joint = joint_models[plot_trajectory]
    if type(m_joint) in (list, tuple):
        m_joint = m_joint[0]

    dim = 3
    mani_dim = m_joint.mu.shape[1] - 1
    assert mani_dim % dim == 0
    n_mani = mani_dim // dim
    n_cols = n_frames + (3 if joint_plot_per_component else 1)

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_mani)

    if n_mani == 1:
        ax = [ax]

    plot_titles = list(def_plot_titles)
    plot_titles = plot_titles[:: n_cols // (4 if includes_rotations else 2)]

    width, height = size
    if height is None:
        size = (width, width * n_mani / (n_cols))
    fig.set_size_inches(*size)

    alpha_marg = 0.4
    alpha_joint = 0.9

    joint_colors = ["tab:red", "tab:purple", "tab:brown"]

    angle_scale = 360 / np.pi
    D = np.diag((1, angle_scale if includes_rotations else 1))

    with_action_dim = False

    frame_dim = dim * (2 if with_action_dim else 1) * (2 if includes_rotations else 1)

    global_min = np.inf
    global_max = -np.inf

    for i in range(n_mani):
        # add plot_titles[i] to the left of ax[i][0]
        ax[i][0].set_ylabel(
            plot_titles[i], rotation=0, size="large", ha="right", va="center"
        )
        for j, frame_name in enumerate(frame_names):
            ax_min = np.inf
            ax_max = -np.inf
            if i == 0:
                ax[i][j].set_title(frame_name)

            set_ax_border(ax[i][j], frame_colors[j], border_width)
            for s in range(n_states):
                for d in range(3):
                    idx = i * 3 + j * frame_dim + 1 + d
                    mu, sigma = model.get_mu_sigma(idx=[0, idx], stack=True, as_np=True)
                    if i % 2 == 1 and includes_rotations:
                        scaled_mu = D @ mu[s]
                        scaled_sigma = D.T @ sigma[s] @ D
                    else:
                        scaled_mu = mu[s]
                        scaled_sigma = sigma[s]
                    plot_ellipsoid_from_gaussian2d(
                        ax[i][j],
                        scaled_mu,
                        scaled_sigma,
                        alpha=alpha_marg,
                        facecolor=dim_colors[d],
                        edgecolor="k",
                        scale=gaussian_scale,
                    )

                    ax_min = min(ax_min, mu[s, 1])
                    ax_max = max(ax_max, mu[s, 1])

                mt_i = models_transformed[plot_trajectory]
                if type(mt_i[0]) in (list, tuple):
                    logger.warning(
                        "Model uses moving frames. Can only plot one at a time. "
                        "Plotting the one from the first time step."
                    )
                    m_trans = mt_i[0][j]
                else:
                    m_trans = mt_i[j]

                for d in range(3):
                    jdx = i * 3 + 1 + d
                    if joint_plot_per_component:
                        k = -1 * (3 - d)
                        c = frame_colors[j]
                    else:
                        k = -1
                        c = dim_colors[d]

                    mu, sigma = m_trans.get_mu_sigma(
                        idx=[0, jdx], stack=True, as_np=True
                    )
                    if i % 2 == 1 and includes_rotations:
                        scaled_mu = D @ mu[s]
                        scaled_sigma = D.T @ sigma[s] @ D
                    else:
                        scaled_mu = mu[s]
                        scaled_sigma = sigma[s]
                    plot_ellipsoid_from_gaussian2d(
                        ax[i][k],
                        scaled_mu,
                        scaled_sigma,
                        alpha=alpha_marg,
                        facecolor=c,
                        edgecolor="k",
                        scale=gaussian_scale,
                    )

            ax[i][j].set_xlim(0, 1)
            if i % 2 == 1 and includes_rotations:
                ax_range_scaled = tuple(a * angle_scale for a in rotation_range)
            else:
                ax_range_scaled = axis_range
            if axis_range is None:
                ax[i][j].set_ylim(ax_min * 1.05, ax_max * 1.05)
            else:
                ax[i][j].set_ylim(*ax_range_scaled)

            if equal_aspect:
                ax[i][j].set_aspect("equal")

            global_min = min(global_min, ax_min)
            global_max = max(global_max, ax_max)

        if plot_joint:
            for s in range(n_states):
                for d in range(3):
                    jdx = i * 3 + 1 + d
                    j = -1 * (3 - d) if joint_plot_per_component else -1
                    c = joint_color if joint_plot_per_component else joint_colors[d]
                    mu, sigma = m_joint.get_mu_sigma(
                        idx=[0, jdx], stack=True, as_np=True
                    )
                    if i % 2 == 1 and includes_rotations:
                        scaled_mu = D @ mu[s]
                        scaled_sigma = D.T @ sigma[s] @ D
                    else:
                        scaled_mu = mu[s]
                        scaled_sigma = sigma[s]
                    plot_ellipsoid_from_gaussian2d(
                        ax[i][j],
                        scaled_mu,
                        scaled_sigma,
                        alpha=alpha_joint,
                        facecolor=c,
                        edgecolor="k",
                        scale=gaussian_scale,
                    )
                    # ax[i][-1].text(mu[s][0], mu[s][1],
                    #             '%s' % (str(s)), size=10, zorder=20, color='k')

        for j in range(n_cols - n_frames):
            if equal_aspect:
                ax[i][-1 * j].set_aspect("equal")
            joint_title = "joint model (projected)"
            if joint_plot_per_component:
                j_title = joint_title + f" {dim_names[j]}"
            else:
                j_title = joint_title
            k = -1 * (3 - j) if joint_plot_per_component else -1

            set_ax_border(ax[i][k], dim_colors[j], border_width)

            if i == 0:
                ax[i][k].set_title(j_title)

            ax[i][k].set_xlim(0, 1)
            if i % 2 == 1 and includes_rotations:
                ax_range_scaled = tuple(a * angle_scale for a in rotation_range)
            else:
                ax_range_scaled = axis_range
            if axis_range is None:
                ax[i][k].set_ylim(global_min, global_max)
            else:
                ax[i][k].set_ylim(*ax_range_scaled)

        if plot_coord_origin:
            raise NotImplementedError
            for j in range(n_frames + 1):
                plot_coordindate_frame(
                    ax=ax[i][j],
                    origin=np.zeros(3),
                    rotation=np.eye(3),
                    arrow_length=0.1,
                    linewidth=1,
                )

                if equal_aspect:
                    ax[i][j].set_aspect("equal")

    if joint_plot_per_component:
        patches = [(Line2D([0], [0], color=c, lw=4)) for c in dim_colors]
        labels = ["x", "y", "z"]

        patches += [
            (Line2D([0], [0], color=c, lw=4)) for c, f in zip(frame_colors, frame_names)
        ] + [(Line2D([0], [0], color=joint_color, lw=4))]
        labels += list(frame_names) + ["joint"]
        fig.legend(patches, labels, loc="outside lower center", ncol=len(patches))

    else:
        patches = [(Line2D([0], [0], color=c, lw=4)) for c in dim_colors] + [
            (Line2D([0], [0], color=c, lw=4)) for c in joint_colors
        ]
        labels = ["x", "y", "z", "xj", "yj", "zj"]

        ax[0][-1].legend(patches, labels)

    xpos = (
        0.85 * ax[0][n_frames - 1].get_position().x1
        + 0.15 * ax[0][n_frames].get_position().x0
    )
    ypos0 = ax[-1][0].get_position().y0
    ypos1 = ax[0][0].get_position().y1
    line = plt.Line2D((xpos, xpos), (ypos0, ypos1), color="black", linewidth=2)
    fig.add_artist(line)


def plot_gmm_frames(
    model,
    joint_models,
    models_transformed,
    frame_names,
    plot_trajectory=0,
    size=(12, None),
    equal_aspect=False,
    plot_joint=False,
    plot_coord_origin=False,
):
    """
    Plot the GMM components in all given coordinate frames and the joint model
    in world frame.

    Parameters
    ----------
    model: GMM
        The base model.
    joint_models: list[GMM] or list[list[GMM]]
        The joint model per trajectory and state. Optionaly per obersevation.
        Shape: (n_traj) or (n_traj, n_obs).
    models_transformed: list[list[list[GMM]]] or
                        list[list[list[list[GMM]]]]
        The local model per trajectory, state and frame. Optionaly per obs.
        Shape: (n_traj, n_states, (n_observations), n_frames).
    frame_names: list[str]
        Names of the coordinate frames.
    plot_trajectory: int
        We work with potentially moving frames as they encode the location of
        potentially moving objects. Hence, the frame2world transforms used for
        the local models are trajectory-dependent. For plotting, just pick one
        trajectory. The joint model will look slightly different depending on
        the trajectory.
    size: tuple[float/int]
        Size of the figure.
        If size[1] is None, the height is automatically determined based on the
        the number of frames and sub-manifolds.
    """

    n_frames = len(frame_names)

    n_states = model.nb_states

    m_joint = joint_models[plot_trajectory]
    if type(m_joint) in (list, tuple):
        m_joint = m_joint[0]

    dim = 3
    mani_dim = m_joint.mu.shape[1]
    assert mani_dim % dim == 0
    n_mani = mani_dim // dim

    fig, ax = plt.subplots(
        ncols=n_frames + 1, nrows=n_mani, subplot_kw={"projection": "3d"}
    )

    width, height = size
    if height is None:
        size = (width, width * n_mani / (n_frames + 1))
    fig.set_size_inches(*size)

    alphas = [0.3 for _ in range(n_states)]

    for i in range(n_mani):
        for j, frame_name in enumerate(frame_names):
            if i == 0:
                ax[i][j].set_title(frame_name)
            for s in range(n_states):
                rgba = matplotlib.colors.to_rgba(tab_colors[j])
                idx = i * 3 + j * 12
                plot_ellipsoid_from_gaussian3d(
                    ax[i][j],
                    model.mu[s, idx : idx + 3],
                    model.sigma[s, idx : idx + 3, idx : idx + 3],
                    color=rgba,
                    alpha=alphas[s],
                )

                mt_i = models_transformed[plot_trajectory]
                if type(mt_i[0]) in (list, tuple):
                    logger.warning(
                        "Model uses moving frames. Can only plot one at a time. "
                        "Plotting the one from the first time step."
                    )
                    m_trans = mt_i[0][j]
                else:
                    m_trans = mt_i[j]

                jdx = i * 3

                plot_ellipsoid_from_gaussian3d(
                    ax[i][-1],
                    m_trans.mu[s, jdx : jdx + 3],
                    m_trans.sigma[s, jdx : jdx + 3, jdx : jdx + 3],
                    color=rgba,
                    alpha=0.1,
                )

            if equal_aspect:
                ax[i][j].set_aspect("equal")

        if plot_joint:
            for s in range(n_states):
                plot_ellipsoid_from_gaussian3d(
                    ax[i][-1],
                    m_joint.mu[s, jdx : jdx + 3],
                    m_joint.sigma[s, jdx : jdx + 3, jdx : jdx + 3],
                    color=joint_color,
                    alpha=0.8,
                )
                ax[i][-1].text(
                    m_joint.mu[s, jdx],
                    m_joint.mu[s, jdx + 1],
                    m_joint.mu[s, jdx + 2],
                    "%s" % (str(s)),
                    size=10,
                    zorder=20,
                    color="k",
                )

        if equal_aspect:
            ax[i][-1].set_aspect("equal")
        ax[i][-1].set_title("joint model (projected)")

        if plot_coord_origin:
            for j in range(n_frames + 1):
                plot_coordindate_frame(
                    ax=ax[i][j],
                    origin=np.zeros(3),
                    rotation=np.eye(3),
                    arrow_length=0.1,
                    linewidth=1,
                )

                if equal_aspect:
                    ax[i][j].set_aspect("equal")


def plot_gmm_components(
    model,
    joint_models,
    models_transformed,
    frame_names,
    plot_trajectory=0,
    size=(None, 6),
):
    """
    Plot the transported marginals and joint model in world frame - per
    component.

    Parameters
    ----------
    model: GMM
        The base model.
    joint_models: list[GMM] or list[list[GMM]]
        The joint model per trajectory and state. Optionaly per obersevation.
        Shape: (n_traj) or (n_traj, n_obs).
    models_transformed: list[list[list[GMM]]] or
                        list[list[list[list[GMM]]]]
        The local model per trajectory, state and frame. Optionaly per obs.
        Shape: (n_traj, n_states, (n_observations), n_frames).
    frame_names: list[str]
        Names of the coordinate frames.
    plot_trajectory: int
        We work with potentially moving frames as they encode the location of
        potentially moving objects. Hence, the frame2world transforms used for
        the local models are trajectory-dependent. For plotting, just pick one
        trajectory. The joint model will look slightly different depending on
        the trajectory.
    size: tuple[float/int]
        Size of the figure.
        If size[0] is None, the width is automatically determined based on the
        height and the number of frames.
    """

    n_frames = len(frame_names)

    n_states = model.nb_states

    fig, ax = plt.subplots(ncols=n_states, subplot_kw={"projection": "3d"})

    if size[0] is None and size[1] is not None:
        size = (size[1] * n_states, size[1])
    fig.set_size_inches(*size)

    patches = []

    mt_i = models_transformed[plot_trajectory]

    # Dbg stuff
    # margs = [m.margin([0, 2]) for m in mt_i[0]]
    # prod = margs[0] * margs[1] * margs[2] * margs[3]

    # plot_ellipsoid_from_gaussian3d(
    #     ax[1], prod.mu[1, :3], prod.sigma[1, :3, :3], color='k', alpha=0.6)

    for s in range(n_states):
        ax[s].set_title(f"component {s}")
        for f, n in enumerate(frame_names):
            rgba = matplotlib.colors.to_rgba(tab_colors[f])

            if s == 0:
                patches.append(Line2D([0], [0], color=rgba, lw=4))

            if type(mt_i[0]) in (list, tuple):
                logger.warning(
                    "Model uses moving frames. Can only plot one at a time. "
                    "Plotting the one from the first time step."
                )
                m_trans = mt_i[0][f]
            else:
                m_trans = mt_i[f]

            plot_ellipsoid_from_gaussian3d(
                ax[s],
                m_trans.mu[s, :3],
                m_trans.sigma[s, :3, :3],
                color=rgba,
                alpha=0.2,
            )

        m_joint = joint_models[plot_trajectory]
        if type(m_joint) in (list, tuple):
            m_joint = m_joint[0]

        plot_ellipsoid_from_gaussian3d(
            ax[s],
            m_joint.mu[s, :3],
            m_joint.sigma[s, :3, :3],
            color=joint_color,
            alpha=0.8,
        )

        if s == 0:
            patches.append(Line2D([0], [0], color=joint_color, lw=4))

        plot_coordindate_frame(
            ax=ax[s], origin=np.zeros(3), rotation=np.eye(3), arrow_length=0.1
        )

        ax[s].set_aspect("equal")

    plt.legend(
        patches,
        frame_names + ("joint model",),
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        borderaxespad=0,
    )


def plot_reconstructions_3d(
    marginals: list[list[GMM]] | list[list[list[GMM]]] | None,
    joint_models: list[GMM],
    reconstructions: list[np.ndarray],
    original_trajectories: list[np.ndarray],
    coord_frames: torch.Tensor | None = None,
    frame_names: tuple[str, ...] | None = None,
    size=(18, None),
    plot_gaussians=True,
    plot_coord_origin=True,
    plot_trajectories=True,
    plot_reconstructions=True,
    equal_aspect=True,
    includes_time=False,
    includes_rotations=False,
    rec_color="tab:orange",
):
    """
    Plot the reconstructions in world frame, including the projected local
    models and the joint model and the orginal trajectory.

    Parameters
    ----------
    marginals: list[list[GMM]] or list[lis[list[GMM]]]
        The local models per trajectory. Optionally, per observation/timestep.
        Shape: (n_traj, n_frames) or (n_traj, n_observations, n_frames).
    joint_models: list[GMM] or list[list[GMM]]
        The joint model per trajectory.
        For moving frames, each timestep of each trajectory has its own model.
        Shape: (n_traj) or (n_traj, n_observations).
    reconstructions: torch.Tensor or list[torch.Tensor]
        The reconstructions per trajectory.
        n_dims needs to be at least 3 (x, y, z), but can be larger.
        If dim = 6, assumes the second 3 dimensions are the actions,
        if dim = 12, assumes the extra dims are rotation and actions (pos and
        rotation) in order.
        Shape: (n_traj, n_frames, n_dims).
    original_trajectories: torch.Tensor or list[torch.Tensor]
        The original coordinates per trajectory.
        Shape and dims are identical to reconstructions.
    coord_frames: torch.Tensor
        The coordinate origins and rotations per trajectory and frame.
        Given in homogeneous coordinates. Contains an empty dimension for
        time/observations.
        Shape: (n_traj, n_frames, 1, 4, 4).
    frame_names: list[str]
        Names of the coordinate frames. Used for annotation.
    n_cols: int
        Number of columns in the plot.
    size: tuple[float/int]
        Size of the figure.
        If size[1] is None, the height is automatically determined based on the
        width and the number of trajectories.
    plot_gaussians: bool
        Whether to plot the local and joint models.
    plot_coord_origin: bool
        Whether to plot the coordinate origins in the pose plots.
    plot_trajectories: bool
        Whether to plot the original trajectories.
    plot_reconstructions: bool
        Whether to plot the reconstructions.
    equal_aspect: bool
        Whether to force equal aspect ratio for all plots.
    """
    n_demos = len(joint_models)
    n_states = (
        joint_models[0][0].nb_states
        if type(joint_models[0]) is tuple
        else joint_models[0].nb_states
    )

    gmm_dim = (
        joint_models[0][0].mu.shape[1]
        if type(joint_models[0]) is tuple
        else joint_models[0].mu.shape[1]
    )

    if includes_time:
        gmm_dim -= 1

    n_cols = gmm_dim // 3

    plot_titles = list(def_plot_titles)
    plot_titles = plot_titles[:: 1 if includes_rotations else 2]

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_demos, subplot_kw={"projection": "3d"})
    if n_cols == 1:
        ax = ax[:, None]
    if size[1] is None and size[0] is not None:
        size = (size[0], size[0] / 4 * n_demos)
    fig.set_size_inches(*size)

    for i, t in zip(range(n_cols), plot_titles):
        ax[0][i].set_title(t)

    for i in range(n_demos):
        if marginals is None:
            marg = tuple()
        elif type(marginals[i][0]) in (list, tuple):
            logger.warning(
                "Model uses moving frames. Can only plot one at a time. "
                "Plotting the one from the first time step."
            )
            # TODO: animate plot by moving the EE pose and plotting the synced
            # marginals and joint model
            marg = marginals[i][0]
        else:
            marg = marginals[i]

        if type(joint_models[i]) in (list, tuple):
            joint = joint_models[i][0]
        else:
            joint = joint_models[i]

        for j in range(n_cols):
            ax[i][j].set_xlabel("x")
            ax[i][j].set_ylabel("y")
            ax[i][j].set_zlabel("z")

            sl = slice(
                j * 3 + (1 if includes_time else 0),
                (j + 1) * 3 + (1 if includes_time else 0),
            )
            if plot_gaussians:
                for f, frame_model in enumerate(marg):
                    rgba = matplotlib.colors.to_rgba(tab_colors[f])
                    for s in range(frame_model.nb_states):
                        plot_ellipsoid_from_gaussian3d(
                            ax[i][j],
                            frame_model.mu[s, sl],
                            frame_model.sigma[s, sl, sl],
                            color=rgba,
                            alpha=0.05,
                        )

                for s in range(n_states):
                    plot_ellipsoid_from_gaussian3d(
                        ax[i][j],
                        joint.mu[s, sl],
                        joint.sigma[s, sl, sl],
                        color=joint_color,
                        alpha=0.2,
                    )

            rec = reconstructions[i][:, sl]
            traj = original_trajectories[i][:, sl]
            axi = ax[i][j]
            if plot_reconstructions:
                axi.plot(
                    rec[:, 0], rec[:, 1], rec[:, 2], color=rec_color, lw=2, zorder=10
                )
            if plot_trajectories:
                axi.scatter(
                    traj[:, 0][0],
                    traj[:, 1][0],
                    traj[:, 2][0],
                    marker="s",
                    color="k",
                    s=40,
                )
                axi.scatter(
                    traj[:, 0][-1],
                    traj[:, 1][-1],
                    traj[:, 2][-1],
                    marker=".",
                    color="k",
                    s=100,
                )
                axi.plot(traj[:, 0], traj[:, 1], traj[:, 2], "k--", lw=2, zorder=11)

        if plot_coord_origin:
            if coord_frames is not None:
                for f, n in enumerate(frame_names):
                    plot_coordindate_frame(
                        ax=ax[i][0],
                        transformation=coord_frames[i][f][0].double(),
                        arrow_length=0.1,
                        linewidth=1,
                        annotation=n,
                        annotation_size=10,
                    )

        if equal_aspect:
            for j in range(n_cols):
                ax[i][j].set_aspect("equal")

    handles = [
        Line2D([0], [0], color="k", lw=2, ls="--", label="original"),
        Line2D([0], [0], color=rec_color, lw=2, label="reconstruction"),
    ]

    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize="large")

    plt.tight_layout()


def plot_reconstructions_time_based(
    marginals: list[list[GMM]] | None,
    joint_models: list[GMM],
    reconstructions: list[np.ndarray],
    original_trajectories: list[np.ndarray],
    coord_frames: torch.Tensor,
    frame_names: tuple[str, ...],
    size=(18, None),
    plot_gaussians=True,
    plot_trajectories=True,
    plot_reconstructions=True,
    equal_aspect=False,
    plot_coord_origin=True,
    includes_rotations=True,
    includes_time=True,
    includes_actions=False,
    includes_action_magnitudes=False,
    includes_gripper_actions=False,
    component_alpha=0.2,
    joint_alpha=0.4,
    gaussian_scale=1.96,
    component_borders: tuple[tuple[float, ...], ...] | None = None,
    only_plot_dims: tuple[int, ...] | None = None,
    save_svg: str | None = None,
):
    if not includes_time and plot_gaussians:
        logger.warning("Cannot plot time-based gaussians without time dim.")
        plot_gaussians = False

    n_demos = len(joint_models)
    n_states = (
        0
        if not plot_gaussians
        else (
            joint_models[0][0].nb_states
            if type(joint_models[0]) is tuple
            else joint_models[0].nb_states
        )
    )

    # gmm_dim = (
    #     joint_models[0][0].mu.shape[1]
    #     if type(joint_models[0]) is tuple
    #     else joint_models[0].mu.shape[1]
    # )

    submanis = joint_models[0].manifold.get_submanifolds()

    mani_names = [s.name for s in submanis]
    mani_dims = [s.n_dimT for s in submanis]

    if only_plot_dims is not None:
        mani_dims = [mani_dims[i] for i in only_plot_dims]
        mani_names = [mani_names[i] for i in only_plot_dims]

    t_off = int(includes_time)

    n_cols = len(mani_dims) - int(includes_time)

    plot_titles = list(def_plot_titles)
    plot_titles = plot_titles[:: 1 if includes_rotations else 2]

    if includes_action_magnitudes:
        plot_titles.append("pos delta magnitude")

        if includes_rotations:
            plot_titles.append("rot delta magnitude")

    if includes_gripper_actions:
        plot_titles.append("gripper")

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_demos)

    angle_scale = 360 / np.pi
    D = np.diag((1, angle_scale if includes_rotations else 1))

    if n_cols == 1:
        ax = ax[:, None]
    if n_demos == 1:
        ax = ax[None, :]

    if size is not None:
        if size[1] is None and size[0] is not None:
            size = (size[0], size[0] / n_cols * n_demos)
        fig.set_size_inches(*size)

    for i, t in zip(range(n_cols), plot_titles):
        ax[0][i].set_title(t)

    for i in range(n_demos):
        if marginals is None:
            marg = tuple()
        elif type(marginals[i][0]) in (list, tuple):
            logger.warning(
                "Model uses moving frames. Can only plot one at a time. "
                "Plotting the one from the first time step."
            )
            # TODO: animate plot by moving the EE pose and plotting the synced
            # marginals and joint model
            marg = marginals[i][0]
        else:
            marg = marginals[i]

        if type(joint_models[i]) in (list, tuple):
            joint = joint_models[i][0]
        else:
            joint = joint_models[i]

        start = t_off

        for j in range(n_cols):
            ax[i][j].set_xlim(0, 1)
            ax[i][j].set_xlabel("time")

            dim = mani_dims[j + t_off]

            stop = start + dim

            sl = slice(start, stop)

            is_quat = mani_names[j + t_off] == "QUAT"

            if plot_coord_origin and dim == 3 and j <= int(includes_rotations):
                for d in range(3):
                    rgba = matplotlib.colors.to_rgba(dim_colors[d])
                    for f, frame in enumerate(coord_frames[i, :, 3 * (j % 2) + d]):
                        if is_quat:
                            frame *= angle_scale
                        ax[i][j].axhline(y=frame, color=rgba, ls="dotted", alpha=0.6)
                        ax[i][j].annotate(
                            frame_names[f],
                            xy=(0, frame),
                            color=rgba,
                            alpha=0.6,
                            ha="right",
                            va="bottom",
                        )

            if plot_gaussians:
                for d in range(dim):
                    for s in range(n_states):
                        rgba = matplotlib.colors.to_rgba(dim_colors[d])
                        idx = start + d
                        mu, sigma = joint.get_mu_sigma(
                            idx=[0, idx], stack=True, as_np=True
                        )
                        if is_quat:
                            scaled_mu = D @ mu[s]
                            scaled_sigma = D.T @ sigma[s] @ D
                        else:
                            scaled_mu = mu[s]
                            scaled_sigma = sigma[s]
                        plot_ellipsoid_from_gaussian2d(
                            ax[i][j],
                            scaled_mu,
                            scaled_sigma,
                            alpha=joint_alpha,
                            facecolor=rgba,
                            edgecolor="k",
                            scale=gaussian_scale,
                            lw=1.5,
                        )

                    for f, frame_model in enumerate(marg):
                        if frame_model is None:
                            continue
                        rgba = matplotlib.colors.to_rgba(dim_colors[d])
                        frame_color = frame_colors[f]
                        for s in range(frame_model.nb_states):
                            mu, sigma = frame_model.get_mu_sigma(
                                idx=[0, idx], stack=True, as_np=True
                            )
                            if is_quat:
                                scaled_mu = D @ mu[s]
                                scaled_sigma = D.T @ sigma[s] @ D
                            else:
                                scaled_mu = mu[s]
                                scaled_sigma = sigma[s]
                            plot_ellipsoid_from_gaussian2d(
                                ax[i][j],
                                scaled_mu,
                                scaled_sigma,
                                alpha=component_alpha,
                                facecolor=rgba,
                                edgecolor=frame_color,
                                scale=gaussian_scale,
                                ls="--",
                                lw=3,
                            )

            rec = reconstructions[i][:, sl]
            traj = original_trajectories[i][:, sl] if plot_trajectories else None
            axi = ax[i][j]
            # axi.set_xlim([0, 1])
            if is_quat:
                rec = angle_scale * rec if plot_reconstructions else None
                traj = angle_scale * traj if plot_trajectories else None
            for d in range(dim):
                if plot_reconstructions:
                    axi.plot(
                        np.linspace(0, 1, rec.shape[0]),
                        rec[:, d],
                        color=dim_colors[d],
                        lw=2,
                        zorder=10,
                    )
                if plot_trajectories:
                    axi.plot(
                        np.linspace(0, 1, traj.shape[0]),
                        traj[:, d],
                        color=dim_colors[d],
                        ls="--",
                        lw=2,
                        zorder=11,
                    )

            start = stop

        if equal_aspect:
            for j in range(n_cols):
                ax[i][j].set_aspect("equal")

        if component_borders is not None:
            for j in range(n_cols):
                for x in component_borders[i]:
                    ax[i][j].axvline(x=x, linestyle=":", color="k", alpha=0.5)

    dim_patches = [
        (Line2D([0], [0], color=c, lw=4, label=l))
        for c, l in zip(dim_colors, ["x", "y", "z"])
    ]
    type_patches = [
        Line2D([0], [0], color="k", lw=1, ls="--", label="original"),
        Line2D([0], [0], color="k", lw=1, ls="-", label="reconstruction"),
    ]

    if plot_coord_origin:
        type_patches.append(
            Line2D(
                [0],
                [0],
                color="k",
                lw=1.5,
                ls="dotted",
                alpha=0.2,
                label="frame origin",
            )
        )

    frame_patches = [
        Circle(
            xy=(0.5, 0.5),
            radius=0.25,
            facecolor="white",
            edgecolor=c,
            label=f,
            linewidth=2,
        )
        for c, f in zip(frame_colors, frame_names)
    ] + [
        Circle(
            xy=(0.5, 0.5),
            radius=0.25,
            facecolor="white",
            edgecolor="k",
            label="joint",
            linewidth=2,
        )
    ]

    dummy_patch = Line2D([0], [0], color="white", lw=0, label="")

    legend_rows = max(len(dim_patches), len(frame_patches), len(type_patches))
    if len(dim_patches) < legend_rows:
        dim_patches += [dummy_patch] * (legend_rows - len(dim_patches))
    if len(type_patches) < legend_rows:
        type_patches += [dummy_patch] * (legend_rows - len(type_patches))
    if len(frame_patches) < legend_rows:
        frame_patches += [dummy_patch] * (legend_rows - len(frame_patches))

    ax[0][0].legend(
        handles=dim_patches + frame_patches + type_patches,
        ncols=3,
        handler_map={Circle: HandlerEllipse()},
    )

    plt.tight_layout()

    if save_svg is not None:
        plt.savefig(save_svg, format="svg")

    plt.show()


def plot_tangent_data(per_frame, joint, size=(18, None), rot_range=[-1, 1]):
    n_traj = len(per_frame)

    frame_dim = joint[0].shape[1] if joint is not None else 12

    dim = 3

    n_frames = per_frame[0].shape[1] // frame_dim

    n_mani = frame_dim // dim

    n_cols = n_frames + 1

    width, height = size
    if height is None:
        size = (width, width * n_mani / (n_cols + 1))

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_mani, subplot_kw={"projection": "3d"})
    fig.set_size_inches(*size)

    mani_names = ["pos", "rot", "pos delta", "rot delta"]

    assert n_mani == 4, "lazyness"

    for i in range(n_mani):
        ax[i][0].set_title(mani_names[i], loc="left")
        for j in range(n_frames):
            for k in range(n_traj):
                ax[i][j].scatter(
                    per_frame[k][:, j * frame_dim + i * dim],
                    per_frame[k][:, j * frame_dim + i * dim + 1],
                    per_frame[k][:, j * frame_dim + i * dim + 2],
                    lw=1,
                )

            if i == 0:
                ax[i][j].set_title("frame {}".format(j))

            if i % 2 == 1 and rot_range is not None:
                ax[i][j].set_xlim(rot_range)
                ax[i][j].set_ylim(rot_range)
                ax[i][j].set_zlim(rot_range)

        if joint is not None:
            for k in range(n_traj):
                ax[i][-1].scatter(
                    joint[k][:, i * dim],
                    joint[k][:, i * dim + 1],
                    joint[k][:, i * dim + 2],
                    lw=1,
                )
            if i == 0:
                ax[i][-1].set_title("joint")


def debug_coord_origs(coord_frames, frame_names=None, n_cols=3, size=(18, None)):
    coord_frames = [c.cpu().numpy() for c in coord_frames]
    n_demos = len(coord_frames)

    n_rows = np.ceil(float(n_demos) / n_cols).astype(np.int) + 1

    fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, subplot_kw={"projection": "3d"})
    if size[1] is None and size[0] is not None:
        size = (size[0], size[0] / n_cols * n_rows)
    fig.set_size_inches(*size)

    ax = ax.reshape(-1)

    for i in range(n_demos):
        ax[i].set_aspect("equal")

        for j in range(coord_frames[i].shape[0]):
            traj = coord_frames[i][j]

            # Plots for when using the coordinates of the origin directly.
            # Below uses the homhogeneous transforms.
            # NOTE: for some reason, this breaks when using plot instead of
            # scatter.
            # ax[i].plot(traj[:, 0], traj[:, 1], traj[:, 2],
            #            color=tab_colors[j], label=frame_names[j],
            #            lw=200 if j ==2 else 2)
            # ax[i].scatter(traj[:, 0], traj[:, 1], traj[:, 2],
            #               color=tab_colors[j], label=frame_names[j])

            ax[i].plot(
                traj[:, 0, 3],
                traj[:, 1, 3],
                traj[:, 2, 3],
                color=tab_colors[j],
                label=frame_names[j],
                #   lw=20 if j == 0 else 2
            )

    ax[0].legend()

    for x in range(3):
        for y in range(n_demos):
            ax[x + i + 1].plot(
                coord_frames[y][x][:, 0, 3],
                coord_frames[y][x][:, 1, 3],
                coord_frames[y][x][:, 2, 3],
                color=tab_colors[y],
                label=f"traj {y}",
            )

            # ax[x+i+1].plot(
            #     coord_frames[y][x][:, 0], coord_frames[y][x][:, 1],
            #     coord_frames[y][x][:, 2], color=tab_colors[y],
            #     label=f'traj {y}')

    ax[i + 1].legend()

    plt.tight_layout()


def plot_hmm_transition_matrix(hmm):
    n_states = hmm.nb_states
    fig, ax = plt.subplots(1)
    pos = ax.imshow(
        np.log(hmm.Trans + 1e-10), interpolation="nearest", vmin=-5, cmap="viridis"
    )
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("log transition prob", rotation=270)
    plt.tight_layout()
    plt.grid()
    plt.xticks(np.arange(0, n_states, 1))
    plt.yticks(np.arange(0, n_states, 1))
    plt.show()


def plot_state_activation_for_demo(demo, models, size=(7.5, 3.6)):
    """
    Plot the state activation for each model for a given demonstration.

    Parameters
    ----------
    demo: torch.Tensor
        The demonstration in all frames.
        Shape: (n_observations, n_dims).
    models: dict[str, GMM/HMM/HSMM]
        The models to plot the state activation for.
    size: tuple[float/int]
        Size of the figure.
    """

    traces = {}

    for name, model in models.items():
        if type(model) == GMM:
            resp = model.compute_resp(demo)
            traces[name] = resp
        elif type(model) in [HMM, HSMM]:
            alpha, _, _, _, _ = model.compute_messages(demo)
            traces[name] = alpha
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

    n_models = len(models)

    fig, ax = plt.subplots(nrows=n_models)
    fig.set_size_inches(*size)

    for i, (name, trace) in enumerate(traces.items()):
        ax[i].plot(trace.T, lw=1)
        ax[i].set_ylim([-0.2, 1.2])
        ax[i].set_xlim([0, len(demo)])
        ax[i].set_title(name)

    plt.xlabel("timestep")
    plt.tight_layout()


def plot_state_sequence_for_demo(demo, model, size=(5, 1)):
    q = model.viterbi(demo)

    if type(model) == GMM:
        act = model.compute_resp(demo)
    elif type(model) in [HMM, HSMM]:
        act, _, _, _, _ = model.compute_messages(demo)

    fig, ax = plt.subplots(1)
    fig.set_size_inches(*size)

    ax.plot(q, lw=3)
    ax.plot(act.T * (model.nb_states - 0.1), lw=1)
    plt.xlabel("timestep")
    ax.set_xlim([0, len(demo)])
    ax.set_yticks(np.arange(0, model.nb_states, 1))


def plot_coordinate_frame_trajs(frame_origs, subsample_n=50):
    n_rows = len(frame_origs)
    n_frames = frame_origs[0].shape[0]
    n_cols = n_frames

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={"projection": "3d"})
    fig.set_size_inches(4 * n_cols, 4 * n_rows)

    if n_rows == 1:
        ax = ax[None, :]
    if n_cols == 1:
        ax = ax[:, None]

    for i in range(n_rows):
        for j in range(n_cols):
            traj = frame_origs[i][j]
            ss_idx = torch.linspace(0, traj.shape[0] - 1, subsample_n).long()
            ss_traj = traj[ss_idx]

            plot_range = [-0.5, 0.5]
            ax[i, j].set_xlim(plot_range)
            ax[i, j].set_ylim(plot_range)
            ax[i, j].set_zlim(plot_range)
            # ax[i, j].scatter(
            #     ss_traj[:, 0, 6], ss_traj[:, 1, 6], ss_traj[:, 2, 6])
            for k in range(subsample_n):
                # Transforms are 7x7 (state and action), extract state part
                state_mask = (
                    [True for _ in range(3)] + [False for _ in range(3)] + [True]
                )
                orig = torch.Tensor(ss_traj[k][state_mask][:, state_mask]).double()
                plot_coordindate_frame(
                    ax[i, j], transformation=orig, linewidth=0.5, arrow_length=0.4
                )
    plt.show()


def plot_log_map_quaternions(n_steps=100, base="180y", axis="z"):
    import riepybdlib

    def rotation_quaternions_x_axis(
        num_quaternions, min_angle=0, max_angle=2 * np.pi + 0.3
    ):
        angles = np.linspace(min_angle, max_angle, num_quaternions, endpoint=True)
        axis = np.tile([1, 0, 0], (num_quaternions, 1))
        sines = np.sin(angles / 2)
        cosines = np.cos(angles / 2)
        quaternions = np.hstack((cosines[:, np.newaxis], axis * sines[:, np.newaxis]))
        return quaternions, angles

    def rotation_quaternions_y_axis(
        num_quaternions, min_angle=0, max_angle=2 * np.pi + 0.3
    ):
        angles = np.linspace(min_angle, max_angle, num_quaternions, endpoint=True)
        axis = np.tile([0, 1, 0], (num_quaternions, 1))
        sines = np.sin(angles / 2)
        cosines = np.cos(angles / 2)
        quaternions = np.hstack((cosines[:, np.newaxis], axis * sines[:, np.newaxis]))
        return quaternions, angles

    def rotation_quaternions_z_axis(
        num_quaternions, min_angle=0, max_angle=2 * np.pi + 0.3
    ):
        angles = np.linspace(min_angle, max_angle, num_quaternions, endpoint=True)
        axis = np.tile([0, 0, 1], (num_quaternions, 1))
        sines = np.sin(angles / 2)
        cosines = np.cos(angles / 2)
        quaternions = np.hstack((cosines[:, np.newaxis], axis * sines[:, np.newaxis]))
        return quaternions, angles

    if axis == "x":
        quats, angles = rotation_quaternions_x_axis(n_steps)
    elif axis == "y":
        quats, angles = rotation_quaternions_y_axis(n_steps)
    elif axis == "z":
        quats, angles = rotation_quaternions_z_axis(n_steps)
    else:
        raise ValueError("Unknown axis")

    if base is None:
        base = riepybdlib.angular_representations.Quaternion(1.00, np.array([0, 0, 0]))
    elif base == "180y":
        base = riepybdlib.angular_representations.Quaternion(
            0.00, np.array([-0.06, -1.0, 0])
        )
    else:
        raise ValueError("Unknown base")

    manifold = riepybdlib.manifold.get_quaternion_manifold()

    mani_data = manifold.np_to_manifold(quats)
    log_data = manifold.log(mani_data, base=base)

    comps = ["x", "y", "z"]

    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(12, 4)

    # make axis 3d
    ax[0].remove()
    ax[0] = fig.add_subplot(1, 4, 1, projection="3d")
    plot_rot_from_quaternion(ax[0], base)

    for i in range(3):
        ax[i + 1].plot(angles, log_data[:, i], c=dim_colors[i])
        ax[i + 1].set_xlabel(f"angle around {axis}-axis (rad)")
        ax[i + 1].set_ylabel(f"log map {comps[i]}")

        fig.suptitle(
            f"Log map components of different rotations around {axis}-axis"
            "under the base shown on the left"
        )
    fig.tight_layout()


def plot_s2_gaussians(
    mus: np.ndarray | None,
    sigmas: np.ndarray | None,
    data: np.ndarray | None = None,
    auto_rotate: bool = True,
):
    if mus is not None and len(mus.shape) == 2:  # single-frame GMM
        mus = mus[None, :]

    n_frames = data.shape[1] if mus is None else mus.shape[0]
    n_gauss = 1 if mus is None else mus.shape[1]

    fig, ax = plt.subplots(n_gauss, 1, subplot_kw={"projection": "3d"})
    fig.set_size_inches(4, 4 * n_gauss)

    if n_gauss == 1:
        ax = [ax]

    for s in range(n_gauss):
        riepybdlib.s2_fcts.plot_manifold(ax[s])
        if mus is not None:
            for f in range(n_frames):
                riepybdlib.s2_fcts.plot_gaussian(
                    ax[s], mus[f, s], sigmas[f, s], color=tab_colors[f]
                )

        if data is not None:
            for f in range(n_frames):
                ax[s].scatter(
                    data[:, f, 0],
                    data[:, f, 1],
                    data[:, f, 2],
                    color=tab_colors[f],
                    # color='k'
                )

        if auto_rotate:
            assert data is not None
            ax_dim_mean = data.reshape(-1, 3).mean(axis=0)
            xy_angle = 90 - np.arctan2(ax_dim_mean[0], ax_dim_mean[1]) * 360 / (
                2 * np.pi
            )
            z_angle = 90 - np.arccos(ax_dim_mean[2]) * 360 / (2 * np.pi)

            ax[s].view_init(elev=z_angle, azim=xy_angle, roll=0)


def plot_traj_topp(
    raw_data: np.ndarray,
    smoothed_data: np.ndarray,
    coord_frames: np.ndarray | None = None,
    segment_idcs: np.ndarray | None = None,
):
    fig, ax = plt.subplots(1, 3)
    time_raw = np.linspace(0, 1, raw_data.shape[0])
    time_smoothed = np.linspace(0, 1, smoothed_data.shape[0])

    for i in range(3):
        ax[0].plot(
            time_raw, raw_data[:, i], label=f"raw {i}", ls="--", color=dim_colors[i]
        )
        ax[0].plot(
            time_smoothed,
            smoothed_data[:, i],
            label=f"smoothed {i}",
            color=dim_colors[i],
        )
        if coord_frames is not None:
            for f in range(coord_frames.shape[0]):
                ax[0].axhline(y=coord_frames[f, i], color=dim_colors[i], ls="dotted")
    for i in range(4):
        if raw_data.shape[1] == 8:
            ax[1].plot(
                time_raw,
                raw_data[:, i + 3],
                label=f"raw {i}",
                ls="--",
                color=quat_colors[i],
            )
            if coord_frames is not None:
                for f in range(coord_frames.shape[0]):
                    ax[1].axhline(
                        y=coord_frames[f, i + 3], color=quat_colors[i], ls="dotted"
                    )
        if smoothed_data.shape[1] == 8:
            ax[1].plot(
                time_smoothed,
                smoothed_data[:, i + 3],
                label=f"smoothed {i}",
                color=quat_colors[i],
            )
    if segment_idcs is not None:
        for i in range(3):
            for s in range(segment_idcs.shape[0]):
                ax[i].axvline(segment_idcs[s] / raw_data.shape[0], color="k", ls="--")
    ax[2].plot(time_raw, raw_data[:, -1], label="gripper raw", ls="--", color="k")
    ax[2].plot(time_smoothed, smoothed_data[:, -1], label="smoothed", color="k")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()
