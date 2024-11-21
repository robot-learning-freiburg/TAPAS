import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import riepybdlib as rbd
import torch
from loguru import logger
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform

from tapas_gmm.utils.geometry_np import (
    quaternion_to_axis_and_angle,
    quaternion_to_axis_angle,
)


def plot_ellipsoid_from_gaussian3d(ax, mean, cov, n_std=1, n_seg=60, **kwargs):
    if np.isnan(cov).any():
        logger.info("Found NaN in covariance matrix, skipping ellipsoid plot.")
        return

    U, s, rotation = np.linalg.svd(cov)
    radii = np.sqrt(s) * n_std

    plot_ellipsoid3d(ax, mean, rotation, radii, n_seg=n_seg, **kwargs)


def plot_ellipsoid3d(ax, center, rotation, radii, n_seg=60, **kwargs):
    u = np.linspace(0.0, 2.0 * np.pi, n_seg)
    v = np.linspace(0.0, np.pi, n_seg)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    for i in range(len(x)):
        for j in range(len(x)):
            [x[i, j], y[i, j], z[i, j]] = (
                np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
            )

    kwargs.update(
        {
            "rstride": 3,
            "cstride": 3,
            "linewidth": 0.1,
            "shade": True,
        }
    )

    ax.plot_surface(x, y, z, **kwargs)


def colorline(
    x,
    y,
    z=None,
    cmap=plt.get_cmap("copper"),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=3,
    alpha=1.0,
):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, array=z, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def colorline_3d(
    x, y, z, c=0, cmap="tab10", norm=None, linewidth=None, alpha_range=(0.2, 1.0)
):
    """
    Plot a colored line with coordinates x, y and z
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    t = len(x)

    alphas = np.linspace(alpha_range[0], alpha_range[1], t)

    colors = np.array(plt.get_cmap(cmap).colors)[c][np.newaxis, :].repeat(t, axis=0)
    colors = np.concatenate((colors[:t], alphas[:, np.newaxis]), axis=1)

    segments = make_segments_3d(x, y, z)
    lc = Line3DCollection(
        segments, array=z, color=colors, norm=norm, linewidth=linewidth
    )

    return lc


def make_segments_3d(x, y, z):
    """
    Create list of line segments from x, y, z coordinates.
    """
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


class Arrow3D(FancyArrowPatch):
    # Taken from https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_rotation_basis(rotations):
    n_components = len(rotations)
    n_frames = len(rotations[0])

    fig, ax = plt.subplots(n_frames, n_components, subplot_kw={"projection": "3d"})
    fig.set_size_inches(n_components * 5, n_frames * 5)

    for i in range(n_frames):
        for j in range(n_components):
            plot_rot_from_quaternion(ax[i][j], rotations[j][i])

    plt.show()


def plot_rot_from_quaternion(ax, quat, **kwargs):
    # axis, angles = quat.axis_angle()
    # angles = angles * axis
    quat_np = quat.to_nparray()
    angles = quaternion_to_axis_angle(quat_np)
    _, ang = quaternion_to_axis_and_angle(quat_np)
    ang_degrees = ang * 180 / np.pi
    plot_axis_angle(ax, angles, **kwargs, annotation=f"basis ({ang_degrees:.2f}°)")


def plot_axis_angle(
    ax,
    angles,
    linewidth=3,
    arrow_length=1,
    color="k",
    annotation=None,
    annotation_size=20,
):
    origin = [0, 0, 0]

    dest = origin + angles * arrow_length

    min_val = np.min(dest)
    max_val = max(np.max(dest), 1)

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)

    ax.arrow3D(
        *origin,
        *dest,
        color=color,
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )

    ax.arrow3D(
        *origin,
        *[1, 0, 0],
        color="r",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )
    ax.arrow3D(
        *origin,
        *[0, 1, 0],
        color="g",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )
    ax.arrow3D(
        *origin,
        *[0, 0, 1],
        color="b",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )

    if annotation is not None:
        ax.text(*dest * 1.2, annotation, size=annotation_size, zorder=20, color="k")

    ax.text(*[1, 0, 0], "x", size=annotation_size, zorder=20, color="r")
    ax.text(*[0, 1, 0], "y", size=annotation_size, zorder=20, color="g")
    ax.text(*[0, 0, 1], "z", size=annotation_size, zorder=20, color="b")


def plot_coordindate_frame(
    ax,
    origin=None,
    rotation=None,
    transformation=None,
    linewidth=3,
    arrow_length=1,
    annotation=None,
    annotation_size=20,
):
    if origin is None:
        assert rotation is None
        assert transformation is not None

        origin = (transformation @ torch.Tensor([0, 0, 0, 1]).double())[:3]
        xhat = (transformation @ torch.Tensor([arrow_length, 0, 0, 1]).double())[:3]
        yhat = (transformation @ torch.Tensor([0, arrow_length, 0, 1]).double())[:3]
        zhat = (transformation @ torch.Tensor([0, 0, arrow_length, 1]).double())[:3]

    else:
        xhat = origin + rotation[:3, 0] * arrow_length
        yhat = origin + rotation[:3, 1] * arrow_length
        zhat = origin + rotation[:3, 2] * arrow_length

    ax.arrow3D(
        *origin,
        *(xhat - origin),
        color="r",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )
    ax.arrow3D(
        *origin,
        *(yhat - origin),
        color="g",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )
    ax.arrow3D(
        *origin,
        *(zhat - origin),
        color="b",
        arrowstyle="-|>",
        mutation_scale=20,
        linewidth=linewidth,
    )

    if annotation is not None:
        ax.text(*origin, annotation, size=annotation_size, zorder=20, color="k")


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)
