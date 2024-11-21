import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, gridspec
from matplotlib.colors import ListedColormap

from tapas_gmm.viz.operations import channel_front2back, rgb2gray


def vis_series(img_tensor, channeled=True, file_name=None):
    num_images = img_tensor.shape[0]

    no_cols = int(math.ceil(math.sqrt(num_images)))
    no_rows = no_cols

    fig = plt.figure(figsize=(no_cols + 1, no_rows + 1))

    gs = gridspec.GridSpec(
        no_rows,
        no_cols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (no_rows + 1),
        bottom=0.5 / (no_rows + 1),
        left=0.5 / (no_cols + 1),
        right=1 - 0.5 / (no_cols + 1),
    )

    for i in range(num_images):
        x = i // no_cols
        y = i % no_cols
        ax = fig.add_subplot(gs[x, y])
        img_i = channel_front2back(img_tensor[i]) if channeled else img_tensor[i]
        ax.imshow(img_i)
        ax.axis("off")
        ax.set_aspect("equal")
    if file_name:
        fig.savefig("{}.png".format(file_name))
        plt.close(fig)
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


def vis_series_w_mask(img_tensor, mask):
    num_images = img_tensor.shape[0]

    no_cols = int(math.ceil(math.sqrt(num_images)))
    no_rows = no_cols

    plt.figure(figsize=(no_cols + 1, no_rows + 1))

    gs = gridspec.GridSpec(
        no_rows,
        no_cols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (no_rows + 1),
        bottom=0.5 / (no_rows + 1),
        left=0.5 / (no_cols + 1),
        right=1 - 0.5 / (no_cols + 1),
    )

    set_map = cm.get_cmap("Set1", 9)
    newcolors = set_map(np.linspace(0, 1, 9))
    transp = np.array([0, 0, 0, 0])
    newcolors[1:, :] = newcolors[:8, :]
    newcolors[0, :] = transp
    mask_map = ListedColormap(newcolors)

    for i in range(num_images):
        x = i // no_cols
        y = i % no_cols
        ax = plt.subplot(gs[x, y])
        ax.imshow(rgb2gray(channel_front2back(img_tensor[i])), cmap="gray", alpha=0.5)
        ax.imshow(
            mask[i], cmap=mask_map, alpha=0.8, interpolation="none", vmin=0, vmax=9
        )
        ax.axis("off")
        ax.set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def vis_series_w_depth(ndarray, depth):
    num_images = ndarray.shape[0]

    no_cols = int(math.ceil(math.sqrt(num_images)))
    no_rows = no_cols

    plt.figure(figsize=(no_cols + 1, no_rows + 1))

    gs = gridspec.GridSpec(
        no_rows,
        no_cols,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (no_rows + 1),
        bottom=0.5 / (no_rows + 1),
        left=0.5 / (no_cols + 1),
        right=1 - 0.5 / (no_cols + 1),
    )

    for i in range(num_images):
        x = i // no_cols
        y = i % no_cols
        ax = plt.subplot(gs[x, y])
        ax.imshow(rgb2gray(channel_front2back(ndarray[i])), cmap="gray", alpha=0.5)
        ax.imshow(
            depth[i], cmap="jet", alpha=0.8, interpolation="none", vmin=0, vmax=0.5
        )
        ax.axis("off")
        ax.set_aspect("equal")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
