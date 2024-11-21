import math

import matplotlib.pyplot as plt
from matplotlib import gridspec

from tapas_gmm.utils.multi_processing import mp_wrapper


@mp_wrapper
def point_cloud_pair_with_mutuals(points_1, points_2, points_m):
    # show point clouds with mutual points highlighted
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        [p[0] for p in points_1],
        [p[1] for p in points_1],
        [p[2] for p in points_1],
        marker="o",
        alpha=0.2,
        s=0.25,
        color="blue",
    )
    ax.scatter(
        [p[0] for p in points_2],
        [p[1] for p in points_2],
        [p[2] for p in points_2],
        marker="o",
        alpha=0.2,
        s=0.25,
        color="green",
    )
    ax.scatter(
        [p[0] for p in points_m],
        [p[1] for p in points_m],
        [p[2] for p in points_m],
        marker="o",
        color="red",
        alpha=1,
        s=100,
    )
    plt.show()


@mp_wrapper
def create_series(array_list, name=None):
    num_images = len(array_list)

    no_cols = int(math.ceil(math.sqrt(num_images)))
    no_rows = int(math.ceil(num_images / no_cols))

    fig = plt.figure(figsize=(no_cols + 1, no_rows + 1), dpi=320)

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
        ax = fig.add_subplot(gs[x, y], projection="3d")
        points = array_list[i]
        alpha = min(1, 1 / int(len(points) ** (1.0 / 3.0)))
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            [p[2] for p in points],
            marker="o",
            alpha=alpha,
            s=0.25,
            color="green",
        )
        ax.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)

    if name is None:
        plt.show()
    else:
        plt.savefig(name)
