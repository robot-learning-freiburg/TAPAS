import matplotlib.pyplot as plt
import numpy as np

# from tapas_gmm.utils.multi_processing import mp_wrapper


# @mp_wrapper
def depth_map_with_points_overlay_uv_list(
    depth,
    indeces,
    mask=None,
    object_poses=None,
    object_labels=None,
):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_size_inches(18.5, 10.5)

    H, W = depth.shape
    X = np.arange(0, W, 1)
    Y = np.arange(0, H, 1)
    X, Y = np.meshgrid(X, Y)

    print("depth.shape", depth.shape)

    im = ax.plot_surface(
        X, Y, depth, cmap="GnBu", linewidth=0, antialiased=False, alpha=0.5
    )

    # add colorbar for surface
    fig.colorbar(im, ax=ax)

    if mask is not None:
        # get idy, idx where the value of mask is in object_labels
        idy, idx = np.where(mask == object_labels[0])
        depthm = depth[idy, idx]
        ax.scatter(idx, idy, depthm, color="Red", alpha=0.1, s=0.1)

    ax.scatter(
        indeces[0],
        indeces[1],
        depth[indeces[1], indeces[0]],
        color="black",
        edgecolors="white",
        s=128,
        alpha=1,
        linewidths=3,
    )

    if object_poses is not None:
        ax.scatter(
            object_poses[:, 0],
            object_poses[:, 1],
            object_poses[:, 2],
            color="b",
            edgecolors="white",
            s=128,
            alpha=1,
            linewidths=3,
        )

    # ax.view_init(elev=75, azim=60)
    # ax.view_init(elev=25, azim=-20, roll=-40)
    ax.view_init(elev=90, azim=-90)
    ax.invert_zaxis()
    ax.invert_yaxis()

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    n_points = len(indeces[0])
    for i, txt in enumerate(range(n_points)):
        ax.text(
            indeces[0][i],
            indeces[1][i],
            depth[indeces[1][i], indeces[0][i]] + 0.1,
            str(i),
            None,
        )

    # ax.set_aspect('equal', 'box')

    plt.show()


def scatter3d(tens):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # color dots by depth
    im = ax.scatter(tens[:, 0], tens[:, 1], tens[:, 2], c=tens[:, 2], cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.scatter(0, 0, 0, color="red")

    # ax.view_init(elev=0, azim=0, roll=0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=90, azim=-90)

    ax.invert_yaxis()

    plt.show()
