import matplotlib.pyplot as plt
import numpy as np

from tapas_gmm.utils.multi_processing import mp_wrapper
from tapas_gmm.viz.operations import channel_front2back, scale


@mp_wrapper
def image_with_points_overlay(img, indeces, mask=None):
    fig1, ax1 = plt.subplots()
    ax1.imshow(img)
    ax1.scatter([i[1] for i in indeces], [i[0] for i in indeces])
    if mask is not None:
        plt.imshow(mask, cmap="Reds", alpha=0.5)
    ax1.axis("off")
    plt.show()


@mp_wrapper
def image_with_points_overlay_uv_list(img, indeces, mask=None):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(indeces[0], indeces[1])
    n_points = len(indeces[0])
    for i, txt in enumerate(range(n_points)):
        ax.annotate(str(i), (indeces[0][i], indeces[1][i]))
    if mask is not None:
        ax.imshow(mask, cmap="Reds", alpha=0.5)
    ax.axis("off")
    plt.show()


def figure_emb_with_points_overlay(
    emb,
    kp,
    dist,
    best,
    threshold=1,
    is_img=True,
    colors="b",
    rescale=True,
    annotate=True,
):
    # mostly taken from live_keypoint viz.
    fig, ax = plt.subplots()

    if is_img:  # 3-channeled img with channels in front. else single channel.
        emb = channel_front2back(emb)
    img = scale(emb.numpy()).astype(np.float32)
    H, W = img.shape[:2]
    ax.imshow(img)

    keypoints_x, keypoints_y = np.array_split(kp.squeeze(0).numpy(), 2)
    if rescale:
        pos_x = [round((i + 1) * (W - 1) / 2) for i in keypoints_x]
        pos_y = [round((i + 1) * (H - 1) / 2) for i in keypoints_y]
    else:
        pos_x = keypoints_x
        pos_y = keypoints_y

    n_points = keypoints_x.shape[0]

    if dist is not None and best is not None:
        colors = [
            "w" if not b else "r" if d < threshold else "b" for d, b in zip(dist, best)
        ]

    ax.scatter(pos_x, pos_y, c=colors)

    if annotate:
        for i, txt in enumerate(range(n_points)):
            ax.annotate(str(i), (pos_x[i], pos_y[i]))

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent = extent.expanded(1.2, 1.2)

    return fig, extent
