import matplotlib.pyplot as plt
from matplotlib import gridspec

from tapas_gmm.utils.multi_processing import mp_wrapper


@mp_wrapper
def image_pair_with_points_overlay(img_1, img_2, indeces_1, indeces_2):
    # show images with dot overlays
    gs = gridspec.GridSpec(
        1,
        2,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (3),
        bottom=0.5 / (2),
        left=0.5 / (3),
        right=1 - 0.5 / (3),
    )
    colors = list(range(len(indeces_1)))
    ax = plt.subplot(gs[0, 0])
    ax.imshow(img_1)
    ax.scatter(
        [i[1] for i in indeces_1],
        [i[0] for i in indeces_1],
        c=colors,
        cmap=plt.cm.rainbow,
    )
    ax.axis("off")
    ax = plt.subplot(gs[0, 1])
    ax.imshow(img_2)
    ax.scatter(
        [i[1] for i in indeces_2],
        [i[0] for i in indeces_2],
        c=colors,
        cmap=plt.cm.rainbow,
    )
    ax.axis("off")
    plt.show()


image_pair_with_matches = image_pair_with_points_overlay


@mp_wrapper
def image_pair_with_non_matches(img_1, img_2, indeces_1, indeces_2, k):
    # show images with dot overlays
    gs = gridspec.GridSpec(
        1,
        2,
        wspace=0,
        hspace=0,
        top=1.0 - 0.5 / (3),
        bottom=0.5 / (2),
        left=0.5 / (3),
        right=1 - 0.5 / (3),
    )
    colors = list(range(len(indeces_1)))
    ax = plt.subplot(gs[0, 0])
    ax.imshow(img_1)
    ax.scatter(
        [i[1] for i in indeces_1],
        [i[0] for i in indeces_1],
        c=colors,
        cmap=plt.cm.rainbow,
    )
    ax.axis("off")
    colors = [i for i in colors for j in range(k)]  # k non-matches per point
    ax = plt.subplot(gs[0, 1])
    ax.imshow(img_2)
    ax.scatter(
        [i[1] for i in indeces_2],
        [i[0] for i in indeces_2],
        c=colors,
        cmap=plt.cm.rainbow,
    )
    ax.axis("off")
    plt.show()
