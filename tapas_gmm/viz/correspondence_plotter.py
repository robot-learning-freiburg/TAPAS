import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle

from tapas_gmm.viz.operations import flattened_pixel_locations_to_u_v


def plot_heatmaps(
    img_a_rgb,
    img_b_rgb,
    embed_a,
    embed_b,
    uv_a=None,
    uv_b=None,
    circ_color="g",
    show=True,
    title=None,
):
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    img_height, img_width = img_a_rgb.shape[:2]
    emb_height, emb_width = embed_a.shape[:2]
    scale_factor = int(img_height / emb_height)
    embed_a = np.repeat(np.repeat(embed_a, scale_factor, axis=1), scale_factor, axis=0)
    embed_b = np.repeat(np.repeat(embed_b, scale_factor, axis=1), scale_factor, axis=0)

    pixel_locs = [uv_a, uv_b, uv_a, uv_b]
    images = [img_a_rgb, img_b_rgb, embed_a, embed_b]
    axes = axes.flat[0:]
    n_points = len(uv_a[0])
    for ax, img, pixel_loc in zip(axes[0:], images, pixel_locs):
        ax.set_aspect("equal")
        ax.imshow(img)
        if uv_a is not None and uv_b is not None:  # TODO: flatten nested if
            if isinstance(pixel_loc[0], (int, float)):
                circ = Circle(
                    pixel_loc,
                    radius=3,
                    facecolor=circ_color,
                    edgecolor="white",
                    fill=True,
                    linewidth=2.0,
                    linestyle="solid",
                )
                ax.add_patch(circ)
            else:
                for x, y in zip(pixel_loc[0], pixel_loc[1]):
                    circ = Circle(
                        (x, y),
                        radius=3,
                        facecolor=circ_color,
                        edgecolor="white",
                        fill=True,
                        linewidth=2.0,
                        linestyle="solid",
                    )
                    ax.add_patch(circ)

            for i, txt in enumerate(range(n_points)):
                ax.annotate(str(i), (pixel_loc[0][i], pixel_loc[1][i]))

    fig.suptitle(title)
    if show:
        plt.show()
        return None
    else:
        return fig, axes


def cross_debug_plot(
    image_a_rgb,
    image_b_rgb,
    image_a_depth_numpy,
    image_b_depth_numpy,
    blind_uv_a,
    blind_uv_b,
):
    num_matches_to_plot = 10

    plot_blind_uv_a, plot_blind_uv_b = subsample_tuple_pair(
        blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot * 10
    )

    plot_correspondences_direct(
        image_a_rgb,
        image_a_depth_numpy,
        image_b_rgb,
        image_b_depth_numpy,
        plot_blind_uv_a,
        plot_blind_uv_b,
        circ_color="k",
        show=True,
        title="Blind non-matches",
    )


def debug_plots(
    image_a_rgb,
    image_b_rgb,
    image_height,
    image_a_depth_numpy,
    image_b_depth_numpy,
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
):
    # downsample so can plot
    num_matches_to_plot = 1
    plot_uv_a, plot_uv_b = subsample_tuple_pair(
        uv_a, uv_b, num_samples=num_matches_to_plot
    )

    plot_uv_a_masked_long, plot_uv_b_masked_non_matches_long = subsample_tuple_pair(
        uv_a_masked_long,
        uv_b_masked_non_matches_long,
        num_samples=num_matches_to_plot * 3,
    )

    (
        plot_uv_a_background_long,
        plot_uv_b_background_non_matches_long,
    ) = subsample_tuple_pair(
        uv_a_background_long,
        uv_b_background_non_matches_long,
        num_samples=num_matches_to_plot * 3,
    )

    blind_uv_a = flattened_pixel_locations_to_u_v(blind_non_matches_a, image_width)
    if blind_uv_b is not None:
        plot_blind_uv_a, plot_blind_uv_b = subsample_tuple_pair(
            blind_uv_a, blind_uv_b, num_samples=num_matches_to_plot * 10
        )

    fig, axes = plot_correspondences_direct(
        image_a_rgb,
        image_a_depth_numpy,
        image_b_rgb,
        image_b_depth_numpy,
        plot_uv_a,
        plot_uv_b,
        show=False,
        masks=[image_a_mask, image_b_mask, image_a_mask, image_b_mask],
    )

    plot_correspondences_direct(
        image_a_rgb,
        image_a_depth_numpy,
        image_b_rgb,
        image_b_depth_numpy,
        plot_uv_a_masked_long,
        plot_uv_b_masked_non_matches_long,
        use_previous_plot=(fig, axes),
        circ_color="r",
        title="sub-sampled matches and non matches, foreground",
    )

    fig, axes = plot_correspondences_direct(
        image_a_rgb,
        image_a_depth_numpy,
        image_b_rgb,
        image_b_depth_numpy,
        plot_uv_a,
        plot_uv_b,
        show=False,
        masks=[image_a_mask, image_b_mask, image_a_mask, image_b_mask],
    )

    plot_correspondences_direct(
        image_a_rgb,
        image_a_depth_numpy,
        image_b_rgb,
        image_b_depth_numpy,
        plot_uv_a_background_long,
        plot_uv_b_background_non_matches_long,
        use_previous_plot=(fig, axes),
        circ_color="b",
        title="sub-sampled matches and non matches, background",
    )

    if blind_uv_b is not None:
        plot_correspondences_direct(
            image_a_rgb,
            image_a_depth_numpy,
            image_b_rgb,
            image_b_depth_numpy,
            plot_blind_uv_a,
            plot_blind_uv_b,
            circ_color="k",
            show=True,
            title="sub-sampled blind matches",
            masks=[image_a_mask, image_b_mask, image_a_mask, image_b_mask],
        )

    # Mask-plotting city
    plt.imshow(np.asarray(image_a_mask))
    plt.title("Mask of img a object pixels")
    plt.axis("off")
    plt.show()

    plt.imshow(1 - np.asarray(image_a_mask))
    plt.title("Mask of img a background")
    plt.axis("off")
    plt.show()

    temp = matches_a_mask.view(image_height, -1)
    plt.imshow(temp)
    plt.title("Mask of img a object pixels for which there was a match")
    plt.axis("off")
    plt.show()

    temp2 = (mask_a_flat - matches_a_mask).view(image_height, -1)
    plt.imshow(temp2)
    plt.title("Mask of img a object pixels for which there was NO match")
    plt.axis("off")
    plt.show()


def plot_correspondences(
    images,
    uv_a,
    uv_b,
    use_previous_plot=None,
    circ_color="g",
    show=True,
    title=None,
    masks=None,
):
    if use_previous_plot is None:
        fig, axes = plt.subplots(nrows=2, ncols=2)
    else:
        fig, axes = use_previous_plot[0], use_previous_plot[1]

    if masks is None:
        masks = [None, None, None, None]

    fig.set_figheight(10)
    fig.set_figwidth(15)
    pixel_locs = [uv_a, uv_b, uv_a, uv_b]
    axes = axes.flat[0:]
    # axes_flat = axes
    if use_previous_plot is not None:
        axes = [axes[1], axes[3]]
        images = [images[1], images[3]]
        pixel_locs = [pixel_locs[1], pixel_locs[3]]
    for ax, img, pixel_loc, mask in zip(axes[0:], images, pixel_locs, masks):
        ax.set_aspect("equal")
        if use_previous_plot is None:
            ax.imshow(img)
        if mask is not None:
            ax.imshow(mask, cmap="Reds", alpha=0.5)
        if isinstance(pixel_loc[0], (int, float)):
            circ = Circle(
                pixel_loc,
                radius=3,
                facecolor=circ_color,
                edgecolor="white",
                fill=True,
                linewidth=1.0,
                linestyle="solid",
            )
            ax.add_patch(circ)
        else:
            for i, (x, y) in enumerate(zip(pixel_loc[0], pixel_loc[1])):
                circ = Circle(
                    (x, y),
                    radius=3,
                    facecolor=circ_color,
                    edgecolor="white",
                    fill=True,
                    linewidth=1.0,
                    linestyle="solid",
                )
                ax.add_patch(circ)

                # ax.annotate(str(i), (pixel_loc[0][i], pixel_loc[1][i]),
                #             c="white", size='x-large')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.suptitle(title)
    if show:
        plt.show()
        # for i, ax in enumerate(axes_flat):
        #     extent = ax.get_window_extent().transformed(
        #         fig.dpi_scale_trans.inverted())
        #     fig.savefig('corr-finder{}.png'.format(i), transparent=True,
        #                 bbox_inches=extent.expanded(1.2, 1.2))
        return None
    else:
        return fig, axes


def plot_correspondences_direct(
    img_a_rgb,
    img_a_depth,
    img_b_rgb,
    img_b_depth,
    uv_a,
    uv_b,
    use_previous_plot=None,
    circ_color="g",
    show=True,
    title=None,
    masks=None,
):
    """

    Plots rgb and depth image pair along with circles at pixel locations
    :param img_a_rgb: PIL.Image.Image
    :param img_a_depth: PIL.Image.Image
    :param img_b_rgb: PIL.Image.Image
    :param img_b_depth: PIL.Image.Image
    :param uv_a: (u,v) pixel location, or list of pixel locations
    :param uv_b: (u,v) pixel location, or list of pixel locations
    :param use_previous_plot:
    :param circ_color: str
    :param show:
    :return:
    """
    images = [img_a_rgb, img_b_rgb, img_a_depth, img_b_depth]
    return plot_correspondences(
        images,
        uv_a,
        uv_b,
        use_previous_plot=use_previous_plot,
        circ_color=circ_color,
        show=show,
        title=title,
        masks=masks,
    )


def subsample_tuple(uv, num_samples):
    """
    Subsamples a tuple of (torch.Tensor, torch.Tensor)
    """
    indexes_to_keep = (
        (torch.rand(num_samples) * len(uv[0])).floor().type(torch.LongTensor)
    )
    return (
        torch.index_select(uv[0], 0, indexes_to_keep),
        torch.index_select(uv[1], 0, indexes_to_keep),
    )


def subsample_tuple_pair(uv_a, uv_b, num_samples):
    """
    Subsamples a pair of tuples, i.e. (torch.Tensor, torch.Tensor),
    (torch.Tensor, torch.Tensor)
    """
    assert len(uv_a[0]) == len(uv_b[0])
    indexes_to_keep = (
        (torch.rand(num_samples) * len(uv_a[0])).floor().type(torch.LongTensor)
    )
    uv_a_downsampled = (
        torch.index_select(uv_a[0], 0, indexes_to_keep),
        torch.index_select(uv_a[1], 0, indexes_to_keep),
    )
    uv_b_downsampled = (
        torch.index_select(uv_b[0], 0, indexes_to_keep),
        torch.index_select(uv_b[1], 0, indexes_to_keep),
    )
    return uv_a_downsampled, uv_b_downsampled
