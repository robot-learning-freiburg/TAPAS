"""

The purpose of this file is to perform data augmentation for images
and lists of pixel positions in them.

- For operations on the images, we can use functions optimized
for image data.

- For operations on a list of pixel indices, we need a matching
implementation.

"""

import random

import numpy as np
import torch

# TODO: in dataset restructure, removed ops moving the channel dim to the back,
# ie. Images are tensors now with (C, H, W). Update docstrings here.
# Remove np, cleanup.


def random_flip(images, uv_pixel_positions):
    """
    This function takes a list of images and a list of pixel positions in the
    image, and picks some subset of available mutations.

    :param images: a list of images (for example the rgb, depth, and mask) for
                        which the **same** mutation will be applied
    :type  images: list of torch tensors

    :param uv_pixel_positions: pixel locations (u, v) in the image.
        See doc/coordinate_conventions.md for definition of (u, v)

    :type  uv_pixel_positions: a tuple of torch Tensors, each of length n, i.e:
        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: aim is to support both torch.LongTensor and torch.FloatTensor,
            and return the mutated_uv_pixel_positions with same type

    :return mutated_image_list, mutated_uv_pixel_positions
    :rtype: list of torch tensor, tuple of torch Tensors

    """

    # Current augmentation is:
    # 50% do nothing
    # 50% rotate the image 180 degrees (by applying flip vertical then flip
    # horizontal)

    if random.random() < 0.5:
        return images, uv_pixel_positions

    else:
        mutated_images, mutated_uv_pixel_positions = flip_vertical(
            images, uv_pixel_positions
        )
        mutated_images, mutated_uv_pixel_positions = flip_horizontal(
            mutated_images, mutated_uv_pixel_positions
        )

        return mutated_images, mutated_uv_pixel_positions


def flip_vertical(images, uv_pixel_positions):
    """
    Fip the images and the pixel positions vertically (flip up/down)

    See random_flip() for documentation of args and
    return types.

    """
    mutated_images = [image.flip(-2) for image in images]
    v_pixel_positions = uv_pixel_positions[1]
    mutated_v_pixel_positions = mutated_images[0].shape[1] - 1 - v_pixel_positions
    mutated_uv_pixel_positions = (uv_pixel_positions[0], mutated_v_pixel_positions)
    return mutated_images, mutated_uv_pixel_positions


def flip_horizontal(images, uv_pixel_positions):
    """
    Randomly flip the image and the pixel positions horizontall (flip
    left/right)

    See random_flip() for documentation of args and
    return types.
    """

    mutated_images = [image.flip(-1) for image in images]
    u_pixel_positions = uv_pixel_positions[0]
    mutated_u_pixel_positions = mutated_images[0].shape[2] - 1 - u_pixel_positions
    mutated_uv_pixel_positions = (mutated_u_pixel_positions, uv_pixel_positions[1])
    return mutated_images, mutated_uv_pixel_positions


def random_domain_randomize_background(image_rgb, image_mask):
    """
    Randomly call domain_randomize_background
    """
    if random.random() < 0.5:
        return image_rgb
    else:
        return domain_randomize_background(image_rgb, image_mask)


def domain_randomize_background(image_rgb, image_mask):
    """
    This function applies domain randomization to the non-masked part of the
    image.

    :param image_rgb: rgb image for which the non-masked parts of the image
                        will be domain randomized
    :type  image_rgb: torch tensor

    :param image_mask: mask of part of image to be left alone, all else will
                       dbe domain randomized
    :type image_mask: torch tensor

    :return domain_randomized_image_rgb:
    :rtype: torch tensor
    """
    # First, mask the rgb image
    three_channel_mask = image_mask.unsqueeze(0).repeat(3, 1, 1)
    image_rgb = image_rgb * three_channel_mask

    # Next, domain randomize all non-masked parts of image
    three_channel_mask_complement = (
        torch.ones_like(three_channel_mask) - three_channel_mask
    )
    random_rgb_image = get_random_image(image_rgb.shape, device=image_rgb.device)
    random_rgb_background = three_channel_mask_complement * random_rgb_image

    domain_randomized_image_rgb = image_rgb + random_rgb_background

    return domain_randomized_image_rgb


def get_random_image(shape, device="cpu"):
    """
    Expects something like shape=(3, 480, 640)

    :param shape: tuple of shape for numpy array, for example from my_array.shape
    :type shape: tuple of ints

    :return random_image:
    :rtype: np.ndarray
    """
    if random.random() < 0.5:
        rand_image = get_random_solid_color_image(shape)
    else:
        rgb1 = get_random_solid_color_image(shape, device=device)
        rgb2 = get_random_solid_color_image(shape, device=device)
        vertical = bool(np.random.uniform() > 0.5)
        rand_image = get_gradient_image(rgb1, rgb2, vertical=vertical)

    if random.random() > 0.5:
        rand_image = add_noise(rand_image)

    # Img is now in float format, so scale. Also clip noise to [0,1].
    return torch.clamp(rand_image / 255, min=0.0, max=1.0)


def get_random_rgb(device="cpu"):
    """
    :return random rgb colors, each in range 0 to 255, for example [13, 25, 255]
    :rtype: torch tensor
    """
    return torch.randint(0, 255, (3,), dtype=torch.uint8, device=device)


def get_random_solid_color_image(shape, device="cpu"):
    """
    Expects something like shape=(480,640,3)

    :return random solid color image:
    :rtype: torch tensor
    """
    color = get_random_rgb(device).unsqueeze(1).unsqueeze(2)
    return color * torch.ones(shape, dtype=torch.uint8, device=device)


def get_random_entire_image(shape, max_noise, device="cpu"):
    """
    Expects something like shape=(3, 480,640)

    Returns an array of that shape, with values in range [0..max_pixel_uint8)

    :param max_noise: maximum value in the image
    :type max_noise: int

    :return random solid color image:
    :rtype: torch tensor of specificed shape, with dtype=np.uint8
    """
    return torch.randint(0, max_noise, shape, dtype=torch.uint8, device=device)


# this gradient code roughly taken from:
# https://github.com/openai/mujoco-py/blob/master/mujoco_py/modder.py


def get_gradient_image(rgb1, rgb2, vertical):
    """
    Interpolates between two images rgb1 and rgb2

    :param rgb1, rgb2: two tensors of shape (H,W,3)

    :return interpolated image:
    :rtype: same as rgb1 and rgb2
    """
    h, w = rgb1.shape[1], rgb1.shape[2]

    if vertical:
        p = torch.tile(torch.linspace(0, 1, h)[:, None], (1, w))
    else:
        p = torch.tile(torch.linspace(0, 1, w), (h, 1))

    p = p.unsqueeze(0)

    bitmap = rgb2 * p + rgb1 * (1.0 - p)

    return bitmap


def add_noise(rgb_image):
    """
    Adds noise, and subtracts noise to the rgb_image

    :param rgb_image: image to which noise will be added
    :type rgb_image: numpy array of shape (H,W,3)

    :return image with noise:
    :rtype: same as rgb_image

    ## Note: do not need to clamp, since uint8 will just overflow -- not bad
    """
    max_noise_to_add_or_subtract = 50
    pos = get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)
    neg = get_random_entire_image(rgb_image.shape, max_noise_to_add_or_subtract)
    return rgb_image + pos - neg


def merge_images_with_occlusions(
    image_a, image_b, mask_a, mask_b, depth_a, depth_b, matches_pair_a, matches_pair_b
):
    """
    This function will take image_a and image_b and "merge" them.

    It will do this by:
    - ALWAYS TAKE image_b as foreground
    (randomly selecting either image_a or image_b to be the background)
    - using the mask for the image that is not the background, it will put the
      other image on top.
    - critically there are two separate sets of matches, one is associated with
      image_a and some other image,
        and the other is associated with image_b and some other image.
    - both of these sets of matches must be pruned for any occlusions that
      occur.

    :param image_a, image_b: the two images to merge
    :type image_a, image_b: each a tensor
    :param mask_a, mask_b: the masks for these images
    :type mask_a, mask_b: each a tensor
    :param matches_a, matches_b:
    :type matches_a, mathces_b: each a tuple of torch Tensors of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors

    :return: merged image, merged_mask, pruned_matches_a,
             pruned_associated_matches_a, pruned_matches_b,
             pruned_associated_matches_b
    :rtype: tensor, tensor, rest are same types as matches_a and matches_b

    """

    # if random.random() < 0.5:
    if True:
        foreground = "B"
        background_image, background_mask, background_matches_pair = (
            image_a,
            mask_a,
            matches_pair_a,
        )
        foreground_image, foreground_mask, foreground_matches_pair = (
            image_b,
            mask_b,
            matches_pair_b,
        )
    else:
        foreground = "A"
        background_image, background_mask, background_matches_pair = (
            image_b,
            mask_b,
            matches_pair_b,
        )
        foreground_image, foreground_mask, foreground_matches_pair = (
            image_a,
            mask_a,
            matches_pair_a,
        )

    # First, mask the foreground rgb image
    three_channel_mask = foreground_mask.unsqueeze(0).repeat(3, 1, 1)
    foreground_image = foreground_image * three_channel_mask

    # Next, zero out this portion in the background image
    three_channel_mask_complement = 1 - three_channel_mask
    background_image = three_channel_mask_complement * background_image

    # Finally, merge these two images
    merged_image = foreground_image + background_image
    merged_depth = depth_a * foreground_mask + depth_b * (1 - foreground_mask)

    # Prune occluded matches
    background_matches_pair = prune_matches_if_occluded(
        foreground_mask, background_matches_pair
    )

    if foreground == "A":
        matches_a = foreground_matches_pair[0]
        associated_matches_a = foreground_matches_pair[1]
        matches_b = background_matches_pair[0]
        associated_matches_b = background_matches_pair[1]
    elif foreground == "B":
        matches_a = background_matches_pair[0]
        associated_matches_a = background_matches_pair[1]
        matches_b = foreground_matches_pair[0]
        associated_matches_b = foreground_matches_pair[1]
    else:
        raise ValueError("Should not be here?")

    merged_masked = foreground_mask + background_mask
    # in future, could preserve identities of masks
    merged_masked = merged_masked.clip(0, 1)
    return (
        merged_image,
        merged_masked,
        merged_depth,
        matches_a,
        associated_matches_a,
        matches_b,
        associated_matches_b,
    )


def prune_matches_if_occluded(foreground_mask, background_matches_pair):
    """
    Checks if any of the matches have been occluded.

    If yes, prunes them from the list of matches.

    NOTE:
    - background_matches is a tuple
    - the first element of the tuple HAS to be the one that we are actually
      checking for occlusions
    - the second element of the tuple must also get pruned

    :param foreground_mask: The mask of the foreground image
    :type foreground_mask: tensor of shape (H,W)
    :param background_matches: a tuple of torch Tensors, each of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors
    """

    background_matches_a = background_matches_pair[0]
    background_matches_b = background_matches_pair[1]

    idxs_to_keep = []

    # this is slow but works
    for i in range(len(background_matches_a[0])):
        u = background_matches_a[0][i]
        v = background_matches_a[1][i]

        if foreground_mask[v, u] == 0:
            idxs_to_keep.append(i)

    if len(idxs_to_keep) == 0:
        return (None, None)

    idxs_to_keep = torch.LongTensor(idxs_to_keep)
    background_matches_a = (
        torch.index_select(background_matches_a[0], 0, idxs_to_keep),
        torch.index_select(background_matches_a[1], 0, idxs_to_keep),
    )
    background_matches_b = (
        torch.index_select(background_matches_b[0], 0, idxs_to_keep),
        torch.index_select(background_matches_b[1], 0, idxs_to_keep),
    )

    return (background_matches_a, background_matches_b)


def merge_matches(matches_one, matches_two):
    """
    :param matches_one, matches_two: each a tuple of torch Tensors, each of length n, i.e:

        (u_pixel_positions, v_pixel_positions)

        Where each of the elements of the tuple are torch Tensors of length n

        Note: only support torch.LongTensors
    """
    concatenated_u = torch.cat((matches_one[0], matches_two[0]))
    concatenated_v = torch.cat((matches_one[1], matches_two[1]))
    return (concatenated_u, concatenated_v)


def random_apply_random_crop(
    images_a, images_b, uv_a, uv_b, crop_size, sample_crop_size
):
    if random.random() < 0.5:
        images_a, uv_a, uv_b = random_crop(
            images_a, uv_a, uv_b, crop_size, sample_crop_size
        )

    if random.random() < 0.5:
        images_b, uv_b, uv_a = random_crop(
            images_b, uv_b, uv_a, crop_size, sample_crop_size
        )

    return images_a, images_b, uv_a, uv_b


# Cut out a random patch from the original image, eg. of size 215px for 256px
# image size, then randomly place that inside the orignal size, pad with zeros.
def random_crop(images_a, uv_a, uv_b, crop_size=(215, 215), sample_crop_size=False):
    h, w = images_a[0].shape[-2:]

    if sample_crop_size:
        crop_size = (random.randint(crop_size[0], h), random.randint(crop_size[1], w))

    crop_max_h = h - crop_size[0]
    crop_max_w = w - crop_size[1]

    hc = random.randint(0, crop_max_h)
    wc = random.randint(0, crop_max_w)

    cropped_images = [
        i[..., hc : hc + crop_size[0], wc : wc + crop_size[1]] for i in images_a
    ]

    trans_max_h = h - crop_size[0]
    trans_max_w = w - crop_size[1]

    ht = random.randint(0, trans_max_h)
    wt = random.randint(0, trans_max_w)

    translated_crops = [torch.zeros_like(i) for i in images_a]

    for i in range(len(translated_crops)):
        translated_crops[i][..., ht : ht + crop_size[0], wt : wt + crop_size[1]] = (
            cropped_images[i]
        )

    if uv_a is not None:
        # uv_pixel_positions: right, down, so swap dims
        mask_u = (wc <= uv_a[0]) & (uv_a[0] < wc + crop_size[1])
        mask_v = (hc <= uv_a[1]) & (uv_a[1] < hc + crop_size[0])
        mask = mask_u & mask_v
        mask_cropped_pixel_pos = (uv_a[0][mask], uv_a[1][mask])

        translated_pixel_pos = (
            mask_cropped_pixel_pos[0] - wc + wt,
            mask_cropped_pixel_pos[1] - hc + ht,
        )

        filtere_uv_b = (uv_b[0][mask], uv_b[1][mask])

        if translated_pixel_pos[0].numel() == 0:
            translated_pixel_pos = None
            filtere_uv_b = None
    else:
        translated_pixel_pos = None
        filtere_uv_b = None

    return translated_crops, translated_pixel_pos, filtere_uv_b
