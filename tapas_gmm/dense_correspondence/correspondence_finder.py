import random

import numpy as np
import torch
from loguru import logger
from scipy.spatial.distance import cdist

from tapas_gmm.utils.geometry_torch import (
    homogenous_transform_from_rot_shift,
    invert_homogenous_transform,
    invert_intrinsics,
    quaternion_to_matrix,
)
from tapas_gmm.viz.operations import flattened_pixel_locations_to_u_v

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def find_best_match(pixel_a, res_a, res_b, metric):  # ='euclidean'):
    """
    Compute the correspondences between the pixel_a location in image_a
    and image_b

    :param pixel_a: vector of (u,v) pixel coordinates
    :param res_a: array of dense descriptors res_a.shape = [H,W,D]
    :param res_b: array of dense descriptors
    :param pixel_b: Ground truth . . .
    :return: (best_match_uv, best_match_diff, norm_diffs)
    best_match_idx is again in (u,v) = (right, down) coordinates

    """

    descriptor_at_pixel = res_a[pixel_a[1], pixel_a[0]]
    height, width, D = res_a.shape

    norm_diffs = cdist(
        np.expand_dims(descriptor_at_pixel, axis=0), np.reshape(res_b, (-1, D)), metric
    )[0]

    norm_diffs = norm_diffs.reshape(height, width)

    best_match_flattened_idx = np.argmin(norm_diffs)
    best_match_xy = np.unravel_index(best_match_flattened_idx, (height, width))
    best_match_diff = norm_diffs[best_match_xy]

    best_match_uv = (best_match_xy[1], best_match_xy[0])

    return best_match_uv, best_match_diff, norm_diffs


def get_norm_diffs(pixel, descriptor, metric):  # ='euclidean'):
    descriptor_at_pixel = descriptor[pixel[1], pixel[0]]
    height, width, D = descriptor.shape

    norm_diffs = cdist(
        np.expand_dims(descriptor_at_pixel, axis=0),
        np.reshape(descriptor, (-1, D)),
        metric,
    )[0]

    norm_diffs = norm_diffs.reshape(height, width)

    return norm_diffs


def get_norm_diffs_to_ref(descriptor_at_pixel, descriptor, metric):  # ='euclidean'):
    height, width, D = descriptor.shape

    norm_diffs = cdist(
        np.expand_dims(descriptor_at_pixel, axis=0),
        np.reshape(descriptor, (-1, D)),
        metric,
    )[0]

    norm_diffs = norm_diffs.reshape(height, width)

    best_match_flattened_idx = np.argmin(norm_diffs)
    best_match_xy = np.unravel_index(best_match_flattened_idx, (height, width))

    best_match_uv = (best_match_xy[1], best_match_xy[0])

    return norm_diffs, best_match_uv


def pytorch_rand_select_pixel(width, height, num_samples=1):
    two_rand_numbers = torch.rand(2, num_samples)
    two_rand_numbers[0, :] = two_rand_numbers[0, :] * width
    two_rand_numbers[1, :] = two_rand_numbers[1, :] * height
    two_rand_ints = torch.floor(two_rand_numbers).long()
    return (two_rand_ints[0], two_rand_ints[1])


def apply_transform_torch(vec3, transform4):
    ones_row = torch.ones_like(vec3[0, :]).type(dtype_float).unsqueeze(0)
    vec4 = torch.cat((vec3, ones_row), 0)
    vec4 = transform4.mm(vec4)
    return vec4[0:3]


def random_sample_from_masked_image(img_mask, num_samples):
    """
    Samples num_samples (row, column) convention pixel locations from the masked image
    Note this is not in (u,v) format, but in same format as img_mask
    :param img_mask: numpy.ndarray
        - masked image, we will select from the non-zero entries
        - shape is H x W
    :param num_samples: int
        - number of random indices to return
    :return: List of np.array
    """
    idx_tuple = img_mask.nonzero()
    num_nonzero = len(idx_tuple[0])
    if num_nonzero == 0:
        empty_list = []
        return empty_list
    rand_inds = random.sample(range(0, num_nonzero), num_samples)

    sampled_idx_list = []
    for idx in idx_tuple:
        sampled_idx_list.append(idx[rand_inds])

    return sampled_idx_list


def random_sample_from_masked_image_torch(img_mask, num_samples):
    """

    :param img_mask: Numpy array [H,W] or torch.Tensor with shape [H,W]
    :type img_mask:
    :param num_samples: an integer
    :type num_samples:
    :return: tuple of torch.LongTensor in (u,v) format. Each torch.LongTensor has shape
    [num_samples]
    :rtype:
    """

    image_height, image_width = img_mask.shape

    if isinstance(img_mask, np.ndarray):
        img_mask_torch = torch.from_numpy(img_mask).float()
    else:
        img_mask_torch = img_mask

    # This code would randomly subsample from the mask
    mask = img_mask_torch.view(image_width * image_height, 1).squeeze(1)
    mask_indices_flat = torch.nonzero(mask)
    if len(mask_indices_flat) == 0:
        return (None, None)

    rand_numbers = torch.rand(num_samples) * len(mask_indices_flat)
    rand_indices = torch.floor(rand_numbers).long()
    uv_vec_flattened = torch.index_select(mask_indices_flat, 0, rand_indices).squeeze(1)
    uv_vec = flattened_pixel_locations_to_u_v(uv_vec_flattened, image_width)
    return uv_vec


def get_mask_center(img_mask: torch.Tensor | np.ndarray, num_samples: int = 1):
    assert num_samples == 1

    if isinstance(img_mask, np.ndarray):
        img_mask_torch = torch.from_numpy(img_mask).float()
    else:
        img_mask_torch = img_mask

    norm = img_mask_torch.sum()

    grids = [
        torch.arange(img_mask_torch.shape[1]).unsqueeze(0),
        torch.arange(img_mask_torch.shape[0]).unsqueeze(1),
    ]

    uv = [
        (img_mask * grids[d]).sum().unsqueeze(0) / norm
        for d in range(img_mask_torch.ndim)
    ]

    return tuple(x.to(dtype=torch.int32) for x in uv)


def get_masked_avg_descriptor(mask: torch.Tensor, descriptor: torch.Tensor):
    assert mask.ndim == 2
    assert descriptor.ndim == 3
    assert mask.shape == descriptor.shape[1:]

    mask = mask.unsqueeze(0).expand_as(descriptor)
    return (descriptor * mask).sum(dim=(1, 2)) / mask.sum()


def create_non_correspondences(
    uv_b_matches, img_b_shape, num_non_matches_per_match=100, img_b_mask=None
):
    """
    Takes in pixel matches (uv_b_matches) that correspond to matches in another
    image, and generates non-matches by just sampling in image space.

    Optionally, the non-matches can be sampled from a mask for image b.

    Returns non-matches as pixel positions in image b.

    Please see 'coordinate_conventions.md' documentation for an explanation of
    pixel coordinate conventions.

    ## Note that arg uv_b_matches are the outputs of
    batch_find_pixel_correspondences()

    :param uv_b_matches: tuple of torch.FloatTensors, where each FloatTensor
        is length n, i.e.: (torch.FloatTensor, torch.FloatTensor)

    :param img_b_shape: tuple of (H,W) which is the shape of the image

    (optional)
    :param num_non_matches_per_match: int

    (optional)
    :param img_b_mask: torch.FloatTensor (can be cuda or not)
        - masked image, we will select from the non-zero entries
        - shape is H x W

    :return: tuple of torch.FloatTensors, i.e.
        (torch.FloatTensor, torch.FloatTensor).
        - The first element of the tuple is all "u" pixel positions, and the
          right element of the tuple is all "v" positions
        - Each torch.FloatTensor is of shape
          torch.Shape([num_matches, non_matches_per_match])
        - This shape makes it so that each row of the non-matches corresponds
          to the row for the match in uv_a
    """
    image_width = img_b_shape[1]
    image_height = img_b_shape[0]

    if uv_b_matches is None or num_non_matches_per_match == 0:
        return None

    num_matches = len(uv_b_matches[0])

    def get_random_uv_b_non_matches():
        return pytorch_rand_select_pixel(
            width=image_width,
            height=image_height,
            num_samples=num_matches * num_non_matches_per_match,
        )

    if img_b_mask is not None:
        img_b_mask_flat = img_b_mask.reshape(-1, 1).squeeze(1)
        mask_b_indices_flat = torch.nonzero(img_b_mask_flat)
        if len(mask_b_indices_flat) == 0:
            logger.warning("warning, empty mask b")
            uv_b_non_matches = get_random_uv_b_non_matches()
        else:
            num_samples = num_matches * num_non_matches_per_match
            rand_numbers_b = torch.rand(num_samples) * len(mask_b_indices_flat)
            rand_indices_b = torch.floor(rand_numbers_b).long()
            randomized_mask_b_indices_flat = torch.index_select(
                mask_b_indices_flat, 0, rand_indices_b
            ).squeeze(1)
            uv_b_non_matches = flattened_pixel_locations_to_u_v(
                randomized_mask_b_indices_flat, image_width
            )
    else:
        uv_b_non_matches = get_random_uv_b_non_matches()

    # for each in uv_a, we want non-matches
    # first just randomly sample "non_matches"
    # we will later move random samples that were too close to being matches
    uv_b_non_matches = (
        uv_b_non_matches[0].view(num_matches, num_non_matches_per_match),
        uv_b_non_matches[1].view(num_matches, num_non_matches_per_match),
    )

    # uv_b_matches can now be used to make sure no "non_matches" are too close
    # to preserve tensor size, rather than pruning, we can perturb these in pixel space
    copied_uv_b_matches_0 = torch.t(
        uv_b_matches[0].repeat(num_non_matches_per_match, 1)
    )
    copied_uv_b_matches_1 = torch.t(
        uv_b_matches[1].repeat(num_non_matches_per_match, 1)
    )

    diffs_0 = copied_uv_b_matches_0 - uv_b_non_matches[0].type(dtype_float)
    diffs_1 = copied_uv_b_matches_1 - uv_b_non_matches[1].type(dtype_float)

    diffs_0_flattened = diffs_0.reshape(-1, 1)
    diffs_1_flattened = diffs_1.reshape(-1, 1)

    diffs_0_flattened = torch.abs(diffs_0_flattened).squeeze(1)
    diffs_1_flattened = torch.abs(diffs_1_flattened).squeeze(1)

    need_to_be_perturbed = torch.zeros_like(diffs_0_flattened)
    ones = torch.zeros_like(diffs_0_flattened)
    num_pixels_too_close = 1.0
    threshold = torch.ones_like(diffs_0_flattened) * num_pixels_too_close

    # determine which pixels are too close to being matches
    need_to_be_perturbed = torch.where(
        diffs_0_flattened < threshold, ones, need_to_be_perturbed
    )
    need_to_be_perturbed = torch.where(
        diffs_1_flattened < threshold, ones, need_to_be_perturbed
    )

    minimal_perturb = num_pixels_too_close / 2
    minimal_perturb_vector = (torch.rand(len(need_to_be_perturbed)) * 2).floor() * (
        minimal_perturb * 2
    ) - minimal_perturb
    std_dev = 10
    random_vector = (
        torch.randn(len(need_to_be_perturbed)) * std_dev + minimal_perturb_vector
    )
    perturb_vector = need_to_be_perturbed * random_vector

    uv_b_non_matches_0_flat = (
        uv_b_non_matches[0].view(-1, 1).type(dtype_float).squeeze(1)
    )
    uv_b_non_matches_1_flat = (
        uv_b_non_matches[1].view(-1, 1).type(dtype_float).squeeze(1)
    )

    uv_b_non_matches_0_flat = uv_b_non_matches_0_flat + perturb_vector
    uv_b_non_matches_1_flat = uv_b_non_matches_1_flat + perturb_vector

    # now just need to wrap around any that went out of bounds

    # handle wrapping in width
    lower_bound = 0.0
    upper_bound = image_width * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_0_flat) * upper_bound

    uv_b_non_matches_0_flat = torch.where(
        uv_b_non_matches_0_flat > upper_bound_vec,
        uv_b_non_matches_0_flat - upper_bound_vec,
        uv_b_non_matches_0_flat,
    )

    uv_b_non_matches_0_flat = torch.where(
        uv_b_non_matches_0_flat < lower_bound_vec,
        uv_b_non_matches_0_flat + upper_bound_vec,
        uv_b_non_matches_0_flat,
    )

    # handle wrapping in height
    lower_bound = 0.0
    upper_bound = image_height * 1.0 - 1
    lower_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * lower_bound
    upper_bound_vec = torch.ones_like(uv_b_non_matches_1_flat) * upper_bound

    uv_b_non_matches_1_flat = torch.where(
        uv_b_non_matches_1_flat > upper_bound_vec,
        uv_b_non_matches_1_flat - upper_bound_vec,
        uv_b_non_matches_1_flat,
    )

    uv_b_non_matches_1_flat = torch.where(
        uv_b_non_matches_1_flat < lower_bound_vec,
        uv_b_non_matches_1_flat + upper_bound_vec,
        uv_b_non_matches_1_flat,
    )

    return (
        uv_b_non_matches_0_flat.view(num_matches, num_non_matches_per_match),
        uv_b_non_matches_1_flat.view(num_matches, num_non_matches_per_match),
    )


# TODO: make this (and the process that calls it) truely batched, ie with
# multiple image pairs. Then it should make sense to run it on GPU.
# Optionally, uv_a specifies the pixels in img_a for which to find matches
# If uv_a is not set, then random correspondences are attempted to be found
def batch_find_pixel_correspondences(
    img_a_depth,
    img_a_pose,
    img_b_depth,
    img_b_pose,
    uv_a=None,
    num_attempts=20,
    device="cpu",
    img_a_mask=None,
    K_a=None,
    K_b=None,
    obj_pose_a=None,
    obj_pose_b=None,
    object_no=None,
):
    """
    Computes pixel correspondences in batch

    :param img_a_depth: depth image for image a
    :type  img_a_depth: torch 2d tensor (H x W)
    --
    :param img_a_pose:  pose for image a, in right-down-forward optical frame
    :type  img_a_pose:  torch 2d tensor, 4 x 4 (homogeneous transform)
    --
    :param img_b_depth: depth image for image b
    :type  img_b_depth: torch 2d tensor (H x W)
    --
    :param img_b_pose:  pose for image a, in right-down-forward optical frame
    :type  img_b_pose:  torch 2d tensor, 4 x 4 (homogeneous transform)
    --
    :param uv_a:        optional arg, a tuple of (u,v) pixel positions for
                        which to find matches
    :type  uv_a:        each element of tuple is either an int, or a list-like
                        (castable to torch.LongTensor)
    --
    :param num_attempts: if random sampling, how many pixels will be _attempted
                            to find matches for.  Note that this is not the
                            same as asking for a specific number of matches,
                            since many attempted matches will either be
                            occluded or outside of field-of-view.
    :type  num_attempts: int
    --
    :param device:      torch device
    :type  device:      string
    --
    :param img_a_mask:  optional arg, an image where each nonzero pixel will
                        be used as a mask
    :type  img_a_mask:  torch tensor, of shape (H, W)
    --
    :param K:           camera intrinsics matrix
    :type  K:           torch tensor, of shape (3, 3)
    --
    :return:            "Tuple of tuples", i.e. pixel position tuples for image
                        a and image b (uv_a, uv_b).
                        Each of these is a tuple of pixel positions
    :rtype:             Each of uv_a is a tuple of torch.FloatTensors
    """
    assert img_a_depth.shape == img_b_depth.shape
    image_width = img_a_depth.shape[1]
    image_height = img_b_depth.shape[0]

    if uv_a is None:
        uv_a = pytorch_rand_select_pixel(
            width=image_width, height=image_height, num_samples=num_attempts
        )
    else:
        # uv_a = (torch.LongTensor([uv_a[0]]).type(
        #     dtype_long), torch.LongTensor([uv_a[1]]).type(dtype_long))
        uv_a = (uv_a[0].long(), uv_a[1].long())
        num_attempts = 1

    if img_a_mask is None:
        # uv_a_vec = (torch.ones(num_attempts).type(dtype_long)
        #             * uv_a[0], torch.ones(num_attempts).type(dtype_long)*uv_a[1])
        uv_a_vec = (
            torch.ones(num_attempts, dtype=torch.long, device=device) * uv_a[0],
            torch.ones(num_attempts, dtype=torch.long, device=device) * uv_a[1],
        )
        uv_a_vec_flattened = uv_a_vec[1] * image_width + uv_a_vec[0]
    else:
        #  img_a_mask = torch.from_numpy(img_a_mask).type(dtype_float)

        uv_a_vec = random_sample_from_masked_image_torch(
            img_a_mask, num_samples=num_attempts
        )
        if uv_a_vec[0] is None:
            return (None, None)

        uv_a_vec_flattened = uv_a_vec[1] * image_width + uv_a_vec[0]

    K_inv = invert_intrinsics(K_a)

    img_a_depth = torch.squeeze(img_a_depth, 0).view(-1, 1)
    depth_vec = torch.index_select(img_a_depth, 0, uv_a_vec_flattened) * 1.0
    depth_vec = depth_vec.squeeze(1)

    # Prune based on
    # Case 1: depth is zero (for this data, this means no-return)
    nonzero_indices = torch.nonzero(depth_vec)
    if nonzero_indices.numel() == 0:
        return (None, None)
    nonzero_indices = nonzero_indices.squeeze(1)
    depth_vec = torch.index_select(depth_vec, 0, nonzero_indices)

    # prune u_vec and v_vec, then multiply by already pruned depth_vec
    u_a_pruned = torch.index_select(uv_a_vec[0], 0, nonzero_indices)
    u_vec = u_a_pruned.float() * depth_vec

    v_a_pruned = torch.index_select(uv_a_vec[1], 0, nonzero_indices)
    v_vec = v_a_pruned.float() * depth_vec

    z_vec = depth_vec

    full_vec = torch.stack((u_vec, v_vec, z_vec))

    point_camera_frame_rdf_vec = K_inv.mm(full_vec)

    point_world_frame_rdf_vec = apply_transform_torch(
        point_camera_frame_rdf_vec, img_a_pose
    )

    # apply relative object transform if available
    if obj_pose_a is not None and obj_pose_b is not None:
        # TODO: index the pose matrix outside and only pass the right one in?
        ref_shift = obj_pose_a[object_no, :3]
        ref_rot = obj_pose_a[object_no, 3:7]
        cur_shift = obj_pose_b[object_no, :3]
        cur_rot = obj_pose_b[object_no, 3:7]

        cur_hom = homogenous_transform_from_rot_shift(
            quaternion_to_matrix(cur_rot), cur_shift
        )
        ref_hom = homogenous_transform_from_rot_shift(
            quaternion_to_matrix(ref_rot), ref_shift
        )

        ref_to_cur = torch.matmul(cur_hom, invert_homogenous_transform(ref_hom))

        point_world_frame_rdf_vec_pose_corrected = apply_transform_torch(
            point_world_frame_rdf_vec, ref_to_cur
        )
    else:
        point_world_frame_rdf_vec_pose_corrected = point_world_frame_rdf_vec

    point_camera_2_frame_rdf_vec = apply_transform_torch(
        point_world_frame_rdf_vec_pose_corrected,
        invert_homogenous_transform(img_b_pose),
    )
    vec2_vec = K_b.mm(point_camera_2_frame_rdf_vec)

    u2_vec = vec2_vec[0] / vec2_vec[2]
    v2_vec = vec2_vec[1] / vec2_vec[2]

    z2_vec = vec2_vec[2]

    # Prune based on
    # Case 2: the pixels projected into image b are outside FOV
    # u2_vec bounds should be: 0, image_width
    # v2_vec bounds should be: 0, image_height

    # do u2-based pruning
    u2_vec_lower_bound = 0.0
    epsilon = 1e-3
    # careful, needs to be epsilon less!!
    u2_vec_upper_bound = image_width * 1.0 - epsilon
    lower_bound_vec = torch.ones_like(u2_vec) * u2_vec_lower_bound
    upper_bound_vec = torch.ones_like(u2_vec) * u2_vec_upper_bound
    zeros_vec = torch.zeros_like(u2_vec)

    u2_vec = torch.where(u2_vec < lower_bound_vec, zeros_vec, u2_vec)
    u2_vec = torch.where(u2_vec > upper_bound_vec, zeros_vec, u2_vec)
    in_bound_indices = torch.nonzero(u2_vec)
    if in_bound_indices.numel() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(
        u_a_pruned, 0, in_bound_indices
    )  # also prune from first list
    v_a_pruned = torch.index_select(
        v_a_pruned, 0, in_bound_indices
    )  # also prune from first list

    # do v2-based pruning
    v2_vec_lower_bound = 0.0
    v2_vec_upper_bound = image_height * 1.0 - epsilon
    lower_bound_vec = torch.ones_like(v2_vec) * v2_vec_lower_bound
    upper_bound_vec = torch.ones_like(v2_vec) * v2_vec_upper_bound
    zeros_vec = torch.zeros_like(v2_vec)

    v2_vec = torch.where(v2_vec < lower_bound_vec, zeros_vec, v2_vec)
    v2_vec = torch.where(v2_vec > upper_bound_vec, zeros_vec, v2_vec)
    in_bound_indices = torch.nonzero(v2_vec)
    if in_bound_indices.numel() == 0:
        return (None, None)
    in_bound_indices = in_bound_indices.squeeze(1)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, in_bound_indices)
    v2_vec = torch.index_select(v2_vec, 0, in_bound_indices)
    z2_vec = torch.index_select(z2_vec, 0, in_bound_indices)
    u_a_pruned = torch.index_select(
        u_a_pruned, 0, in_bound_indices
    )  # also prune from first list
    v_a_pruned = torch.index_select(
        v_a_pruned, 0, in_bound_indices
    )  # also prune from first list

    # Prune based on
    # Case 3: the pixels in image b are occluded, OR there is no depth return
    # in image b so we aren't sure

    img_b_depth = torch.squeeze(img_b_depth, 0).view(-1, 1)

    # simply round to int -- good enough # TODO: what's that about?
    uv_b_vec_flattened = v2_vec.long() * image_width + u2_vec.long()
    # occlusion check for smooth surfaces

    depth2_vec = torch.index_select(img_b_depth, 0, uv_b_vec_flattened) * 1.0
    depth2_vec = depth2_vec.squeeze(1)

    # occlusion margin, in meters
    occlusion_margin = 0.003
    z2_vec = z2_vec - occlusion_margin
    zeros_vec = torch.zeros_like(depth2_vec)

    # to be careful, prune any negative depths
    depth2_vec = torch.where(depth2_vec < zeros_vec, zeros_vec, depth2_vec)
    depth2_vec = torch.where(
        depth2_vec < z2_vec, zeros_vec, depth2_vec
    )  # prune occlusions
    non_occluded_indices = torch.nonzero(depth2_vec)
    if non_occluded_indices.numel() == 0:
        return (None, None)
    non_occluded_indices = non_occluded_indices.squeeze(1)
    depth2_vec = torch.index_select(depth2_vec, 0, non_occluded_indices)

    # apply pruning
    u2_vec = torch.index_select(u2_vec, 0, non_occluded_indices)
    v2_vec = torch.index_select(v2_vec, 0, non_occluded_indices)
    u_a_pruned = torch.index_select(
        u_a_pruned, 0, non_occluded_indices
    )  # also prune from first list
    v_a_pruned = torch.index_select(
        v_a_pruned, 0, non_occluded_indices
    )  # also prune from first list

    uv_b_vec = (u2_vec, v2_vec)
    uv_a_vec = (u_a_pruned, v_a_pruned)

    return (uv_a_vec, uv_b_vec)
