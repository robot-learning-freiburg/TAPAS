import numpy as np
import torch


def scale(x, out_range=(0, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def channel_front2back(camera_obs):
    return camera_obs.permute(1, 2, 0)


# TODO: unify with channel_front2back?
def channel_front2back_batch(camera_obs):
    return camera_obs.permute(0, 2, 3, 1)


def np_channel_front2back(camera_obs):
    return np.transpose(camera_obs, (1, 2, 0))


def np_channel_back2front(camera_obs):
    return np.transpose(camera_obs, (2, 0, 1))


def channel_back2front(camera_obs):
    return camera_obs.permute(2, 0, 1)


# TODO: unify with channel_back2front
def channel_back2front_batch(camera_obs):
    return camera_obs.permute(0, 3, 1, 2)


def int_to_float_range(image):
    return image / 255


def to_float_range(image):
    raise ValueError("Verify range of image arg!")


def obs_to_plot_format(camera_obs):
    return to_float_range(channel_front2back(camera_obs))


def flatten_point_cloud(img, height=256, width=256):
    return img.reshape(height * width, 3)


def flat_points_from_indeces(points, indeces, image_width=256):
    return [points[i * image_width + j] for i, j in indeces]


def unflatten_indeces(indeces, image_width=256):
    return [(i // image_width, i % image_width) for i in indeces]


def flattened_pixel_locations_to_u_v(flat_pixel_locations, image_width=256):
    return (
        flat_pixel_locations % image_width,
        torch.div(flat_pixel_locations, image_width, rounding_mode="floor"),
    )


def uv_to_flattened_pixel_locations(uv_tuple, image_width=256):
    return uv_tuple[1] * image_width + uv_tuple[0]


def get_image_tensor_mean(img_tensor, dims_to_keep=(-1,)):
    """
    Works with arbitrary tensor dims and dims_to_keep.
    Usual use case: 3 or 4 dim tensor, keep last dim, ie. channels.
    """
    dims_to_aggr = list(range(len(img_tensor.shape)))
    for d in dims_to_keep:
        dims_to_aggr.pop(d)
    return img_tensor.mean(dim=dims_to_aggr)


def get_image_tensor_std(img_tensor, dims_to_keep=(-1,)):
    """
    Works with arbitrary tensor dims and dims_to_keep.
    Usual use case: 3 or 4 dim tensor, keep last dim, ie. channels.
    """
    dims_to_aggr = list(range(len(img_tensor.shape)))
    for d in dims_to_keep:
        dims_to_aggr.pop(d)
    return img_tensor.std(dim=dims_to_aggr)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
