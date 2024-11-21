import numpy as np
import torch
from numba import prange
from numpy.linalg import inv

from tapas_gmm.utils.select_gpu import device


# TODO: probably have something like this already somewhere.
def pinhole_projection_image_to_world_coordinates(uv, z, K, camera_to_world):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :param camera_to_world: 4 x 4 homogeneous transform
    :type camera_to_world: numpy array
    :return: (x,y,z) in world
    :rtype: numpy.array size (3,)
    """

    pos_in_camera_frame = pinhole_projection_image_to_camera_coordinates(uv, z, K)
    pos_in_camera_frame_homog = np.append(pos_in_camera_frame, 1)
    pos_in_world_homog = camera_to_world.dot(pos_in_camera_frame_homog)
    return pos_in_world_homog[:3]


def pinhole_projection_image_to_camera_coordinates(uv, z, K):
    """
    Takes a (u,v) pixel location to it's 3D location in camera frame.
    See https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for a detailed explanation.

    :param uv: pixel location in image
    :type uv:
    :param z: depth, in camera frame
    :type z: float
    :param K: 3 x 3 camera intrinsics matrix
    :type K: numpy.ndarray
    :return: (x,y,z) in camera frame
    :rtype: numpy.array size (3,)
    """
    u_v_1 = np.array([uv[0], uv[1], 1])
    K_inv = inv(K)

    pos = z * K_inv.dot(u_v_1)
    return pos


def hard_pixels_to_3D_world(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W
    camera_to_world,  # N, 4, 4 , on cpu
    K,  # N, 3, 3, on cpu
    img_width,
    img_height,
):
    N = y_vision.shape[0]
    k = int(y_vision.shape[1] / 2)

    # print("y vision", y_vision.shape)
    # print(depth.shape)
    # print(camera_to_world.shape)
    # print(K.shape)
    # print(img_width, img_height)

    y_vision_3d = torch.zeros(N, k * 3).to(device)

    for i in range(N):
        for j in range(k):
            u_normalized_coordinate = y_vision[i, j]
            v_normalized_coordinate = y_vision[i, k + j]
            u_pixel_coordinate = (
                ((u_normalized_coordinate / 2.0 + 0.5) * img_width)
                .long()
                .detach()
                .cpu()
                .numpy()
            )
            v_pixel_coordinate = (
                ((v_normalized_coordinate / 2.0 + 0.5) * img_height)
                .long()
                .detach()
                .cpu()
                .numpy()
            )
            uv = (u_pixel_coordinate, v_pixel_coordinate)

            z = depth[i, v_pixel_coordinate, u_pixel_coordinate].cpu().numpy()

            K_one = K[i, :, :].cpu().numpy()
            camera_to_world_one = camera_to_world[i, :, :].cpu().numpy()

            # print(uv, z, K_one, camera_to_world_one)
            point_3d_world = pinhole_projection_image_to_world_coordinates(
                uv, z, K_one, camera_to_world_one
            )
            y_vision_3d[i, j + 0 * k] = point_3d_world[0]
            y_vision_3d[i, j + 1 * k] = point_3d_world[1]
            y_vision_3d[i, j + 2 * k] = point_3d_world[2]

    return y_vision_3d


def raw_pixels_to_3D_world(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W, on cpu
    camera_to_world,  # N, 4, 4 , on cpu
    K,  # N, 3, 3, on cpu
):
    """
    Like the function above, but takes raw pixels, ie. in [0,H/W]), not [-1.1].
    """
    N = y_vision.shape[0]
    k = int(y_vision.shape[1] / 2)
    y_vision_3d = torch.zeros(N, k * 3).to(device)

    # print("y vision", y_vision)
    # print(depth)
    # print(camera_to_world)
    # print(K)

    for i in range(N):
        for j in range(k):
            u_pixel_coordinate = y_vision[i, j].long().detach().cpu().numpy()
            v_pixel_coordinate = y_vision[i, k + j].long().detach().cpu().numpy()

            uv = (u_pixel_coordinate, v_pixel_coordinate)

            z = depth[i, v_pixel_coordinate, u_pixel_coordinate].cpu().numpy()

            K_one = K[i, :, :].cpu().numpy()
            camera_to_world_one = camera_to_world[i, :, :].cpu().numpy()

            # print(uv, z, K_one, camera_to_world_one)
            point_3d_world = pinhole_projection_image_to_world_coordinates(
                uv, z, K_one, camera_to_world_one
            )
            y_vision_3d[i, j + 0 * k] = point_3d_world[0]
            y_vision_3d[i, j + 1 * k] = point_3d_world[1]
            y_vision_3d[i, j + 2 * k] = point_3d_world[2]

    return y_vision_3d


def raw_pixels_to_camera_frame(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W, on cpu
    K,  # N, 3, 3, on cpu
):
    N = y_vision.shape[0]
    k = int(y_vision.shape[1] / 2)
    y_vision_3d = torch.zeros(N, k, 3).to(device)

    # print("y vision", y_vision)
    # print(depth)
    # print(camera_to_world)
    # print(K)

    for i in range(N):
        for j in range(k):
            u_pixel_coordinate = y_vision[i, j].long().detach().cpu().numpy()
            v_pixel_coordinate = y_vision[i, k + j].long().detach().cpu().numpy()

            uv = (u_pixel_coordinate, v_pixel_coordinate)

            z = depth[i, v_pixel_coordinate, u_pixel_coordinate].cpu().numpy()

            K_one = K[i, :, :].cpu().numpy()

            # print(uv, z, K_one, camera_to_world_one)
            point_3d_cam = pinhole_projection_image_to_camera_coordinates(uv, z, K_one)
            y_vision_3d[i, j, 0] = point_3d_cam[0]
            y_vision_3d[i, j, 1] = point_3d_cam[1]
            y_vision_3d[i, j, 2] = point_3d_cam[2]

    return y_vision_3d


def compute_expected_z(softmax_activations, depth):
    """
    softmax_activations: N, nm, H, W
    depth: N, C=1, H, W
    """
    N = softmax_activations.shape[0]
    num_matches = softmax_activations.shape[1]
    expected_z = torch.zeros(N, num_matches).to(device)

    downsampled_depth = torch.nn.functional.interpolate(
        depth, scale_factor=1.0 / 8, mode="bilinear", align_corners=True
    )
    # N, C=1, H/8, W/8

    for i in range(N):
        one_expected_z = torch.sum(
            (softmax_activations[i] * downsampled_depth[i, 0]).unsqueeze(0), dim=(2, 3)
        )  # 1, nm
        expected_z[i, :] = one_expected_z

    return expected_z


def soft_pixels_to_3D_world(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    sm_activations,  # N, k, H, W (small H, W size)
    depth,  # N, H, W, on cpu (full H, W size)
    camera_to_world,  # N, 4, 4 , on cpu
    K,  # N, 3, 3, on cpu
    img_width,
    img_height,
):
    N = y_vision.shape[0]
    k = int(y_vision.shape[1] / 2)

    y_vision_3d = torch.zeros(N, k * 3).to(device)

    for i in range(N):
        for j in range(k):
            u_normalized_coordinate = y_vision[i, j]
            v_normalized_coordinate = y_vision[i, k + j]
            u_pixel_coordinate = (
                ((u_normalized_coordinate / 2.0 + 0.5) * img_width)
                .long()
                .detach()
                .cpu()
                .numpy()
            )
            v_pixel_coordinate = (
                ((v_normalized_coordinate / 2.0 + 0.5) * img_height)
                .long()
                .detach()
                .cpu()
                .numpy()
            )
            uv = (u_pixel_coordinate, v_pixel_coordinate)

            this_depth = (
                (depth[i]).unsqueeze(0).unsqueeze(1).to(device)
            )  # convert to meters, from millimeters
            z = (
                compute_expected_z(
                    sm_activations[i, j].unsqueeze(0).unsqueeze(0), this_depth
                )
                .cpu()
                .detach()
                .numpy()
            )

            K_one = K[i, :, :].cpu().numpy()
            camera_to_world_one = camera_to_world[i, :, :].cpu().numpy()

            # HACK TO MAKE THIS IN CAMERA FRAME!
            camera_to_world_one = np.eye(4)

            # print(uv, z, K_one, camera_to_world_one)
            point_3d_world = pinhole_projection_image_to_world_coordinates(
                uv, z, K_one, camera_to_world_one
            )
            y_vision_3d[i, j + 0 * k] = point_3d_world[0]
            y_vision_3d[i, j + 1 * k] = point_3d_world[1]
            y_vision_3d[i, j + 2 * k] = point_3d_world[2]

    return y_vision_3d


def rigid_transform(xyz, transform):
    """Applies a rigid transform (or series of them) to an (N, 3) pointcloud."""
    xyz_h = np.concatenate(
        [xyz, np.ones((*xyz.shape[:-1], 1), dtype=np.float32)], axis=-1
    )
    xyz_t_h = np.dot(transform, xyz_h.T).swapaxes(-1, -2)
    return xyz_t_h[..., :, :3]


def cam2pix(cam_pts, intr_batch):
    """Convert camera coordinates to pixel coordinates."""
    B, n, _ = cam_pts.shape
    pix = np.empty((B, n, 2), dtype=np.int64)
    for b in range(B):
        intr = intr_batch[b].astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        for i in prange(n):
            pix[b, i, 0] = int(
                np.round((cam_pts[b, i, 0] * fx / cam_pts[b, i, 2]) + cx)
            )
            pix[b, i, 1] = int(
                np.round((cam_pts[b, i, 1] * fy / cam_pts[b, i, 2]) + cy)
            )
    return pix


def project_onto_cam(point_cloud, depth_im, cam_intr, cam_pose):
    """
    Modified from what I wrote in tsdf fusion for 3D point cloud tensors and
    3D cam_pose tensors. Ie. both are batches.
    """
    pcs = point_cloud.shape  # (B, Np, 3)
    cam_pts = np.empty_like(point_cloud)
    assert len(pcs) == 3  # should be batched
    for b in range(pcs[0]):
        cam_pts[b] = rigid_transform(point_cloud[b], np.linalg.inv(cam_pose[b]))
    pix_z = cam_pts[:, :, 2]

    pix = cam2pix(cam_pts, cam_intr)
    pix_x, pix_y = pix[:, :, 0], pix[:, :, 1]

    B, im_h, im_w = depth_im.shape
    # Eliminate pixels outside view frustum. Ie map them to zero here.
    # TODO: instead clip to image border to allow easier navigation?
    valid_pix = np.logical_and(
        pix_x >= 0,
        np.logical_and(
            pix_x < im_w,
            np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
        ),
    )

    final = np.where(np.repeat(valid_pix[:, :, np.newaxis], 2, axis=2), pix, 0)

    return final


def append_depth_to_uv(
    y_vision,  # N, 2*k, where x features are stacked on top of y features
    depth,  # N, H, W
    img_width,
    img_height,
):
    N = y_vision.shape[0]
    k = int(y_vision.shape[1] / 2)

    y_vision_3d = torch.zeros(N, k * 3, device=y_vision.device)

    for i in range(N):
        for j in range(k):
            u_normalized_coordinate = y_vision[i, j]
            v_normalized_coordinate = y_vision[i, k + j]
            u_pixel_coordinate = (
                ((u_normalized_coordinate / 2.0 + 0.5) * img_width).long().detach()
            )
            v_pixel_coordinate = (
                ((v_normalized_coordinate / 2.0 + 0.5) * img_height).long().detach()
            )

            z = depth[i, v_pixel_coordinate, u_pixel_coordinate]

            y_vision_3d[i, j + 0 * k] = u_pixel_coordinate
            y_vision_3d[i, j + 1 * k] = v_pixel_coordinate
            y_vision_3d[i, j + 2 * k] = z

    return y_vision_3d
