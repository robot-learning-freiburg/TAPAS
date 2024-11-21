import numpy as np
import open3d
from pyrep.objects.vision_sensor import VisionSensor

from tapas_gmm.env import Environment

project = VisionSensor.pointcloud_from_depth_and_camera_params


coordinate_boxes = {
    Environment.MANISKILL: np.array(
        [[-0.75, 0.75], [-0.75, 0.75], [0.0, 2.0]]  # x  # TODO: set  # y
    ),  # z
    Environment.PANDA: np.array([[-0.60, 1.35], [-0.85, 0.85], [0.5, 1.9]]),
    Environment.RLBENCH: np.array([[-0.15, 1.35], [-0.85, 0.85], [0.76, 1.75]]),
}

gripper_dists = {
    Environment.MANISKILL: 0.05,  # TODO: set
    Environment.PANDA: 0.05,
    Environment.RLBENCH: 0.1,
}


def cut_volume_with_box(vol_bnd: np.ndarray, box: np.ndarray) -> np.ndarray:
    refined = np.zeros_like(vol_bnd)
    refined[:, 0] = np.maximum(vol_bnd[:, 0], box[:, 0])
    refined[:, 1] = np.minimum(vol_bnd[:, 1], box[:, 1])

    return refined


def filter_background(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    coordinate_box: np.ndarray,
) -> np.ndarray:
    """
    Project depth images into world coordinates and zero-out the depth image
    where the point is outside the given coordinate_box.

    Parameters
    ----------
    depth : np.array(N, H, W, 1)
    extrinsics : np.array(N, 4, 4)
    intrinsics : np.array(N, 3, 3)
    coordinate_box : np.array(3, 2)

    Returns
    -------
    type
        The filtered depth map. Filtered values are zeroed out.
    """
    filtered_depth = np.empty_like(depth)

    for i in range(depth.shape[0]):
        pointcloud = project(depth[i], extrinsics[i], intrinsics[i])

        shape = pointcloud.shape
        pointcloud = pointcloud.reshape(-1, shape[-1])
        lower = coordinate_box[:, 0]
        upper = coordinate_box[:, 1]

        point_is_outside = np.any(
            np.logical_or(lower >= pointcloud, pointcloud >= upper), axis=1
        )

        point_is_outside = point_is_outside.reshape(shape[0], shape[1])

        filtered_depth[i] = np.where(point_is_outside, 0, depth[i])

    return filtered_depth


def get_plane_indeces(raw_pointcloud, ransac_n=3, num_iterations=1000, threshold=0.01):
    """
    Detect points from pointcloud that are within a given threshold of a plane.
    Used to filter out the plane of the table.
    Parameters
    ----------
    raw_pointcloud : np.array(N, 3)
    ransac_n : int
        Number of points to use for RANSAC.
    num_iterations : int
        Number of iterations for RANSAC.
    threshold : float
        Threshold for plane distance.
    Returns
    -------
    np.array(N')
        Indeces of the plane points
    """
    # Fit plane
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(raw_pointcloud)
    # open3d.visualization.draw_geometries([point_cloud])
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=threshold, ransac_n=ransac_n, num_iterations=num_iterations
    )

    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # plane_cloud = point_cloud.select_by_index(inliers)
    # plane_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # plane_normal = np.array([a, b, c])
    # plane_normal /= np.linalg.norm(plane_normal)

    # obb = plane_cloud.get_oriented_bounding_box()
    # open3d.visualization.draw_geometries([obb, plane_cloud, outlier_cloud])

    # return np.asarray(outlier_cloud.points), inliers
    return inliers


def filter_plane_from_mesh_and_pointcloud(vertices, faces):
    """
    Remove plane representing the table from the mesh.
    First fits plane to pointcloud and then removes the plane's faces from
    the mesh.
    Parameters
    ----------
    vertices : np.array(N, 3) (3D coordinates of the vertices)
    faces : np.array(M, 3) (Triangle faces indexing the vertices)
    Returns
    -------
    np.array(N', 3)
        The filtered vertices.
    np.array(M', 3)
        The filtered faces.
    """
    plane_indeces = get_plane_indeces(vertices)

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(faces)
    mesh.remove_vertices_by_index(plane_indeces)

    # open3d.visualization.draw_geometries([mesh])

    filtered_vertices = np.asarray(mesh.vertices)
    filtered_faces = np.asarray(mesh.triangles)

    return filtered_vertices, filtered_faces


def filter_gripper(depth: np.ndarray, gripper_dist: float) -> np.ndarray:
    """
    Remove gripper artifacts via filtering all points below the defined depth
    threshold.

    Parameters
    ----------
    depth : np.array(N, H, W, 1)
    extrinsics : np.array(N, 4, 4)
    intrinsics : np.array(N, 3, 3)

    Returns
    -------
    type
        The filtered depth map. Filtered values are zeroed out.
    """
    filtered_depth = np.empty_like(depth)

    for i in range(depth.shape[0]):
        point_is_gripper = depth[i] <= gripper_dist

        filtered_depth[i] = np.where(point_is_gripper, 0, depth[i])

    return filtered_depth
