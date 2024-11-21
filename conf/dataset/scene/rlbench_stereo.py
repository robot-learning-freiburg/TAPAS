from dataclasses import dataclass

from omegaconf import MISSING

from tapas_gmm.dataset.scene import SceneDatasetConfig

scene_dataset_config = SceneDatasetConfig(
    data_root=MISSING,
    camera_names=("wrist", "front"),
    image_size=(256, 256),
    image_crop=None,
    subsample_by_difference=False,
    subsample_to_length=None,
    object_labels=None,
    ground_truth_object_pose=True,
    shorten_cam_names=True,
    ignore_gt_labels=(
        10,
        31,
        34,
        35,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        48,
        52,
        53,
        54,
        55,
        56,
        57,
        90,
        92,
        255,
        16777215,
    ),
)
