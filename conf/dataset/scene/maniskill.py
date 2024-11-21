from omegaconf import MISSING

from tapas_gmm.dataset.scene import SceneDatasetConfig

scene_dataset_config = SceneDatasetConfig(
    data_root=MISSING,
    camera_names=("wrist", "base"),
    image_size=(256, 256),
    image_crop=None,
    subsample_by_difference=False,
    subsample_to_length=None,
    object_labels=None,
    ground_truth_object_pose=True,
    shorten_cam_names=True,
    ignore_gt_labels=tuple(),
)
