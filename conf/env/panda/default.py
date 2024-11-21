from omegaconf import MISSING

from tapas_gmm.env.franka import FrankaEnvironmentConfig

franka_env_config = FrankaEnvironmentConfig(
    task=MISSING,
    cameras=("wrist",),
    teleop=True,
    eval=True,
    image_size=(480, 640),
    image_crop=None,
    static=False,
    headless=False,
    scale_action=True,
    delay_gripper=True,
    gripper_plot=False,
    physical_cameras={
        "wrist": ("/camera_wrist_depth_optical_frame", "934222071497"),
    },
)
