from omegaconf import MISSING

from tapas_gmm.env.franka import FrankaEnvironmentConfig

franka_env_config = FrankaEnvironmentConfig(
    task=MISSING,
    cameras=("wrist",),
    teleop=False,
    tele_grasp=True,
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
    neutral_joints=(
        -1.5795,
        -1.2399,
        0.5783,
        -2.3193,
        0.9019,
        1.4223,
        -0.1759,
    ),
)
