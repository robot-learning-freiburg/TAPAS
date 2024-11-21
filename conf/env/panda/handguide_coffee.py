from omegaconf import MISSING

from tapas_gmm.env.franka import FrankaEnvironmentConfig

franka_env_config = FrankaEnvironmentConfig(
    task=MISSING,
    cameras=("wrist",),
    teleop=False,
    tele_grasp=True,
    eval=False,
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
    neutral_joints=(0.8402, 1.1271, -2.2545, -2.6843, 1.7169, 1.0783, -1.2701),
)
