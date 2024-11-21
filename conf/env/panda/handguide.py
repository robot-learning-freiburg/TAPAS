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
        -1.8156525802403096,
        -1.450205922266609,
        0.7125642112622628,
        -2.734993861927857,
        1.485460503301478,
        1.4261113916813748,
        -0.6592043286255856,
    ),
)
