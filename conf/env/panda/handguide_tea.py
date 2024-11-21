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
    neutral_joints=(
        -2.1522849914045086,
        -1.4276576301005848,
        0.6726070108555111,
        -2.648713635747443,
        1.4144920868880242,
        1.2100827259779383,
        -0.852995067441629,
    ),
)
