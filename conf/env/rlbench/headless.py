from omegaconf import MISSING

from tapas_gmm.env.rlbench import RLBenchEnvironmentConfig

rlbench_env_config = RLBenchEnvironmentConfig(
    task=MISSING,
    cameras=("wrist", "front"),
    camera_pose={
        "base": (0.2, 0.0, 0.2, 0, 0.194, 0.0, -0.981),
        "overhead": (0.2, 0.0, 0.2, 7.7486e-07, -0.194001, 7.7486e-07, 0.981001),
    },
    image_size=(256, 256),
    static=False,
    headless=True,
    gripper_plot=False,
    scale_action=False,
    delay_gripper=False,
    postprocess_actions=True,
)
