from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.real_world.default import policy
from conf.evaluate.franka.coffee import eval, franka_env_config
from tapas_gmm.evaluate import Config

policy.encoder_config.disk_read_keypoints = False

policy.obs_encoder.image_encoder.disk_read_keypoints = False
policy.obs_encoder.constant_image_encoding_per_episode = True  # fixed keypoints
policy.obs_encoder.online_encoding = "kp"
policy.obs_encoder.pre_encoding = None

eval.horizon = int(eval.horizon / policy.n_action_steps)

franka_env_config.dynamic_camera_visualization = False
franka_env_config.gripper_threshold = 0.0796 / 2  # 0.0796 is the recorded gripper width

policy.encoder_naming.task = "Coffee"

policy.suffix = "onsn3q5w"

config = Config(
    env_config=franka_env_config,
    eval=eval,
    policy=policy,
    data_naming=data_naming_config,
    policy_type="diffusion",
)
