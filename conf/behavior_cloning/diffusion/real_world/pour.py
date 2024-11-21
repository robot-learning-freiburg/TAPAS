from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.default import (
    action_dim,
    batch_size,
    bc_data,
    bc_training,
    data_loader,
    horizon,
    n_action_steps,
    n_obs_steps,
    scheduler,
    training,
)
from conf.encoder.vit_keypoints.nofilter import vit_keypoints_predictor_config
from tapas_gmm.behavior_cloning import Config, DataLoaderConfig
from tapas_gmm.encoder.encoder import ObservationEncoderConfig
from tapas_gmm.policy.diffusion import ConditionalUnet1DConfig, DiffusionPolicyConfig

# from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import ObservationConfig

task = "Pour"

obs_dim = 7 + 15

unet = ConditionalUnet1DConfig(
    input_dim=action_dim,
    global_cond_dim=obs_dim * n_obs_steps,
)

policy = DiffusionPolicyConfig(
    suffix=None,
    obs_as_local_cond=False,
    obs_as_global_cond=True,
    pred_action_steps_only=False,
    horizon=horizon,
    n_obs_steps=n_obs_steps,
    n_action_steps=n_action_steps,
    action_dim=action_dim,
    obs_dim=obs_dim,
    oa_step_convention=True,
    num_inference_steps=100,
    use_ee_pose=True,
    add_object_poses=True,
    action_scaling=True,
    unet=unet,
    scheduler=scheduler,
    training=training,
)

observation_config = ObservationConfig(
    cameras=tuple(),
    image_crop=None,
)

bc_data.kp_pre_encoding = "exp"

data_loader = DataLoaderConfig(
    train_split=1.0,
    batch_size=batch_size,
    eval_batchsize=batch_size,
    train_workers=2,
)

# encoder_naming_config = DataNamingConfig(
#     task=None,  # If None, values are taken from data_naming_config
#     feedback_type="demos",
#     data_root=None,
# )

vit_keypoints_predictor_config.disk_read_keypoints = True

obs_encoder = ObservationEncoderConfig(
    ee_pose=True,
    proprio_obs=False,
    object_poses=False,
    pre_encoding="kp",
    image_encoder=vit_keypoints_predictor_config,
)

data_naming_config.feedback_type = "demos"
data_naming_config.task = task


policy.training.lr_warmup_steps = 50
policy.add_object_poses = False
bc_training.save_freq = 1500

policy.encoder_name = "vit_keypoints"
policy.encoder_config = vit_keypoints_predictor_config
policy.encoder_suffix = None
policy.encoder_naming = data_naming_config
policy.observation = observation_config

policy.obs_encoder = obs_encoder

config = Config(
    policy_type="diffusion",
    policy=policy,
    training=bc_training,
    data_naming=data_naming_config,
    data_loader=data_loader,
    bc_data=bc_data,
)
