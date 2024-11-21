from conf._machine import data_naming_config
from tapas_gmm.behavior_cloning import (
    BCDataConfig,
    Config,
    DataLoaderConfig,
    DataNamingConfig,
    EMAConfig,
    TrainingConfig,
)
from tapas_gmm.encoder.encoder import ObservationEncoderConfig
from tapas_gmm.policy.diffusion import (
    ConditionalUnet1DConfig,
    DDPMSchedulerConfig,
    DiffusionPolicyConfig,
    DiffusionPolicyTrainingConfig,
)

batch_size = 128

epochs = 1500

training = DiffusionPolicyTrainingConfig(
    lr=3e-5,
    betas=[0.95, 0.999],
    eps=1e-8,
    weight_decay=1e-6,
    lr_num_epochs=epochs,
)

scheduler = DDPMSchedulerConfig()

obs_dim = 7 + 21

horizon = 16

n_obs_steps = 2
n_action_steps = 8

action_dim = 8

unet = ConditionalUnet1DConfig(
    input_dim=action_dim,
    global_cond_dim=obs_dim * n_obs_steps,
)

obs_encoder = ObservationEncoderConfig(
    ee_pose=True,
    object_poses=True,
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
    # use_ee_pose=True,
    # add_object_poses=True,
    action_scaling=True,
    unet=unet,
    scheduler=scheduler,
    training=training,
    obs_encoder=obs_encoder,
)

ema = EMAConfig()

bc_training = TrainingConfig(
    steps=None,
    epochs=epochs,
    eval_freq=1,
    save_freq=100,
    auto_early_stopping=False,
    wandb_mode="online",
    use_ema=True,
    ema=ema,
    seed=None,
    full_set_training=True,
)

data_loader = DataLoaderConfig(
    train_split=0.9,
    batch_size=batch_size,
    eval_batchsize=batch_size,
    train_workers=2,
)

bc_data = BCDataConfig(
    fragment_length=horizon + 1,
    pre_padding=n_obs_steps - 1,
    post_padding=n_action_steps - 1,
)


config = Config(
    policy_type="diffusion",
    policy=policy,
    training=bc_training,
    data_naming=data_naming_config,
    data_loader=data_loader,
    bc_data=bc_data,
)
