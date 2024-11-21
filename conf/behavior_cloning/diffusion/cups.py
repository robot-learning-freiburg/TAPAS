from conf._machine import data_naming_config
from conf.behavior_cloning.diffusion.default import (
    action_dim,
    bc_data,
    bc_training,
    data_loader,
    horizon,
    n_action_steps,
    n_obs_steps,
    scheduler,
    training,
)
from tapas_gmm.behavior_cloning import Config
from tapas_gmm.policy.diffusion import ConditionalUnet1DConfig, DiffusionPolicyConfig

obs_dim = 7 + 28

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

config = Config(
    policy_type="diffusion",
    policy=policy,
    training=bc_training,
    data_naming=data_naming_config,
    data_loader=data_loader,
    bc_data=bc_data,
)
