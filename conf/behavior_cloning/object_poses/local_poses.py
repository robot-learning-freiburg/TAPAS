from conf._machine import data_naming_config
from tapas_gmm.behavior_cloning import Config, TrainingConfig
from tapas_gmm.dataset.bc import BCDataConfig
from tapas_gmm.policy.lstm import LSTMPolicyConfig, LSTMPolicyTrainingConfig
from tapas_gmm.utils.config import SET_PROGRAMMATICALLY
from tapas_gmm.utils.data_loading import DataLoaderConfig
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig

bc_data_config = BCDataConfig(
    fragment_length=30,
    pre_embedding=False,
    kp_pre_encoding=None,
    mask_type=MaskTypes.GT,
)

dataloader_config = DataLoaderConfig(
    train_split=0.7,
    batch_size=14,
    eval_batchsize=15,
)

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="pretrain_manual",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

policy_training_config = LSTMPolicyTrainingConfig(
    learning_rate=3e-4,
    weight_decay=3e-6,
)

policy_config = LSTMPolicyConfig(
    suffix=None,
    visual_embedding_dim=7,  # Set by encoder
    proprio_dim=7,
    action_dim=7,
    lstm_layers=2,
    use_ee_pose=True,
    add_gripper_state=False,
    add_object_poses=True,
    training=policy_training_config,
    action_scaling=True,
    poses_in_ee_frame=False,
)

training_config = TrainingConfig(
    steps=10000,
    epochs=None,
    eval_freq=25,
    save_freq=1000,
    auto_early_stopping=False,
    seed=5,
    wandb_mode="online",
)

config = Config(
    policy_type="lstm",
    policy=policy_config,
    training=training_config,
    # observation=observation_config,
    data_naming=data_naming_config,
    data_loader=dataloader_config,
    bc_data=bc_data_config,
)
