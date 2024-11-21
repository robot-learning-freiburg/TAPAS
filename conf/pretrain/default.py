from tapas_gmm.pretrain import TrainingConfig
from tapas_gmm.utils.observation import ObservationConfig

training_config = TrainingConfig(
    steps=int(1e5),
    eval_freq=5,
    save_freq=20000,
    auto_early_stopping=False,
    seed=1,
    wandb_mode="online",
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
)
