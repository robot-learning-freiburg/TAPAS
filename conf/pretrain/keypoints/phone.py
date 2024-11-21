from conf._machine import data_naming_config
from conf.dataset.dc.phone import smo_dc_dat_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.observation.wrist_nocrop import observation_config
from conf.pretrain.default import observation_config, training_config
from tapas_gmm.pretrain import Config, TrainingConfig
from tapas_gmm.utils.data_loading import DataLoaderConfig

dataloader_config = DataLoaderConfig(
    train_split=0.7,
    batch_size=128,
    eval_batchsize=128,
)

training_config = TrainingConfig(
    steps=500,
    eval_freq=2,
    save_freq=100,
    auto_early_stopping=False,
    seed=1,
    wandb_mode="online",
)

config = Config(
    encoder_name="keypoints",
    encoder_config=keypoints_predictor_config,
    encoder_suffix=None,
    training=training_config,
    observation=observation_config,
    data_naming=data_naming_config,
    dataloader=dataloader_config,
    dc_data=smo_dc_dat_config,
)
