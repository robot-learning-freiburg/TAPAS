from omegaconf import MISSING

from conf._machine import data_naming_config
from conf.encoder.keypoints.phone import keypoints_predictor_config
from conf.observation.panda import observation_config
from conf.pretrain.default import training_config
from tapas_gmm.behavior_cloning import Config, TrainingConfig
from tapas_gmm.dataset.bc import BCDataConfig
from tapas_gmm.pretrain import Config, TrainingConfig
from tapas_gmm.utils.data_loading import DataLoaderConfig
from tapas_gmm.utils.observation import ObservationConfig
