from conf.pretrain.default import training_config  # noqa
from tapas_gmm.utils.observation import ObservationConfig

observation_config = ObservationConfig(
    image_crop=(160, 0, 0, 0), cameras=("wrist",)  # l,r,t,b
)
