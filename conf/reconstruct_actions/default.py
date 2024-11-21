from conf._machine import data_naming_config
from conf.encoder.vit_keypoints.nofilter import vit_keypoints_predictor_config
from tapas_gmm.dataset.bc import BCDataConfig
from tapas_gmm.policy.encoder import PseudoEncoderPolicyConfig
from tapas_gmm.reconstruct_actions import Config
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="demos",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=tuple(),  # ("wrist",),
    image_crop=None,
    image_dim=None,  # (480, 640),
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

bc_data_config = BCDataConfig(
    fragment_length=-1, force_load_raw=True, mask_type=MaskTypes.NONE, cameras=tuple()
)

config = Config(
    data_naming=data_naming_config,
    observation=observation_config,
    bc_data=bc_data_config,
)
