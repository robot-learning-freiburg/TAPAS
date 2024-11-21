from conf._machine import data_naming_config
from conf.encoder.vit_keypoints.nofilter import vit_keypoints_predictor_config
from tapas_gmm.dataset.bc import BCDataConfig
from tapas_gmm.kp_encode_trajectories import Config
from tapas_gmm.policy.encoder import PseudoEncoderPolicyConfig
from tapas_gmm.utils.misc import DataNamingConfig
from tapas_gmm.utils.observation import MaskTypes, ObservationConfig

encoder_naming_config = DataNamingConfig(
    task=None,  # If None, values are taken from data_naming_config
    feedback_type="demos",
    data_root=None,
)

observation_config = ObservationConfig(
    cameras=("wrist",),
    image_crop=None,
    image_dim=(480, 640),
    disk_read_embedding=False,
    disk_read_keypoints=False,
)

policy_config = PseudoEncoderPolicyConfig(
    encoder_name="vit_keypoints",
    encoder_config=vit_keypoints_predictor_config,
    encoder_suffix=None,
    encoder_naming=encoder_naming_config,
    observation=observation_config,
)

bc_data_config = BCDataConfig(
    fragment_length=-1, force_load_raw=True, mask_type=MaskTypes.NONE
)

config = Config(
    policy=policy_config,
    data_naming=data_naming_config,
    observation=observation_config,
    bc_data=bc_data_config,
    constant_encode_with_first_obs=True,
)
