from tapas_gmm.dataset.dc import DCDataConfig, DcDatasetDataType
from tapas_gmm.utils.observation import MaskTypes

datatype_probabilites = (
    (DcDatasetDataType.SINGLE_OBJECT_WITHIN_SCENE, 0),
    (DcDatasetDataType.SINGLE_OBJECT_ACROSS_SCENE, 0),
    (DcDatasetDataType.DIFFERENT_OBJECT, 0),
    (DcDatasetDataType.MULTI_OBJECT, 0.7),
    (DcDatasetDataType.SYNTHETIC_MULTI_OBJECT, 0.3),
)

smo_dc_dat_config = DCDataConfig(
    contrast_set=None,
    contr_cam=None,
    debug=False,
    domain_randomize=True,
    random_crop=True,
    crop_size=(128, 128),
    sample_crop_size=True,
    random_flip=True,
    sample_matches_only_off_mask=True,
    num_matching_attempts=10000,
    num_non_matches_per_match=150,
    _use_image_b_mask_inv=True,
    fraction_masked_non_matches="auto",
    cross_scene_num_samples=10000,  # for different/same obj cross scene
    data_type_probabilities=datatype_probabilites,
    only_use_labels=None,
    only_use_first_object_label=False,
    conflate_so_object_labels=False,
    use_object_pose=False,
    mask_type=MaskTypes.TSDF,
)
