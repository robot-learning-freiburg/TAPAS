from tapas_gmm.encoder.vit_extractor import (
    VitFeatureEncoderConfig,
    VitFeatureModelConfig,
)

vit_model_config = VitFeatureModelConfig(
    load_size=252,
    layer=11,
    facet="token",  # 'key', 'query', 'value', 'token'
    bin=False,
    thresh=0.05,
    vision_net="dinov2_vits14",  # 'vit_base_patch8_224',
    stride=7,
    include_cls=False,
    pad=True,
)

vit_feature_encoder_config = VitFeatureEncoderConfig(
    encoder=vit_model_config, frozen=True
)
