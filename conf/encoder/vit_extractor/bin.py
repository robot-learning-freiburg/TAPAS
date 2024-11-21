from tapas_gmm.encoder.vit_extractor import (
    VitFeatureEncoderConfig,
    VitFeatureModelConfig,
)

vit_model_config = VitFeatureModelConfig(
    load_size=None,
    layer=11,
    facet="token",  # 'key', 'query', 'value', 'token'
    bin=True,
    thresh=0.05,
    vision_net="dino_vits8",  # 'vit_base_patch8_224',
    stride=8,
    include_cls=False,
)

vit_feature_encoder_config = VitFeatureEncoderConfig(
    encoder=vit_model_config, frozen=True
)
