from .registry import Registry

LOSS_REGISTRY = Registry('Loss')
BACKBONE_REGISTRY = Registry('Backbone')
ENCODER_REGISTRY = Registry('Encoder')
DECODER_REGISTRY = Registry('Decoder')


def build_backbone(cfg):
    return BACKBONE_REGISTRY.build_from_cfg(cfg)
