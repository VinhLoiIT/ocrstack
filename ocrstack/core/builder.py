from .registry import Registry, RegistryComposite

LOSS_REGISTRY = Registry('Loss')
BACKBONE_REGISTRY = Registry('Backbone')
ENCODER_REGISTRY = Registry('Encoder')
DECODER_REGISTRY = Registry('Decoder')
EMBEDDING_REGISTRY = Registry('Embedding')
MODEL_REGISTRY = RegistryComposite('Model')


def build_backbone(cfg):
    return BACKBONE_REGISTRY.build_from_cfg(cfg)


def build_encoder(cfg):
    return ENCODER_REGISTRY.build_from_cfg(cfg)


def build_decoder(cfg):
    return DECODER_REGISTRY.build_from_cfg(cfg)


def build_loss(cfg):
    return LOSS_REGISTRY.build_from_cfg(cfg)


def build_embedding(cfg):
    return EMBEDDING_REGISTRY.build_from_cfg(cfg)


def build_model(cfg):
    return MODEL_REGISTRY.build_from_cfg(cfg)
