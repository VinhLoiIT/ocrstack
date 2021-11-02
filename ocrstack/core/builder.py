from .registry import Registry

LOSS_REGISTRY = Registry('Loss')
MODULE_REGISTRY = Registry('Module')
DATASET_REGISTRY = Registry('Dataset')


def build_loss(cfg):
    return LOSS_REGISTRY.build_from_cfg(cfg)


def build_module(cfg):
    return MODULE_REGISTRY.build_from_cfg(cfg)


def build_dataset(cfg):
    return DATASET_REGISTRY.build_from_cfg(cfg)
