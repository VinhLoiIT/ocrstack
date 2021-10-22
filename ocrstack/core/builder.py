from .registry import Registry

MODULE_REGISTRY = Registry('Module')


def build_module(cfg):
    return MODULE_REGISTRY.build_from_cfg(cfg)
