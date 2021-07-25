import torch.nn as nn
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch


class ModelInterface:

    def example_inputs(self):
        raise NotImplementedError()

    def train_batch(self, batch: Batch):
        raise NotImplementedError()

    def predict(self, batch: Batch):
        raise NotImplementedError()


class BaseModel(nn.Module, ModelInterface):

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def example_inputs(self):
        raise NotImplementedError()

    def train_batch(self, batch: Batch):
        raise NotImplementedError()

    def predict(self, batch: Batch):
        raise NotImplementedError()

    def build_backbone(self, cfg: Config) -> nn.Module:
        cfg_node = cfg.MODEL.BACKBONE
        if cfg_node.TYPE[:6] == 'resnet':
            from ocrstack.models.backbone.resnet import resnet
            backbone = resnet(cfg)
            return backbone
        if cfg_node.TYPE[:8] == 'densenet':
            from ocrstack.models.backbone.densenet import densenet
            backbone = densenet(cfg)
            return backbone

        raise ValueError(f'Backbone arch = {cfg_node.TYPE} is not supported')
