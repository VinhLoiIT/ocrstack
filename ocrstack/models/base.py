import torch.nn as nn
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from ocrstack.models.conv import resnet_feature


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
        if cfg_node.TYPE == 'resnet18':
            backbone, _ = resnet_feature('resnet18', cfg_node.PRETRAINED)
            return backbone

        raise ValueError(f'Backbone type = {cfg_node.TYPE} is not supported')
