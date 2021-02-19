from typing import Union

import torch
import torch.nn as nn
from torchvision import models

from .global_context import GlobalContextBlock

__all__ = [
    'ConvNetAdapter', 'CollumwisePool', 'CollumwisePool',
    'ResNetAdapter', 'ModifiedResNetAdapter', 'GCResNetAdapter',
]


class ConvNetAdapter(nn.Module):
    def __init__(self):
        super(ConvNetAdapter, self).__init__()

    @property
    def in_channels(self):
        return self.__in_channels

    @in_channels.setter
    def in_channels(self, num_channels):
        self.__in_channels = num_channels

    @property
    def out_channels(self):
        return self.__out_channels

    @out_channels.setter
    def out_channels(self, num_channels):
        self.__out_channels = num_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pass


class CollumwiseConcat(ConvNetAdapter):
    def __init__(self, input_size: int, down_sampled_height: int):
        super(CollumwiseConcat, self).__init__()
        self.in_channels = input_size
        self.out_channels = input_size * down_sampled_height

    def forward(self, images: torch.Tensor):
        '''
        images: [B, C, H, W]

        Outputs:
        - images: [B, T, E]
        '''
        B, C, H, W = images.shape
        images = images.reshape(B, C*H, W).unsqueeze(-2)                    # [B, C*H, 1, W]
        return images


class CollumwisePool(ConvNetAdapter):
    def __init__(self, input_size: int, mode: str):
        super(CollumwisePool, self).__init__()
        self.pool: Union[nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d]
        if mode == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        elif mode == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, None))
        else:
            raise ValueError(f'mode should be one of "avg", "max". mode = {mode}')

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        images: [B, C, H, W]

        Outputs:
        - images: [B, T, E]
        '''
        images = self.pool(images)          # [B, C, 1, W]
        return images


class ResNetAdapter(ConvNetAdapter):

    archs = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    def __init__(self, arch, pretrained, droplast: int):
        super().__init__()
        backbone = ResNetAdapter.archs[arch](pretrained)
        self.in_channels = backbone.conv1.in_channels
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        layers = [
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ][:4-droplast]
        self.layers = nn.ModuleList(layers)
        self.out_channels = _resnet_layer_out_channel(self.layers[-1])

        if droplast == 2:
            self.down_sampled = (8, 8)
        elif droplast == 1:
            self.down_sampled = (16, 16)
        elif droplast == 0:
            self.down_sampled = (32, 32)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.conv1(images)
        images = self.bn1(images)
        images = self.relu(images)
        images = self.maxpool(images)

        for layer in self.layers:
            images = layer(images)

        return images


class ModifiedResNetAdapter(ConvNetAdapter):

    archs = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
    }

    def __init__(self, arch: str, pretrained: bool, droplast: int):
        super(ModifiedResNetAdapter, self).__init__()
        backbone: models.ResNet = ModifiedResNetAdapter.archs[arch](pretrained)
        self.in_channels = backbone.conv1.in_channels
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        backbone.layer3[0].conv1.stride = (2, 1)
        backbone.layer3[0].downsample[0].stride = (2, 1)
        backbone.layer4[0].conv1.stride = (2, 1)
        backbone.layer4[0].downsample[0].stride = (2, 1)

        layers = [
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        ][:4-droplast]

        self.layers = nn.ModuleList(layers)
        self.out_channels = _resnet_layer_out_channel(self.layers[-1])

        if droplast == 0:
            self.down_sampled = (32, 8)
        elif droplast == 1:
            self.down_sampled = (16, 8)
        elif droplast == 2:
            self.down_sampled = (8, 8)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.conv1(images)
        images = self.bn1(images)
        images = self.relu(images)
        images = self.maxpool(images)

        for layer in self.layers:
            images = layer(images)

        return images


class GCResNetAdapter(ConvNetAdapter):
    def __init__(self, resnet_adapter: ResNetAdapter):
        self.resnet_adapter = resnet_adapter
        self.gcblocks = nn.ModuleList([])
        for layer in resnet_adapter.layers:
            self.gcblocks.append(GlobalContextBlock(_resnet_layer_out_channel(layer)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.resnet_adapter.conv1(images)
        images = self.resnet_adapter.bn1(images)
        images = self.resnet_adapter.relu(images)
        images = self.resnet_adapter.maxpool(images)

        for i in range(len(self.resnet_adapter.layers)):
            images = self.resnet_adapter.layers[i](images)
            images = self.gcblocks[i](images)

        return images


def _resnet_layer_out_channel(layer):
    for op in reversed(list(layer[-1].children())):
        if isinstance(op, nn.BatchNorm2d):
            return op.num_features
