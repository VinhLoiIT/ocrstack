from typing import OrderedDict

import torch.nn as nn
import torchvision.models as models


def resnet_feature(arch: str, pretrained):
    if arch == 'resnet18':
        resnet = models.resnet18(pretrained=pretrained)
        out_channels = 512
    elif arch == 'resnet34':
        resnet = models.resnet34(pretrained=pretrained)
        out_channels = 512
    elif arch == 'resnet50':
        resnet = models.resnet50(pretrained=pretrained)
        out_channels = 2048

    features = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2]))
    return features, out_channels


def densenet_feature(arch: str, pretrained):
    if arch == 'densenet121':
        densenet = models.densenet121(pretrained=pretrained)
        out_channels = 1024
    elif arch == 'densenet161':
        densenet = models.densenet161(pretrained=pretrained)
        out_channels = 2208
    elif arch == 'densenet169':
        densenet = models.densenet169(pretrained=pretrained)
        out_channels = 1664
    elif arch == 'densenet201':
        densenet = models.densenet201(pretrained=pretrained)
        out_channels = 1920

    features = densenet.features
    return features, out_channels
