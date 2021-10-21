import pytest
import torch

from ocrstack.core.builder import build_backbone


@pytest.mark.parametrize('arch', ['resnet18', 'resnet34', 'resnet50'])
@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('in_channels', [1, 3])
def test_resnet(arch, batch_size, in_channels):
    cfg = {
        'name': arch,
        'kwargs': {
            'in_channels': in_channels
        }
    }
    backbone = build_backbone(cfg)
    dummy_input = torch.rand(batch_size, in_channels, 64, 64)
    outputs = backbone.forward(dummy_input)
