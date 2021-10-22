import pytest
from common_utils import _create_dummy_image

from ocrstack.core.builder import build_module


@pytest.mark.parametrize('arch', ['resnet18', 'resnet34', 'resnet50'])
@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('in_channels', [1, 3])
@pytest.mark.parametrize('num_layers', [4, 3, 2])
def test_resnet(arch, batch_size, in_channels, num_layers):
    cfg = {
        'name': arch,
        'args': {
            'in_channels': in_channels,
            'num_layers': num_layers
        }
    }
    backbone = build_module(cfg)
    dummy_input = _create_dummy_image(batch_size, in_channels)
    outputs = backbone.forward(dummy_input)
