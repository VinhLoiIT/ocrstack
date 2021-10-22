import pytest
import torch
from common_utils import _create_dummy_image, _create_dummy_sequence

from ocrstack.core.builder import (build_backbone, build_embedding,
                                   build_encoder)


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
    backbone = build_backbone(cfg)
    dummy_input = _create_dummy_image(batch_size, in_channels)
    outputs = backbone.forward(dummy_input)


@pytest.mark.parametrize('batch_size', [1, 2])
def test_transformer_pe_1d(batch_size):
    cfg = {
        'name': 'TransformerPE1D',
        'args': {
            'd_model': 6,
        }
    }
    module = build_embedding(cfg)
    dummy_input = _create_dummy_sequence(batch_size, channels=6)
    outputs = module.forward(dummy_input)


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('length', [1, 2])
@pytest.mark.parametrize('d_model', [12])
@pytest.mark.parametrize('nhead', [1, 2])
def test_transformer_encoder(batch_size, length, d_model, nhead):
    cfg = {
        'name': 'TransformerEncoder',
        'args': {
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': 18,
            'dropout': 0.1,
            'activation': 'relu',
            'num_layers': 1,
            'layer_norm': True,
        }
    }

    module = build_encoder(cfg)
    dummy_input = _create_dummy_sequence(batch_size, length, channels=d_model)
    outputs = module.forward(dummy_input)
    assert outputs.shape == torch.Size([batch_size, length, d_model])
