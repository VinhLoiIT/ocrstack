import pytest
import torch
from common_utils import (_create_dummy_image, _create_dummy_sequence,
                          _create_dummy_sequence_indices)

from ocrstack.core.builder import build_module
from ocrstack.core.loading import load_yaml
from ocrstack.models.recognition.sequence_decoder import (
    generate_padding_mask_from_lengths, generate_square_subsequent_mask)


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

    module = build_module(cfg)
    dummy_input = _create_dummy_sequence(batch_size, length, channels=d_model)
    outputs = module.forward(dummy_input)
    assert outputs.shape == torch.Size([batch_size, length, d_model])


def test_build_seq2seq_module():
    cfg = load_yaml('test/config/r18_transformer.yaml')
    module = build_module(cfg)
    inputs = {
        'images': _create_dummy_image(batch_size=1),
        'targets': _create_dummy_sequence_indices(batch_size=1, length=10, max_index=100)[0]
    }
    outputs = module.forward(inputs)
    assert outputs['logits'].shape == torch.Size([1, 10, 100])


def test_generate_square_subsequent_mask():
    mask = generate_square_subsequent_mask(5)
    expected_mask = torch.tensor([[1, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0],
                                  [1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)

    mask = generate_square_subsequent_mask(1)
    expected_mask = torch.tensor([[True]])
    assert torch.equal(mask, expected_mask)


def test_generate_padding_mask_from_lengths():
    lengths = torch.tensor([3, 2, 5, 1])
    mask = generate_padding_mask_from_lengths(lengths)
    expected_mask = torch.tensor([[0, 0, 0, 1, 1],
                                  [0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)

    lengths = torch.tensor([1])
    mask = generate_padding_mask_from_lengths(lengths)
    expected_mask = torch.tensor([[False]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)
