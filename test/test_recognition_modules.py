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


@pytest.mark.parametrize('eval_mode', [True, False])
def test_build_seq2seq_module(eval_mode):
    cfg = load_yaml('test/config/r18_transformer.yaml')
    module = build_module(cfg)
    if eval_mode:
        module.eval()
    inputs = {
        'images': _create_dummy_image(batch_size=1),
        'targets': _create_dummy_sequence_indices(batch_size=1, length=10, max_index=100)[0]
    }
    outputs = module.forward(inputs)
    assert outputs['logits'].shape == torch.Size([1, 10, 100])


def test_build_seq2seq_module_infer():
    cfg = load_yaml('test/config/r18_transformer.yaml')
    module = build_module(cfg).eval()
    inputs = {
        'images': _create_dummy_image(batch_size=1),
        'max_length': 10,
    }
    with torch.no_grad():
        outputs = module.forward(inputs)
    assert outputs['predicts'].shape[0] == 1
    assert outputs['predicts'].shape[1] <= inputs['max_length'] + 2
    assert (outputs['scores'] <= 1.0).all() and (outputs['scores'] >= 0).all()


def test_generate_square_subsequent_mask():
    mask = generate_square_subsequent_mask(3)
    expected_mask = torch.tensor([[0, float('-inf'), float('-inf')],
                                  [0, 0, float('-inf')],
                                  [0, 0, 0]])
    assert torch.equal(mask, expected_mask)

    mask = generate_square_subsequent_mask(1)
    expected_mask = torch.tensor([[0.0]])
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
