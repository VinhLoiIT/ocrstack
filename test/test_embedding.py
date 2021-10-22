import pytest
from common_utils import _create_dummy_sequence

from ocrstack.core.builder import build_module


@pytest.mark.parametrize('batch_size', [1, 2])
def test_transformer_pe_1d(batch_size):
    cfg = {
        'name': 'TransformerPE1D',
        'args': {
            'd_model': 6,
        }
    }
    module = build_module(cfg)
    dummy_input = _create_dummy_sequence(batch_size, channels=6)
    outputs = module.forward(dummy_input)
