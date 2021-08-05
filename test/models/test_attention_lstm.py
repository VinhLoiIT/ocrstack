import torch
import pytest
from ocrstack.models.layers.attention_lstm import AttentionGRUCell, AttentionLSTMCell


@pytest.mark.parametrize('num_layers', (1, 2))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('is_init_prev_state', (True, False))
def test_attention_lstm_cell(num_layers, batch_size, is_init_prev_state):

    memory_size = 8
    hidden_size = 15
    embed_size = 10
    memory_length = 5

    prev_state = None
    if is_init_prev_state:
        hx = torch.rand(batch_size, num_layers * hidden_size)
        prev_state = (hx, hx)

    attn_lstm_cell = AttentionLSTMCell(memory_size, embed_size, hidden_size, num_layers)
    memory = torch.rand(batch_size, memory_length, memory_size)
    prev_predict = torch.rand(batch_size, embed_size)
    prev_attn = torch.rand(batch_size, hidden_size)
    memory_key_padding_mask = None

    out, context, (hidden, cell) = attn_lstm_cell(memory, prev_predict, prev_attn,
                                                  prev_state, memory_key_padding_mask)

    assert out.shape == torch.Size((batch_size, embed_size))
    assert context.shape == torch.Size((batch_size, hidden_size))
    assert hidden.shape == torch.Size((batch_size, num_layers * hidden_size))
    assert cell.shape == torch.Size((batch_size, num_layers * hidden_size))

    # test backward compatible
    out.mean().backward()


@pytest.mark.parametrize('num_layers', (1, 2))
@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('is_init_prev_state', (True, False))
def test_attention_gru_cell(num_layers, batch_size, is_init_prev_state):

    memory_size = 8
    hidden_size = 15
    embed_size = 10
    memory_length = 5

    prev_state = None
    if is_init_prev_state:
        prev_state = torch.rand(batch_size, num_layers * hidden_size)

    attn_lstm_cell = AttentionGRUCell(memory_size, embed_size, hidden_size, num_layers)
    memory = torch.rand(batch_size, memory_length, memory_size)
    prev_predict = torch.rand(batch_size, embed_size)
    prev_attn = torch.rand(batch_size, hidden_size)
    memory_key_padding_mask = None

    out, context, hidden = attn_lstm_cell(memory, prev_predict, prev_attn,
                                          prev_state, memory_key_padding_mask)

    assert out.shape == torch.Size((batch_size, embed_size))
    assert context.shape == torch.Size((batch_size, hidden_size))
    assert hidden.shape == torch.Size((batch_size, num_layers * hidden_size))

    # test backward compatible
    out.mean().backward()
