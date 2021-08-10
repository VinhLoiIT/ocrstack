import pytest
import torch
import torch.nn as nn
from ocrstack.models.layers.attention import DotProductAttention
from ocrstack.models.layers.attention_lstm import AttentionLSTMCell
from ocrstack.models.layers.sequence_decoder import (AttentionRecurrentDecoder,
                                                     TransformerDecoder)
from ocrstack.models.layers.transformer import TransformerDecoderLayer


@pytest.mark.parametrize('max_length', (1, 4))
def test_transformer_decoder(max_length):
    embed_size = 10
    vocab_size = 5
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2

    decoder = TransformerDecoder(
        nn.Embedding(vocab_size, embed_size, pad_idx),
        nn.Linear(embed_size, vocab_size),
        TransformerDecoderLayer(DotProductAttention(embed_dim=embed_size),
                                DotProductAttention(embed_dim=embed_size),
                                embed_size),
        sos_idx, eos_idx, pad_idx
    )

    B, T, S = 2, max_length, 8
    src = torch.rand(B, S, embed_size)
    tgt = torch.randint(vocab_size, (B, T))

    output = decoder.forward(src, tgt)
    assert output.shape == torch.Size([B, T, vocab_size])

    decoder.eval()
    output = decoder.decode(src, max_length)
    assert output.size(0) == B
    assert output.size(1) <= T + 1
    assert output.size(2) == vocab_size


@pytest.mark.parametrize('max_length', (1, 4))
def test_attention_lstm_decoder_forward(max_length):
    hidden_size = 20
    embed_size = 20
    vocab_size = 10
    memory_size = 15
    num_layers = 1
    num_cells = 1
    num_heads = 1
    sos_idx = 0
    eos_idx = 1

    model = AttentionRecurrentDecoder(
        nn.Embedding(vocab_size, embed_size),
        nn.Linear(embed_size, vocab_size),
        AttentionLSTMCell(memory_size, embed_size, hidden_size, num_cells, num_heads),
        sos_idx,
        eos_idx,
        num_layers,
    )

    B, T, S = 2, max_length, 8
    memory = torch.rand(B, S, memory_size)
    tgt = torch.randint(vocab_size, (B, T))
    outputs = model.forward(memory, tgt)
    assert outputs.shape == torch.Size([B, T, vocab_size])

    predicts = model.decode(memory, T)
    assert predicts.size(0) == B
    assert predicts.size(1) <= T + 1
    assert predicts.size(2) == vocab_size
