import pytest
from ocrstack.models.layers.attention_lstm import AttentionLSTMCell
from ocrstack.config.config import Config
import torch
import torch.nn as nn
from ocrstack.models.layers.sequence_decoder import (AttentionRecurrentDecoder,
                                                     TransformerDecoderAdapter)


def test_transformer_decoder_adapter_decode():
    cfg = Config()
    cfg.MODEL.DECODER.TYPE = 'tf_decoder'
    cfg.MODEL.DECODER.D_MODEL = 10
    cfg.MODEL.DECODER.NUM_HEADS = 5
    cfg.MODEL.DECODER.NUM_LAYERS = 2
    cfg.MODEL.DECODER.MAX_LENGTH = 10

    cfg.MODEL.TEXT_EMBED.EMBED_SIZE = 10
    cfg.MODEL.TEXT_EMBED.VOCAB_SIZE = 5
    cfg.MODEL.TEXT_EMBED.SOS_IDX = 0
    cfg.MODEL.TEXT_EMBED.EOS_IDX = 1
    cfg.MODEL.TEXT_EMBED.PAD_IDX = 2
    cfg.MODEL.TEXT_EMBED.OUT_BIAS = False
    cfg.MODEL.TEXT_EMBED.SHARE_WEIGHT_IN_OUT = True

    decoder = TransformerDecoderAdapter(cfg)

    B, T, S = 2, 4, 8
    src = torch.rand(B, S, cfg.MODEL.DECODER.D_MODEL)
    tgt = torch.randint(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE, (B, T))

    decoder.eval()
    output = decoder.decode(src, 10)
    output = decoder.forward(src, tgt)
    assert output.shape == torch.Size([B, T, cfg.MODEL.TEXT_EMBED.VOCAB_SIZE])


def test_transformer_decoder_adapter_pe():
    cfg = Config()
    cfg.MODEL.DECODER.TYPE = 'tf_decoder'
    cfg.MODEL.DECODER.D_MODEL = 10
    cfg.MODEL.DECODER.NUM_HEADS = 5
    cfg.MODEL.DECODER.NUM_LAYERS = 2
    cfg.MODEL.DECODER.MAX_LENGTH = 10

    cfg.MODEL.TEXT_EMBED.POS_ENC_TYPE == 'sinusoidal'
    cfg.MODEL.TEXT_EMBED.EMBED_SIZE = 10
    cfg.MODEL.TEXT_EMBED.VOCAB_SIZE = 5
    cfg.MODEL.TEXT_EMBED.SOS_IDX = 0
    cfg.MODEL.TEXT_EMBED.EOS_IDX = 1
    cfg.MODEL.TEXT_EMBED.PAD_IDX = 2
    cfg.MODEL.TEXT_EMBED.OUT_BIAS = False
    cfg.MODEL.TEXT_EMBED.SHARE_WEIGHT_IN_OUT = True

    decoder = TransformerDecoderAdapter(cfg)

    B, T, S = 2, 4, 8
    src = torch.rand(B, S, cfg.MODEL.DECODER.D_MODEL)
    tgt = torch.randint(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE, (B, T))

    decoder.eval()
    output = decoder.decode(src, 10)
    output = decoder.forward(src, tgt)
    assert output.shape == torch.Size([B, T, cfg.MODEL.TEXT_EMBED.VOCAB_SIZE])


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
