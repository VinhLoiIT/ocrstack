from ocrstack.config.config import Config
import torch
from ocrstack.models.layers.sequence_decoder import (AttentionLSTMDecoder,
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

    decoder = TransformerDecoderAdapter(cfg)

    B, T, S = 2, 4, 8
    src = torch.rand(B, S, cfg.MODEL.DECODER.D_MODEL)
    tgt = torch.randint(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE, (B, T))

    decoder.eval()
    output = decoder.decode(src, 10)
    assert isinstance(output, tuple)

    output = decoder.forward(src, tgt)
    assert output.shape == torch.Size([B, T, cfg.MODEL.TEXT_EMBED.VOCAB_SIZE])


def test_attention_lstm_decoder_forward():
    cfg = Config()
    cfg.MODEL.DECODER.TYPE = 'attn_lstm'
    cfg.MODEL.DECODER.HIDDEN_SIZE = 20
    cfg.MODEL.DECODER.MAX_LENGTH = 5
    cfg.MODEL.DECODER.TEACHER_FORCING = False

    cfg.MODEL.TEXT_EMBED.EMBED_SIZE = 20
    cfg.MODEL.TEXT_EMBED.VOCAB_SIZE = 10
    cfg.MODEL.TEXT_EMBED.SOS_IDX = 0
    cfg.MODEL.TEXT_EMBED.EOS_IDX = 1
    cfg.MODEL.TEXT_EMBED.PAD_IDX = 2

    model = AttentionLSTMDecoder(cfg)

    B, T, S = 2, 4, 8
    src = torch.rand(B, S, cfg.MODEL.DECODER.HIDDEN_SIZE)
    tgt = torch.randint(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE, (B, T))
    outputs = model.forward(src, tgt)
    assert outputs.shape == torch.Size([B, T, cfg.MODEL.TEXT_EMBED.VOCAB_SIZE])
