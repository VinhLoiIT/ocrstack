import torch
from ocrstack.config.config import Config
from ocrstack.ops.sequence_encoder import TransformerEncoderAdapter


def test_transformer_encoder_adapter_forward():

    cfg = Config()
    cfg.MODEL.ENCODER.TYPE = 'tf_encoder'
    cfg.MODEL.ENCODER.D_MODEL = 10
    cfg.MODEL.ENCODER.NUM_HEADS = 5
    cfg.MODEL.ENCODER.NUM_LAYERS = 2
    model = TransformerEncoderAdapter(cfg)

    B, S = 2, 4
    src = torch.rand(B, S, cfg.MODEL.ENCODER.D_MODEL)
    outputs = model.forward(src)
    assert outputs.shape == torch.Size([B, S, cfg.MODEL.ENCODER.D_MODEL])
