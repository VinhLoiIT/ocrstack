import torch
import torch.nn as nn
from ocrstack.opts.sequence_encoder import TransformerEncoderAdapter


def test_transformer_encoder_adapter_forward():
    d_model = 10
    nhead = 5
    dim_feedforward = 100
    batch_size = 2
    src_length = 4

    tf_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
    tf_encoder = nn.TransformerEncoder(tf_encoder_layer, 1)

    encoder = TransformerEncoderAdapter(tf_encoder)

    src = torch.rand(batch_size, src_length, d_model)

    output = encoder.forward(src)
    assert output.shape == torch.Size([batch_size, src_length, d_model])
