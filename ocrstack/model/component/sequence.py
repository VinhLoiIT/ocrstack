import torch
import torch.nn as nn
from torch import Tensor


class LSTMEncoderAdapter(nn.Module):
    def __init__(self, lstm: nn.LSTM):
        super(LSTMEncoderAdapter, self).__init__()
        self.lstm = lstm
        self.in_channels = self.lstm.input_size
        self.out_channels = self.lstm.hidden_size

    def forward(self, input_seq: Tensor) -> Tensor:
        input_seq = self.lstm(input_seq)[0]
        return input_seq


class GRUEncoderAdapter(nn.Module):
    def __init__(self, gru: nn.GRU):
        super(GRUEncoderAdapter, self).__init__()
        self.gru = gru
        self.in_channels = self.gru.input_size
        self.out_channels = self.gru.hidden_size

    def forward(self, input_seq: Tensor) -> Tensor:
        input_seq = self.gru(input_seq)[0]
        return input_seq


class TransformerEncoderAdapter(nn.Module):
    def __init__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int, norm=None):
        super(TransformerEncoderAdapter, self).__init__()
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm)
        self.in_channels = encoder_layer.self_attn.embed_dim
        self.out_channels = self.in_channels

    def forward(self, input_seq: Tensor) -> Tensor:
        input_seq = input_seq.transpose(0, 1)
        output = self.encoder(src=input_seq)
        output = output.transpose(0, 1)
        return output


class TransformerDecoderAdapter(nn.Module):
    def __init__(self, decoder_layer: nn.TransformerDecoderLayer, num_layers: int, norm=None):
        super(TransformerDecoderAdapter, self).__init__()
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, norm)
        self.out_channels = decoder_layer.self_attn.embed_dim

    def forward(self,
                input_seq: Tensor,
                target_seq: Tensor,
                ) -> Tensor:
        target_seq = target_seq.transpose(0, 1)
        input_seq = input_seq.transpose(0, 1)
        attn_mask = _generate_square_subsequent_mask(target_seq.size(0)).to(input_seq.device)
        output = self.decoder(target_seq, input_seq, tgt_mask=attn_mask)
        output = output.transpose(0, 1)
        return output


def _generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
