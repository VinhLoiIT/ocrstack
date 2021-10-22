from typing import Optional

import torch.nn as nn
from torch import Tensor

from ocrstack.core.builder import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class TransformerEncoder(nn.TransformerEncoder):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 num_layers: int = 1,
                 layer_norm: bool = False,
                 layer_norm_eps: float = 0.00001,
                 ):
        norm = None
        if layer_norm:
            norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                batch_first=True,
        )
        super(TransformerEncoder, self).__init__(layer, num_layers, norm)

    def forward(self, src, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - `src` of shape (B, T, E)
        - `src_key_padding_mask` of shape (B, T)

        Outputs:
        --------
        - output of shape (B, T, E)
        '''
        output = super().forward(src, src_key_padding_mask=src_key_padding_mask)
        return output
