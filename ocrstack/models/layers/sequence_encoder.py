from typing import Optional

import torch.nn as nn
from torch import Tensor


class BaseEncoder(nn.Module):
    '''
    Base class for encoding a sequence

    This and all derived classes should receive a sequence and return a corresponding sequence with the same length
    '''
    def __init__(self):
        super(BaseEncoder, self).__init__()


class TransformerEncoderAdapter(BaseEncoder):
    def __init__(self, encoder: nn.TransformerEncoderLayer):
        super(TransformerEncoderAdapter, self).__init__()
        self.encoder = encoder

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
        src = src.transpose(0, 1)
        output = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)
        return output
