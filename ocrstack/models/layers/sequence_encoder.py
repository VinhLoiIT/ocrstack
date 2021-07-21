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


class PoolColEncoder(BaseEncoder):

    def __init__(self, type: str, batch_first: bool = True):
        super().__init__()
        if type == 'max':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        elif type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        else:
            raise ValueError(f'Type = {type} is unsupported')

        self.batch_first = batch_first

    def forward(self, src, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - `src` of shape (B, C, H, W)
        - `src_key_padding_mask` of shape (B, H, W)

        Outputs:
        --------
        - output of shape (B, out_height, E)
        '''
        src = self.pool(src)        # [B, C, 1, W]
        src = src.squeeze(-2)       # [B, E=C, L=W]
        if self.batch_first:
            src = src.transpose(-1, -2)                   # B, T=W, C
        else:
            src = src.permute(2, 0, 1)                    # T=W, B, C

        if src_key_padding_mask is not None:
            # TODO: Only pool locations where there are not paddings
            pass

        return src
