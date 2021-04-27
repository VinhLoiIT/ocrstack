from typing import Any, Optional, Tuple

import torch.nn as nn
from ocrstack.opts.sequence_decoder import BaseDecoder
from ocrstack.opts.sequence_encoder import BaseEncoder
from torch import Tensor


class Seq2Seq(nn.Module):
    def __init__(self,
                 decoder: BaseDecoder,
                 encoder: Optional[BaseEncoder] = None,
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - src tensor of shape (B, S, E)
        - tgt tensor of shape (B, T, E)
        '''
        memory = src
        if self.encoder is not None:
            memory = self.encoder(src, src_key_padding_mask)
        logits = self.decoder(memory, tgt,
                              memory_key_padding_mask=memory_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)

        return logits

    def decode(self, src, max_length, src_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        # type: (Tensor, int, Optional[Tensor], Optional[Tensor], Any) -> Tuple[Tensor, Tensor]
        memory = src
        if self.encoder is not None:
            memory = self.encoder(src, src_key_padding_mask)        # [B, S, E]

        predicts, lengths = self.decoder.decode(memory, max_length, memory_key_padding_mask, **kwargs)
        return predicts, lengths
