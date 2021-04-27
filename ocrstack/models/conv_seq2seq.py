from typing import Any, Optional, Tuple
from ocrstack.models.seq2seq import Seq2Seq

import torch
import torch.nn as nn

from .base import BaseModel


class ConvSeq2Seq(BaseModel):
    def __init__(self, conv, seq2seq, feature_size, d_model):
        # type: (nn.Module, Seq2Seq, int, int, int) -> None
        super(ConvSeq2Seq, self).__init__()
        self.conv = conv
        self.conv_embed = nn.Conv2d(feature_size, d_model, (1, 1))
        self.seq2seq = seq2seq

    def forward(self, images, text=None, lengths=None):
        images = self.conv(images)                              # B, C, H, W
        images = self.conv_embed(images)                        # B, E, H, W
        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E
        text_padding_mask = _generate_padding_mask_from_lengths(lengths).to(images.device)      # B, S
        outputs = self.seq2seq(images, text, None, None, text_padding_mask)                     # B, T, V
        return outputs

    def decode(self, images, max_length, image_padding_mask=None, **kwargs):
        # type: (torch.Tensor, int, Optional[torch.Tensor], Any) -> Tuple[torch.Tensor, torch.Tensor]
        images = self.conv(images)                              # B, C, H, W
        images = self.conv_embed(images)                        # B, E, H, W
        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E
        outputs = self.seq2seq.decode(images, max_length, None, None)
        return outputs


def _generate_padding_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    B, S = len(lengths), lengths.max()
    padding_mask = torch.arange(0, S).expand(B, S) >= lengths.unsqueeze(-1)
    return padding_mask
