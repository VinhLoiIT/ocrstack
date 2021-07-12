from typing import Optional, Tuple

import torch
import torch.nn as nn
from ocrstack.opts.sequence_decoder import BaseDecoder
from ocrstack.opts.sequence_encoder import BaseEncoder
from torch import Tensor


class ConvAttnRNN(nn.Module):
    def __init__(self, conv, decoder, feature_size, d_model, encoder=None):
        # type: (nn.Module, BaseDecoder, int, int, Optional[BaseEncoder]) -> None
        super(ConvAttnRNN, self).__init__()
        self.conv = conv
        self.conv_embed = nn.Conv2d(feature_size, d_model, (1, 1))
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, text, lengths, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        images = self.conv(images)                              # B, C, H, W
        images = self.conv_embed(images)                        # B, E, H, W
        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E

        if image_padding_mask is not None:
            image_padding_mask = image_padding_mask.reshape(B, H * W)

        if self.encoder is not None:
            images = self.encoder(images, image_padding_mask)

        text_padding_mask = _generate_padding_mask_from_lengths(lengths).to(images.device)      # B, S
        logits = self.decoder(images, text,
                              memory_key_padding_mask=image_padding_mask,
                              tgt_key_padding_mask=text_padding_mask)
        return logits

    def decode(self, images, max_length, sos_onehot, eos_onehot, image_padding_mask=None):
        # type: (Tensor, int, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        images = self.conv(images)                              # B, C, H, W
        images = self.conv_embed(images)                        # B, E, H, W
        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E

        if image_padding_mask is not None:
            image_padding_mask = image_padding_mask.reshape(B, H * W)

        if self.encoder is not None:
            images = self.encoder(images, image_padding_mask)

        predicts, lengths = self.decoder.decode(images, max_length, sos_onehot, eos_onehot, image_padding_mask)
        return predicts, lengths


def _generate_padding_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    B, S = len(lengths), lengths.max()
    padding_mask = torch.arange(0, S).expand(B, S) >= lengths.unsqueeze(-1)
    return padding_mask
