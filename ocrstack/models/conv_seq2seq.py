from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.data.collate import Batch
from ocrstack.opts.sequence_decoder import BaseDecoder
from ocrstack.opts.sequence_encoder import BaseEncoder
from ocrstack.opts.string_decoder import StringDecoder
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .base import BaseModel


@dataclass
class ConvSeq2SeqConfig:
    feature_size: int
    vocab_size: int
    d_model: int
    sos_index: int
    eos_index: int
    max_length: int


class ConvSeq2Seq(BaseModel):
    def __init__(self, cfg, backbone, decoder, string_decode, encoder=None):
        # type: (ConvSeq2SeqConfig, nn.Module, BaseDecoder, StringDecoder ,Optional[BaseEncoder],) -> None
        super(ConvSeq2Seq, self).__init__()
        self.backbone = backbone
        self.conv_embed: Optional[nn.Conv2d] = None
        if cfg.feature_size != cfg.d_model:
            self.conv_embed = nn.Conv2d(cfg.feature_size, cfg.d_model, (1, 1))
        self.encoder = encoder
        self.decoder = decoder

        sos_onehot = F.one_hot(torch.tensor([cfg.sos_index]), cfg.vocab_size).float()
        eos_onehot = F.one_hot(torch.tensor([cfg.eos_index]), cfg.vocab_size).float()

        self.register_buffer('sos_onehot', sos_onehot)
        self.register_buffer('eos_onehot', eos_onehot)
        self.max_length = cfg.max_length

        self.string_decode = string_decode

    def predict(self, batch: Batch):
        predicts, lengths = self.forward(batch.images)
        chars, probs = self.string_decode(predicts, lengths - 1)
        return chars, probs

    def train_batch(self, batch: Batch):
        logits = self.forward(batch.images, batch.text, batch.lengths)
        return logits

    def compute_loss(self, logits, targets, lengths):
        packed_predicts = pack_padded_sequence(logits, lengths, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        loss = F.cross_entropy(packed_predicts, packed_targets)
        return loss

    def forward(self, images, text=None, lengths=None, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        images = self.backbone(images)                              # B, C, H, W

        if self.conv_embed:
            images = self.conv_embed(images)                        # B, E, H, W

        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E

        if image_padding_mask is not None:
            image_padding_mask = image_padding_mask.reshape(B, H * W)

        if self.encoder is not None:
            images = self.encoder(images, image_padding_mask)

        if self.training:
            text_padding_mask = _generate_padding_mask_from_lengths(lengths - 1).to(images.device)      # B, S
            logits = self.decoder(images, text[:, :-1].float(),
                                  memory_key_padding_mask=image_padding_mask,
                                  tgt_key_padding_mask=text_padding_mask)
            loss = self.compute_loss(logits, text.argmax(dim=-1)[:, 1:], lengths - 1)
            return loss
        else:
            predicts, lengths = self.decoder.decode(images, self.max_length, self.sos_onehot,
                                                    self.eos_onehot, image_padding_mask)
            return predicts, lengths


def _generate_padding_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    B, S = len(lengths), lengths.max()
    padding_mask = torch.arange(0, S, device=lengths.device).expand(B, S) >= lengths.unsqueeze(-1)
    return padding_mask
