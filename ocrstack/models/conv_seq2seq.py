from typing import Optional

import torch
import torch.nn.functional as F
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .base import BaseModel
from .layers.sequence_decoder import (AttentionLSTMDecoder, BaseDecoder,
                                      TransformerDecoderAdapter)
from .layers.sequence_encoder import BaseEncoder
from .layers.string_decoder import StringDecoder
from .utils import generate_padding_mask_from_lengths


class GeneralizedConvSeq2Seq(BaseModel):

    def __init__(self, cfg):
        # type: (Config,) -> None
        super().__init__(cfg)

        self.backbone = self.build_backbone(cfg)
        self.encoder = self.build_encoder(cfg)
        self.decoder = self.build_decoder(cfg)

        self.max_length = cfg.MODEL.DECODER.MAX_LENGTH

    def build_encoder(self, cfg: Config) -> BaseEncoder:
        cfg_node = cfg.MODEL.ENCODER
        if cfg_node.TYPE == 'tf_encoder':
            return None

        raise ValueError(f'Encoder type = {cfg_node.TYPE} is not supported')

    def build_decoder(self, cfg: Config) -> BaseDecoder:
        cfg_node = cfg.MODEL.DECODER
        if cfg_node.TYPE == 'tf_decoder':
            decoder = TransformerDecoderAdapter(cfg)
            return decoder
        elif cfg_node.TYPE == 'attn_lstm':
            decoder = AttentionLSTMDecoder(cfg)
            return decoder

        raise ValueError(f'Decoder type = {cfg_node.TYPE} is not supported')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def predict(self, batch: Batch):
        predicts = self.forward(batch.images)
        return predicts

    def train_batch(self, batch: Batch):
        logits = self.forward(batch.images, batch.text, batch.lengths)
        return logits

    def compute_loss(self, logits, targets, lengths):
        packed_predicts = pack_padded_sequence(logits, lengths, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        loss = F.cross_entropy(packed_predicts, packed_targets)
        return loss

    def example_inputs(self):
        return (torch.rand(1, 3, 64, 256), )

    def forward(self, images, text=None, lengths=None, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        images = self.backbone(images)                              # B, C, H, W

        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E

        if image_padding_mask is not None:
            image_padding_mask = image_padding_mask.reshape(B, H * W)

        if self.encoder is not None:
            images = self.encoder(images, image_padding_mask)

        if self.training:
            text_padding_mask = generate_padding_mask_from_lengths(lengths - 1).to(images.device)      # B, S
            logits = self.decoder(images, text[:, :-1],
                                  memory_key_padding_mask=image_padding_mask,
                                  tgt_key_padding_mask=text_padding_mask)
            loss = self.compute_loss(logits, text[:, 1:], lengths - 1)
            return loss
        else:
            predicts = self.decoder.decode(images, self.max_length, image_padding_mask)
            return predicts
