from typing import Optional

import torch
import torch.nn.functional as F
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from torch import Tensor

from .base import BaseModel
from .layers.sequence_decoder import VisualLSTMDecoder
from .layers.sequence_encoder import PoolColEncoder
from .layers.string_decoder import StringDecoder


class GeneralizedCRNN(BaseModel):

    def __init__(self, cfg: Config, string_decoder: StringDecoder):
        super().__init__(cfg)
        self.backbone = self.build_backbone(cfg)
        self.encoder = self.build_encoder(cfg)
        self.decoder = self.build_decoder(cfg)
        self.string_decoder = string_decoder
        self.BLANK_IDX = cfg.MODEL.DECODER.BLANK_IDX
        self.batch_first = cfg.MODEL.DECODER.BATCH_FIRST

    def build_encoder(self, cfg: Config):
        cfg_node = cfg.MODEL.ENCODER
        if cfg_node.TYPE == 'max_pool':
            return PoolColEncoder('max', cfg_node.BATCH_FIRST)
        elif cfg_node.TYPE == 'avg_pool':
            return PoolColEncoder('avg', cfg_node.BATCH_FIRST)

        raise ValueError(f'Unsupported encoder type = {cfg_node.TYPE}')

    def build_decoder(self, cfg: Config):
        cfg_node = cfg.MODEL.DECODER
        if cfg_node.TYPE == 'lstm':
            return VisualLSTMDecoder(cfg)

        raise ValueError(f'Unsupported decoder type = {cfg_node.TYPE}')

    def predict(self, batch: Batch):
        outputs, _ = self.forward(batch.images)
        chars, probs = self.string_decoder(outputs)
        return chars, probs

    def train_batch(self, batch: Batch):
        logits = self.forward(batch.images, batch.text, batch.lengths)
        return logits

    def compute_loss(self, outputs, targets, output_lengths, target_lengths):
        outputs = F.log_softmax(outputs, dim=-1)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        loss = F.ctc_loss(outputs, targets, output_lengths, target_lengths, blank=self.BLANK_IDX)
        return loss

    def example_inputs(self):
        return (torch.rand(1, 3, 64, 256), )

    def forward(self, images, text=None, lengths=None, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        images = self.backbone(images)                              # B, C, H, W

        if self.encoder is not None:
            images = self.encoder(images, image_padding_mask)       # B, T, C or T, B, C

        outputs, out_lengths = self.decoder(images)

        if self.training:
            loss = self.compute_loss(outputs, text.argmax(dim=-1), out_lengths, lengths)
            return loss
        else:
            return outputs, out_lengths
