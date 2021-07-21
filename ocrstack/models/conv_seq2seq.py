from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from .base import BaseModel
from .layers.attention import DotProductAttention
from .layers.sequence_decoder import (AttentionLSTMDecoder, BaseDecoder,
                                      TransformerDecoderAdapter)
from .layers.sequence_encoder import BaseEncoder
from .layers.string_decoder import StringDecoder
from .utils import generate_padding_mask_from_lengths


class GeneralizedConvSeq2Seq(BaseModel):

    def __init__(self, cfg, string_decode):
        # type: (Config, StringDecoder) -> None
        super().__init__(cfg)

        self.backbone = self.build_backbone(cfg)
        self.encoder = self.build_encoder(cfg)
        self.decoder = self.build_decoder(cfg)

        sos_onehot = F.one_hot(torch.tensor([cfg.MODEL.DECODER.SOS_IDX]), cfg.MODEL.DECODER.VOCAB_SIZE).float()
        eos_onehot = F.one_hot(torch.tensor([cfg.MODEL.DECODER.EOS_IDX]), cfg.MODEL.DECODER.VOCAB_SIZE).float()

        self.register_buffer('sos_onehot', sos_onehot)
        self.register_buffer('eos_onehot', eos_onehot)
        self.max_length = cfg.MODEL.DECODER.MAX_LENGTH

        self.string_decode = string_decode

    def build_encoder(self, cfg: Config) -> BaseEncoder:
        cfg_node = cfg.MODEL.ENCODER
        if cfg_node.TYPE == 'tf_encoder':
            return None

        raise ValueError(f'Encoder type = {cfg_node.TYPE} is not supported')

    def build_decoder(self, cfg: Config) -> BaseDecoder:
        cfg_node = cfg.MODEL.DECODER
        if cfg_node.TYPE == 'tf_decoder':
            decoder = TransformerDecoderAdapter(
                text_embedding=nn.Linear(cfg_node.VOCAB_SIZE, cfg_node.D_MODEL),
                text_classifier=nn.Linear(cfg_node.D_MODEL, cfg_node.VOCAB_SIZE),
                decoder=nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(cfg_node.D_MODEL, cfg_node.NUM_HEADS),
                    cfg_node.NUM_LAYERS
                ),
            )
            return decoder
        elif cfg_node.TYPE == 'attn_lstm':
            decoder = AttentionLSTMDecoder(
                text_embedding=nn.Linear(cfg_node.VOCAB_SIZE, cfg_node.HIDDEN_SIZE),
                text_classifier=nn.Linear(cfg_node.HIDDEN_SIZE, cfg_node.VOCAB_SIZE),
                lstm=nn.LSTMCell(cfg_node.VOCAB_SIZE + cfg_node.HIDDEN_SIZE, cfg_node.HIDDEN_SIZE),
                attention=DotProductAttention(scaled=True),
                teacher_forcing=False,
            )
            return decoder

        raise ValueError(f'Decoder type = {cfg_node.TYPE} is not supported')

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

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
            logits = self.decoder(images, text[:, :-1].float(),
                                  memory_key_padding_mask=image_padding_mask,
                                  tgt_key_padding_mask=text_padding_mask)
            loss = self.compute_loss(logits, text.argmax(dim=-1)[:, 1:], lengths - 1)
            return loss
        else:
            predicts, lengths = self.decoder.decode(images, self.max_length, self.sos_onehot,
                                                    self.eos_onehot, image_padding_mask)
            return predicts, lengths
