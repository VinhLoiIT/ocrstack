from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from ocrstack.data.collate import Batch
from ocrstack.models.base import IS2SModel
from ocrstack.models.layers.attention import ScaledDotProductAttention
from ocrstack.models.layers.embedding import Embedding
from ocrstack.models.layers.positional_encoding import (PositionalEncoding1d,
                                                        PositionalEncoding2d)
from ocrstack.models.layers.sequence_decoder import TransformerDecoder
from ocrstack.models.layers.transformer import TransformerDecoderLayer
from torch import Tensor, nn
from torchvision import models


@dataclass()
class ResNetTransformerCfg:
    backbone_arch: str = 'resnet18'
    backbone_pretrained: bool = False
    backbone_feat_dim: int = 256
    drop_backbone_layers: int = 1
    vocab_size: int = 113
    embed_dim: int = 512
    num_heads: int = 1
    sos_idx: int = 0
    eos_idx: int = 1
    pad_idx: int = 2
    decoder_num_layers: int = 1
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    dim_feedforward: int = 2048


class ResNetTransformer(IS2SModel):
    def __init__(self, cfg: ResNetTransformerCfg):
        super().__init__()
        self.image_embed = self.build_image_embed(cfg)
        self.text_embed = self.build_text_embed(cfg)
        self.fc = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=True)

        decoder_layer = TransformerDecoderLayer(
            ScaledDotProductAttention(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads),
            ScaledDotProductAttention(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads),
            embed_dim=cfg.embed_dim,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
        )
        self.decoder = TransformerDecoder(self.text_embed, self.fc, decoder_layer,
                                          cfg.sos_idx, cfg.eos_idx, cfg.pad_idx, cfg.decoder_num_layers)
        self.cfg = cfg

    def build_text_embed(self, cfg: ResNetTransformerCfg):
        in_embed = Embedding(cfg.vocab_size, cfg.embed_dim, cfg.pad_idx, cfg.scale_grad_by_freq, cfg.dropout)
        pos_encoding = PositionalEncoding1d(cfg.embed_dim, batch_first=True)
        return nn.Sequential(OrderedDict([
            ('in_embed', in_embed),
            ('pos_encoding', pos_encoding),
        ]))

    def build_image_embed(self, cfg: ResNetTransformerCfg):
        if cfg.backbone_arch == 'resnet18':
            backbone = models.resnet18(pretrained=cfg.backbone_pretrained)
        elif cfg.backbone_arch == 'resnet34':
            backbone = models.resnet34(pretrained=cfg.backbone_pretrained)
        elif cfg.backbone_arch == 'resnet50':
            backbone = models.resnet50(pretrained=cfg.backbone_pretrained)
        else:
            raise ValueError(f"Unsupported backbone = {cfg.backbone_arch}."
                             "Please consider 'resnet18', 'resnet34' or 'resnet50'")
        backbone = list(backbone.named_children())[:-2 - cfg.drop_backbone_layers]
        backbone = nn.Sequential(OrderedDict(backbone))

        bottleneck = nn.Conv2d(cfg.backbone_feat_dim, cfg.embed_dim, (1, 1))

        pos_encoding = PositionalEncoding2d(cfg.embed_dim)
        return nn.Sequential(OrderedDict([
            ('backbone', backbone),
            ('bottleneck', bottleneck),
            ('pos_encoding', pos_encoding),
        ]))

    def forward(self, images: Tensor, targets: Tensor):
        memory = self.image_embed(images)                   # B, C, H, W
        B, C, H, W = memory.shape
        memory = memory.reshape(B, C, H*W).transpose(1, 2)  # B, S, E
        logits = self.decoder(memory, targets)
        return logits

    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        r'''
        Shapes:
            logits: (B, T, V)
            targets: (B, T)
        '''
        logits = logits.transpose(1, 2)                     # B, V, T
        loss = F.cross_entropy(logits, targets, ignore_index=self.cfg.pad_idx, reduction='mean')
        return loss

    def forward_batch(self, batch: Batch) -> Tensor:
        logits = self(batch.images, batch.text[:, :-1])     # B, T, V
        tgt = batch.text[:, 1:]                             # B, T
        loss = self.compute_loss(logits, tgt)
        return loss

    @torch.jit.export
    def decode_greedy(self, images: Tensor, max_length: int, image_mask: Optional[Tensor] = None) -> Tensor:
        memory = self.image_embed(images)
        B, C, H, W = memory.shape
        memory = memory.reshape(B, C, H*W).transpose(1, 2)  # B, S, E
        predicts = self.decoder.decode(memory, max_length)
        return predicts
