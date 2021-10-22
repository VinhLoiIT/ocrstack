from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ocrstack.core.builder import MODULE_REGISTRY, build_module
from ocrstack.data.collate import Batch

from .base import ITrainableS2S
from .sequence_decoder import generate_square_subsequent_mask


@MODULE_REGISTRY.register()
class Seq2SeqModule(ITrainableS2S):
    def __init__(self, **cfg):
        super().__init__()

        self.backbone = None
        if cfg.get('backbone', None) is not None:
            self.backbone = build_module(cfg['backbone'])

        self.encoder = None
        if cfg.get('encoder', None) is not None:
            self.encoder = build_module(cfg['encoder'])

        if 'decoder' not in cfg.keys():
            raise KeyError(
                '"decoder" must be in config. Available keys are '
                f'"{list(cfg.keys())}"'
            )
        self.decoder = build_module(cfg['decoder'])

        self.src_embedding = None
        if cfg.get('src_embedding', None) is not None:
            self.src_embedding = build_module(cfg['src_embedding'])

        self.tgt_embedding = None
        if cfg.get('tgt_embedding', None) is not None:
            self.tgt_embedding = build_module(cfg['tgt_embedding'])

        self.classifier = build_module(cfg['classifier'])

        self.pad_idx: int = cfg['pad_idx']

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        images: Tensor = inputs['images']

        memory = images
        if self.backbone is not None:
            memory = self.backbone(images)                                  # B, C, H, W

        B, C, H, W = memory.shape
        memory = memory.reshape(B, C, H*W).transpose(1, 2).contiguous()     # B, S, E

        if self.src_embedding is not None:
            memory = self.src_embedding(memory)

        if self.encoder is not None:
            memory = self.encoder(memory)

        if not self.training and 'targets' not in inputs.keys():
            return self._forward_infer(memory)

        targets: Tensor = inputs['targets']
        return self._forward_train(memory, targets)

    def _forward_train(self, memory: Tensor, targets: Tensor) -> Dict[str, Any]:
        r'''Forward inputs during training

        Args:
            - memory: (B, S, E)
            - tgt: (B, T)

        Returns:
            a dictionary contains:
            - logits: (B, T, V)
        '''
        tgt_key_padding_mask = (targets == self.pad_idx)

        if self.tgt_embedding is not None:
            targets = self.tgt_embedding(targets)
        tgt_mask = generate_square_subsequent_mask(targets.size(1)).to(memory.device)
        memory_mask = None
        out = self.decoder(targets, memory, tgt_mask, memory_mask, tgt_key_padding_mask)
        out = self.classifier(out)                   # [B, T, V]
        logits = self.decoder(memory, targets)
        return {
            'logits': logits
        }

    def _forward_infer(self, memory: torch.Tensor) -> Dict[str, Any]:
        pass

    @torch.jit.export
    def decode_greedy(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = memory.size(0)
        sos_inputs = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=memory.device)  # [B, 1]
        scores = torch.zeros(batch_size, device=memory.device)                                          # [B]

        inputs = sos_inputs                                                                             # [B, T=1]
        end_flag = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)                      # [B]
        for _ in range(max_length + 1):
            output = self.forward(memory, inputs, memory_key_padding_mask)                              # [B, T, V]
            output = F.log_softmax(output[:, [-1]], dim=-1)                                             # [B, 1, V]
            score, index = output.max(dim=-1)                                                           # [B, 1]
            scores = scores + score.squeeze(1)                                                          # [B]
            inputs = torch.cat((inputs, index), dim=1)                                                  # [B, T+1]

            # early break
            end_flag = end_flag | (index == self.eos_idx)                                               # [B]
            if end_flag.all():
                break

        return inputs, torch.exp(scores)                                                                # [B, T], [B]

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
