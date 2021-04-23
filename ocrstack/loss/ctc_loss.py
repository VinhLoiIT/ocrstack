from ocrstack.data.vocab import CTCVocab
from ocrstack.data.collate import Batch
import torch
import torch.nn as nn


class CTCLoss(nn.CTCLoss):
    def __init__(self, vocab: CTCVocab, **kwargs):
        super(CTCLoss, self).__init__(blank=vocab.BLANK_IDX, **kwargs)

    def forward(self, batch: Batch, train_outputs):
        targets = batch.text.argmax(dim=-1)                                                 # B, T
        targets_lengths = batch.lengths                                                     # B
        outputs_lengths = torch.ones_like(targets_lengths) * train_outputs.size(0)          # B
        loss = super().forward(train_outputs, targets, outputs_lengths, targets_lengths)
        return loss
