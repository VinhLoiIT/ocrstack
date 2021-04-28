import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.data.collate import Batch
from torch.nn.utils.rnn import pack_padded_sequence


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, batch: Batch, train_outputs):
        targets = batch.text.argmax(dim=-1)[:, 1:]          # B, T
        targets_lengths = batch.lengths                     # B, T
        logits = train_outputs                              # B, T, V
        logits = F.log_softmax(logits, dim=-1)

        packed_predicts = pack_padded_sequence(logits, targets_lengths + 1, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets, targets_lengths + 1, batch_first=True)[0]

        with torch.no_grad():
            true_dist = torch.zeros_like(packed_predicts)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, packed_targets.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * packed_predicts, dim=self.dim))
