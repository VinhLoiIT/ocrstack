import torch.nn as nn
from ocrstack.data.collate import Batch
from torch.nn.utils.rnn import pack_padded_sequence


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(*args, **kwargs)

    def forward(self, batch: Batch, train_outputs):
        targets = batch.text.argmax(dim=-1)                                                 # B, T
        targets_lengths = batch.lengths                                                     # B, T
        logits = train_outputs

        packed_predicts = pack_padded_sequence(logits, targets_lengths + 1, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets, targets_lengths + 1, batch_first=True)[0]
        loss = super().forward(packed_predicts, packed_targets)
        return loss
