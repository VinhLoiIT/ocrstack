from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.data.collate import Batch
from ocrstack.opts.string_decoder import StringDecoder

from .base import BaseModel


class ConvRNN(BaseModel):
    def __init__(self, conv, rnn, vocab_size, blank_index, string_decode):
        # type: (nn.Module, Union[nn.Module, nn.LSTM, nn.GRU], int, int, StringDecoder) -> None
        super(ConvRNN, self).__init__()
        self.conv = conv
        self.rnn = rnn
        bidirectional = rnn.bidirectional
        batch_first = rnn.batch_first
        hidden_size = rnn.hidden_size
        num_direction = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.out = nn.Linear(num_direction * hidden_size, vocab_size)
        self.string_decode = string_decode
        self.blank_index = blank_index

    def compute_loss(self, outputs, targets, lengths):
        outputs_lengths = torch.ones_like(lengths) * outputs.size(0)          # B
        loss = F.ctc_loss(outputs, targets, outputs_lengths, lengths, blank=self.blank_index)
        return loss

    def example_inputs(self):
        return (torch.rand(1, 3, 64, 256), )

    def forward(self, images, text=None, lengths=None):
        # type: (torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]) -> torch.Tensor
        images = self.conv(images)                              # B, C, H, W
        images = torch.mean(images, dim=2, keepdim=False)       # B, C, W
        if self.batch_first:
            images = images.transpose(-1, -2)                   # B, T=W, C
        else:
            images = images.permute(2, 0, 1)                    # T=W, B, C
        outputs, _ = self.rnn(images)                           # T, B, H*num_direction
        outputs = self.out(outputs)                             # T, B, V

        if self.training:
            outputs = F.log_softmax(outputs, dim=-1)            # T, B, V
            loss = self.compute_loss(outputs, text, lengths)
            return loss
        else:
            outputs = F.softmax(outputs, dim=-1)                # T, B, V
            outputs = outputs.transpose(0, 1)                   # B, T, V
            return outputs

    def train_batch(self, batch: Batch):
        return self.forward(batch.images, batch.text.argmax(dim=-1), batch.lengths)

    def predict(self, batch: Batch):
        outputs = self.forward(batch.images)
        return self.string_decode(outputs)
