from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import ModuleAttributeError

from .base import BaseModel


class ConvRNN(BaseModel):
    def __init__(self, conv, rnn, vocab_size):
        # type: (nn.Module, Union[nn.Module, nn.LSTM, nn.GRU], int) -> None
        super(ConvRNN, self).__init__()
        self.conv = conv
        self.rnn = rnn
        try:
            bidirectional = rnn.bidirectional
            batch_first = rnn.batch_first
            hidden_size = rnn.hidden_size
        except ModuleAttributeError as e:
            print('''If you use custom RNN (not `nn.GRU` nor `nn.LSTM`), please make sure that your implementation
                     has 'batch_first', 'bidirectional', and 'hidden_size' attributes''')
            raise e
        num_direction = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.out = nn.Linear(num_direction * hidden_size, vocab_size)

    def forward(self, images: torch.Tensor):
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
            return outputs
        else:
            outputs = F.softmax(outputs, dim=-1)                # T, B, V
            outputs = outputs.transpose(0, 1)                   # B, T, V
            return outputs
