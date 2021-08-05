from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .attention import DotProductAttention


class AttentionLSTMCell(nn.Module):
    r"""An attention-based long short-term memory (LSTM) cell.

    Note:
    At the moment, this class uses `DotProductAttention` for attention mechanism. Therefore, `memory_size` is required
    to equal to `hidden_size`

    Shapes:
    -------
    - memory: (B, S, E1)
    - prev_predict: (B, E2)
    - prev_hidden, prev_cell: (B, num_layers * hidden_size)
    - memory_key_padding_mask: (B, S)
    """

    def __init__(self, memory_size, embed_size, hidden_size, num_layers=1, num_heads=1, bias=True):
        super().__init__()
        assert memory_size == hidden_size

        self.lstm_cells = nn.ModuleList()
        first_cell = nn.LSTMCell(hidden_size + embed_size, hidden_size, bias)
        later_cells = [nn.LSTMCell(hidden_size, hidden_size, bias=bias) for _ in range(num_layers - 1)]
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm_cells.append(first_cell)
        self.lstm_cells.extend(later_cells)

        self.attention = DotProductAttention(scaled=True, embed_dim=hidden_size,
                                             k_dim=memory_size, v_dim=memory_size, num_heads=num_heads)
        self.out = nn.Linear(hidden_size, embed_size)

    def forward(self,
                memory: Tensor,
                prev_predict: Tensor,
                prev_attn: Tensor,
                prev_state: Optional[Tuple[Tensor, Tensor]] = None,
                memory_key_padding_mask: Optional[Tensor] = None
                ) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:

        inputs = torch.cat((prev_predict, prev_attn), dim=1)                    # B, E + H

        if prev_state is None:
            zeros = torch.zeros(inputs.size(0), self.num_layers * self.hidden_size, device=inputs.device)
            prev_state = (zeros, zeros)

        prev_hidden, prev_cell = prev_state

        out_hidden = torch.empty_like(prev_hidden)
        out_cell = torch.empty_like(prev_cell)

        for i, lstm_cell in enumerate(self.lstm_cells):
            in_hidden_i = prev_hidden[:, i*self.hidden_size:(i+1)*self.hidden_size]
            in_cell_i = prev_cell[:, i*self.hidden_size:(i+1)*self.hidden_size]
            out_hidden_i, out_cell_i = lstm_cell(inputs, (in_hidden_i, in_cell_i))
            out_hidden[:, i*self.hidden_size:(i+1)*self.hidden_size] = out_hidden_i
            out_cell[:, i*self.hidden_size:(i+1)*self.hidden_size] = out_cell_i
            inputs = out_hidden_i

        context = self.attention(out_hidden_i.unsqueeze(1), memory, memory,
                                 k_padding_mask=memory_key_padding_mask)[0]     # B, 1, E
        context = context.squeeze(1)                                            # B, H

        out = self.out(context)                                                 # B, E
        return out, context, (out_hidden, out_cell)
