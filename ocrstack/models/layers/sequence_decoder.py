from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import Attention


def _decode_unimplemented(self, *input: Any) -> None:
    r"""Defines the computation performed at every call.

    Should be overridden by all subclasses.

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError


class BaseDecoder(nn.Module):

    '''
    Base class for the Decoder component in Seq2Seq architecture

    All derivated classes from this class should perform:
    - Embedding text string from sequence of token indexes to the corresponding Tensor
    - Classify embedded tensor from embedded dimension to the size of vocabulary
    - Decoding a sequence from the source sequence
    '''
    decode: Callable[..., Any] = _decode_unimplemented

    def __init__(self, text_embedding: nn.Module, text_classifier: nn.Module):
        super(BaseDecoder, self).__init__()
        self.text_embedding = text_embedding
        self.text_classifier = text_classifier


class TransformerDecoderAdapter(BaseDecoder):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self, text_embedding: nn.Module, text_classifier: nn.Module, decoder: nn.TransformerDecoder):
        super(TransformerDecoderAdapter, self).__init__(text_embedding, text_classifier)
        self.decoder = decoder

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - memory: (B, S, E)
        - tgt: (B, T, V)

        Returns:
        --------
        - logits: (B, T, V)
        '''
        # Since transformer components working with time-first tensor, we should transpose the shape first
        tgt = self.text_embedding(tgt)              # [B, S, E]
        tgt = tgt.transpose(0, 1)                   # [S, B, E]
        memory = memory.transpose(0, 1)             # [T, B, E]
        tgt_mask = _generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
        memory_mask = None
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = output.transpose(0, 1)                 # [B, T, E]
        output = self.text_classifier(output)           # [B, T, V]
        return output

    @torch.jit.export
    def decode(self, memory, max_length, sos_onehot, eos_onehot, memory_key_padding_mask=None):
        # type: (Tensor, int, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = memory.size(0)
        predicts = sos_onehot.unsqueeze(0).repeat(batch_size, 1, 1)     # [B, 1, V]
        ends = eos_onehot.argmax(-1).repeat(batch_size).squeeze(-1)     # [B]

        predicts = predicts.to(memory.device)
        ends = ends.to(memory.device)

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            text = self.forward(memory, predicts, memory_key_padding_mask)               # [B, T, V]
            output = F.softmax(text[:, [-1]], dim=-1)                    # [B, 1, V]
            predicts = torch.cat([predicts, output], dim=1)     # [B, T + 1, V]

            # set flag for early break
            output = output.squeeze(1).argmax(-1)               # [B]
            current_end = output == ends                        # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        return predicts[:, 1:], lengths  # remove <sos>


class AttentionLSTMDecoder(BaseDecoder):

    def __init__(self, text_embedding, text_classifier, lstm, attention, teacher_forcing=False):
        # type: (nn.Module, nn.Module, nn.LSTMCell, Attention, bool) -> None
        super(AttentionLSTMDecoder, self).__init__(text_embedding, text_classifier)
        self.lstm = lstm
        self.attention = attention
        self.teacher_forcing = teacher_forcing

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        '''
        memory: (B, T, E)
        tgt: (B, T, V)
        '''
        hidden, cell = self._init_hidden(memory.size(0), memory.device)
        outputs = []
        for t in range(tgt.size(1)):
            context = self.attention(hidden.unsqueeze(1), memory, memory)[0]        # B, 1, E
            context = context.squeeze(1)                                            # B, E
            if self.teacher_forcing or t == 0:
                inputs = torch.cat([context, tgt[:, t]], dim=-1)                        # B, E + V
            else:
                inputs = torch.cat([context, F.softmax(outputs[-1], dim=-1)], dim=-1)   # B, E + V
            hidden, cell = self.lstm(inputs, (hidden, cell))                            # B, H
            output = self.text_classifier(hidden)                                       # B, V
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def _init_hidden(self, batch_size: int, device: str) -> Tuple[Tensor, Tensor]:
        hidden_size = self.lstm.hidden_size
        h0 = torch.zeros(batch_size, hidden_size, device=device)
        c0 = torch.zeros(batch_size, hidden_size, device=device)
        return (h0, c0)

    @torch.jit.export
    def decode(self, memory, max_length, sos_onehot, eos_onehot, memory_key_padding_mask=None):
        # type: (Tensor, int, Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        '''
        memory: (B, T, E)
        tgt: (B, T, V)
        '''
        batch_size = memory.size(0)
        hidden, cell = self._init_hidden(batch_size, memory.device)
        sos_onehot = sos_onehot.to(memory.device)
        eos_onehot = eos_onehot.to(memory.device)
        outputs = []

        sos_onehot = sos_onehot.expand(batch_size, sos_onehot.size(-1))                   # [B, V]
        ends = eos_onehot.argmax(-1).squeeze(-1).expand(batch_size)     # [B]

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            context = self.attention(hidden.unsqueeze(1), memory, memory)[0]        # B, 1, E
            context = context.squeeze(1)                                            # B, E
            if t == 0:
                inputs = torch.cat([context, sos_onehot], dim=-1)                   # B, E + V
            else:
                inputs = torch.cat([context, outputs[-1]], dim=-1)                  # B, E + V
            hidden, cell = self.lstm(inputs, (hidden, cell))                        # B, H
            output = self.text_classifier(hidden)                                   # B, V
            output = F.softmax(output, dim=-1)                                      # B, V
            outputs.append(output)

            # set flag for early break
            output = output.argmax(-1)               # [B]
            current_end = output == ends             # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break
        outputs = torch.stack(outputs, dim=1)
        return outputs, lengths
