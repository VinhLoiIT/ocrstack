from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from torch import Tensor

from ..utils import generate_square_subsequent_mask
from .attention import DotProductAttention


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

    def build_embedding(self, cfg: Config) -> Tuple[nn.Module, nn.Module]:
        out_embed = nn.Linear(cfg.MODEL.TEXT_EMBED.EMBED_SIZE,
                              cfg.MODEL.TEXT_EMBED.VOCAB_SIZE,
                              bias=False)
        in_embed = nn.Embedding(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE,
                                cfg.MODEL.TEXT_EMBED.EMBED_SIZE,
                                cfg.MODEL.TEXT_EMBED.PAD_IDX)
        return in_embed, out_embed


class TransformerDecoderAdapter(BaseDecoder):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self, cfg):
        super(TransformerDecoderAdapter, self).__init__()
        self.in_embed, self.out_embed = self.build_embedding(cfg)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(cfg.MODEL.DECODER.D_MODEL, cfg.MODEL.DECODER.NUM_HEADS),
            cfg.MODEL.DECODER.NUM_LAYERS
        )
        self.sos_idx = cfg.MODEL.TEXT_EMBED.SOS_IDX
        self.eos_idx = cfg.MODEL.TEXT_EMBED.EOS_IDX

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - memory: (B, S, E)
        - tgt: (B, T)

        Returns:
        --------
        - logits: (B, T, V)
        '''
        # Since transformer components working with time-first tensor, we should transpose the shape first
        tgt = self.in_embed(tgt)                    # [B, T, E]
        tgt = tgt.transpose(0, 1)                   # [T, B, E]
        memory = memory.transpose(0, 1)             # [S, B, E]
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
        memory_mask = None
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = output.transpose(0, 1)                 # [B, T, E]
        output = self.out_embed(output)                 # [B, T, V]
        return output

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = memory.size(0)
        inputs = torch.empty(batch_size, 1, dtype=torch.long, device=memory.device).fill_(self.sos_idx)
        outputs = []
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            text = self.forward(memory, inputs, memory_key_padding_mask)        # [B, T, V]
            output = F.softmax(text[:, -1], dim=-1)                             # [B, V]
            outputs.append(output)                                              # [[B, V]]
            output = output.argmax(-1, keepdim=True)                            # [B, 1]
            inputs = torch.cat((inputs, output), dim=1)                         # [B, T + 1]

            # set flag for early break
            output = output.squeeze(1)               # [B]
            current_end = output == self.eos_idx     # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        outputs = torch.stack(outputs, dim=1)                                   # [B, T, V]
        return outputs, lengths  # remove <sos>


class AttentionLSTMDecoder(BaseDecoder):

    def __init__(self, cfg: Config):
        super(AttentionLSTMDecoder, self).__init__()
        self.in_embed, self.out_embed = self.build_embedding(cfg)
        self.lstm = nn.LSTMCell(cfg.MODEL.TEXT_EMBED.EMBED_SIZE + cfg.MODEL.DECODER.HIDDEN_SIZE,
                                cfg.MODEL.DECODER.HIDDEN_SIZE)
        self.attention = DotProductAttention(scaled=True)
        self.teacher_forcing = cfg.MODEL.DECODER.TEACHER_FORCING
        self.sos_idx = cfg.MODEL.TEXT_EMBED.SOS_IDX
        self.eos_idx = cfg.MODEL.TEXT_EMBED.EOS_IDX

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        '''
        memory: (B, T, E)
        tgt: (B, T)
        '''
        hidden, cell = self._init_hidden(memory.size(0), memory.device)
        outputs = []

        for t in range(tgt.size(1)):
            context = self.attention(hidden.unsqueeze(1), memory, memory)[0]        # B, 1, E
            context = context.squeeze(1)                                            # B, H
            if self.teacher_forcing or t == 0:
                inputs = torch.cat((context, self.in_embed(tgt[:, t])), dim=-1)     # B, H + E
            else:
                inputs = torch.cat((context, hidden), dim=-1)   # B, E + V
            hidden, cell = self.lstm(inputs, (hidden, cell))                            # B, H
            output = self.out_embed(hidden)                                       # B, V
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def _init_hidden(self, batch_size: int, device: str) -> Tuple[Tensor, Tensor]:
        hidden_size = self.lstm.hidden_size
        h0 = torch.zeros(batch_size, hidden_size, device=device)
        c0 = torch.zeros(batch_size, hidden_size, device=device)
        return (h0, c0)

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        '''
        memory: (B, T, E)
        '''
        batch_size = memory.size(0)
        hidden, cell = self._init_hidden(batch_size, memory.device)
        inputs = torch.empty(batch_size, dtype=torch.long, device=memory.device).fill_(self.sos_idx)
        outputs = []

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            context = self.attention(hidden.unsqueeze(1), memory, memory)[0]        # B, 1, E
            context = context.squeeze(1)                                            # B, E
            if t == 0:
                inputs = torch.cat((context, self.in_embed(inputs)), dim=-1)        # B, E + V
            else:
                inputs = torch.cat((context, hidden), dim=-1)                       # B, E + V
            hidden, cell = self.lstm(inputs, (hidden, cell))                        # B, H
            output = self.out_embed(hidden)                                         # B, V
            output = F.softmax(output, dim=-1)                                      # B, V
            outputs.append(output)

            # set flag for early break
            output = output.argmax(-1)               # [B]
            current_end = output == self.eos_idx     # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break
        outputs = torch.stack(outputs, dim=1)
        return outputs, lengths


class VisualLSTMDecoder(BaseDecoder):
    def __init__(self, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            cfg.MODEL.DECODER.INPUT_SIZE,
            cfg.MODEL.DECODER.HIDDEN_SIZE,
            cfg.MODEL.DECODER.NUM_LAYERS,
            cfg.MODEL.DECODER.BIAS,
            cfg.MODEL.DECODER.BATCH_FIRST,
            cfg.MODEL.DECODER.DROPOUT,
            cfg.MODEL.DECODER.BIDIRECTIONAL,
        )

        NUM_DIRECTIONS = 2 if cfg.MODEL.DECODER.BIDIRECTIONAL else 1
        self.out = nn.Linear(
            NUM_DIRECTIONS * cfg.MODEL.DECODER.HIDDEN_SIZE,
            cfg.MODEL.DECODER.VOCAB_SIZE,
            cfg.MODEL.DECODER.OUT_BIAS,
        )
        self.decode = self.forward

    def forward(self, images):
        outputs, _ = self.lstm(images)                          # T, B, H*num_direction
        outputs = self.out(outputs)                             # T, B, V or B, T, V

        if self.lstm.batch_first:
            B, T = outputs.size(0), outputs.size(1)
        else:
            B, T = outputs.size(1), outputs.size(0)

        lengths = torch.empty(B, device=images.device, dtype=torch.long).fill_(T)

        return outputs, lengths
