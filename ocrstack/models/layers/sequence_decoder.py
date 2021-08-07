from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from torch import Tensor

from ..utils import generate_square_subsequent_mask
from .positional_encoding import PositionalEncoding1d


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
                              bias=cfg.MODEL.TEXT_EMBED.OUT_BIAS)

        in_embed = nn.Embedding(cfg.MODEL.TEXT_EMBED.VOCAB_SIZE,
                                cfg.MODEL.TEXT_EMBED.EMBED_SIZE,
                                cfg.MODEL.TEXT_EMBED.PAD_IDX)

        if cfg.MODEL.TEXT_EMBED.SHARE_WEIGHT_IN_OUT:
            in_embed.weight = out_embed.weight

        return in_embed, out_embed


class TransformerDecoderAdapter(BaseDecoder):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self, cfg):
        super(TransformerDecoderAdapter, self).__init__()
        self.in_embed, self.out_embed = self.build_embedding(cfg)
        self.positional_encoding = self.build_positional_encoding(cfg)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(cfg.MODEL.DECODER.D_MODEL, cfg.MODEL.DECODER.NUM_HEADS),
            cfg.MODEL.DECODER.NUM_LAYERS
        )
        self.sos_idx = cfg.MODEL.TEXT_EMBED.SOS_IDX
        self.eos_idx = cfg.MODEL.TEXT_EMBED.EOS_IDX

    def build_positional_encoding(self, cfg: Config) -> nn.Module:
        if cfg.MODEL.TEXT_EMBED.POS_ENC_TYPE is None:
            return None

        if cfg.MODEL.TEXT_EMBED.POS_ENC_TYPE == 'sinusoidal':
            return PositionalEncoding1d(cfg.MODEL.TEXT_EMBED.EMBED_SIZE)

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

        if self.positional_encoding is not None:
            tgt = self.positional_encoding(tgt)     # [T, B, E]

        memory = memory.transpose(0, 1)             # [S, B, E]
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
        memory_mask = None
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = output.transpose(0, 1)                 # [B, T, E]
        output = self.out_embed(output)                 # [B, T, V]
        return output

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        batch_size = memory.size(0)
        inputs = torch.empty(batch_size, 1, dtype=torch.long, device=memory.device).fill_(self.sos_idx)
        outputs: List[Tensor] = [
            F.one_hot(inputs, num_classes=self.in_embed.num_embeddings).float().to(inputs.device)
        ]
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(max_length):
            text = self.forward(memory, inputs, memory_key_padding_mask, None)  # [B, T, V]
            output = F.softmax(text[:, [-1]], dim=-1)                           # [B, 1, V]
            outputs.append(output)                                              # [[B, 1, V]]
            output = output.argmax(-1, keepdim=False)                           # [B, 1]
            inputs = torch.cat((inputs, output), dim=1)                         # [B, T + 1]

            # set flag for early break
            output = output.squeeze(1)               # [B]
            current_end = output == self.eos_idx     # [B]
            current_end = current_end.cpu()
            end_flag |= current_end
            if end_flag.all():
                break

        return torch.cat(outputs, dim=1)                                   # [B, T, V]


class AttentionRecurrentDecoder(BaseDecoder):

    def __init__(self,
                 in_embed: nn.Module,
                 out_embed: nn.Module,
                 recurrent_layer,
                 sos_idx: int,
                 eos_idx: int,
                 num_layers: int = 1,
                 p_teacher_forcing: float = 1.):
        super(AttentionRecurrentDecoder, self).__init__()

        self.in_embed = in_embed
        self.out_embed = out_embed

        self.recurrent_layer = recurrent_layer
        self.p_teacher_forcing = p_teacher_forcing
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        assert num_layers == 1, 'Multilayer is not supported yet'
        self.num_layers = num_layers

    def forward(self, memory, tgt, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        '''
        memory: (B, T, E)
        tgt: (B, T)
        '''
        outputs: List[Tensor] = []

        prev_predict = torch.full((memory.size(0),), fill_value=self.sos_idx, dtype=torch.long, device=memory.device)
        prev_predict = self.in_embed(prev_predict)
        prev_attn = None
        prev_state = None

        for t in range(tgt.size(1)):
            out, context, state = self.recurrent_layer(memory, prev_predict, prev_attn,
                                                       prev_state, memory_key_padding_mask)
            output = self.out_embed(out)                                            # B, V
            outputs.append(output)                                                  # [[B, V]]

            teacher_forcing = (torch.rand(1) < self.p_teacher_forcing).item()
            if teacher_forcing:
                prev_predict = self.in_embed(tgt[:, t])
            else:
                prev_predict = self.in_embed(output.argmax(-1))
            prev_attn = context
            prev_state = state

        return torch.stack(outputs, dim=1)

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None) -> Tensor:
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        '''
        memory: (B, T, E)
        '''

        sos = torch.full((memory.size(0),), fill_value=self.sos_idx, dtype=torch.long, device=memory.device)
        prev_predict = self.in_embed(sos)
        prev_attn = None
        prev_state = None

        outputs: List[Tensor] = []

        end_flag = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for t in range(max_length):
            out, context, state = self.recurrent_layer(memory, prev_predict, prev_attn,
                                                       prev_state, memory_key_padding_mask)
            output = self.out_embed(out)                                # B, V

            prev_predict = self.in_embed(output.argmax(-1))             # B, E
            prev_attn = context
            prev_state = state

            output = F.softmax(output, dim=-1)                          # B, V
            outputs.append(output)                                      # [[B, V]]

            # set flag for early break
            output = output.argmax(-1)                                  # [B]
            current_end = output == self.eos_idx                        # [B]
            end_flag |= current_end
            if end_flag.all():
                break

        predicts = torch.stack(outputs, dim=1)                          # B, T, V
        one_hot_sos = F.one_hot(sos.unsqueeze(-1), predicts.size(-1)).to(predicts.device)
        predicts = torch.cat((one_hot_sos, predicts), dim=1)
        return torch.stack(outputs, dim=1)


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
