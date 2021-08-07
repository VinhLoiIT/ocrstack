from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from torch import Tensor

from ..utils import generate_square_subsequent_mask


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


class TransformerDecoder(BaseDecoder):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self,
                 in_embed: nn.Module,
                 out_embed: nn.Module,
                 transformer_layer,
                 sos_idx: int,
                 eos_idx: int,
                 pad_idx: int,
                 num_layers: int = 1,
                 layer_norm: Optional[nn.LayerNorm] = None):
        super(TransformerDecoder, self).__init__()
        self.in_embed = in_embed
        self.out_embed = out_embed
        self.layers = _get_clones(transformer_layer, num_layers)
        self.layer_norm = layer_norm

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.num_layers = num_layers

    def forward(self, memory, tgt, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - memory: (B, S, E)
        - tgt: (B, T)

        Returns:
        --------
        - logits: (B, T, V)
        '''
        tgt_key_padding_mask = (tgt == self.pad_idx)
        tgt = self.in_embed(tgt)                    # [B, T, E]
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).unsqueeze(0).to(memory.device)
        memory_mask = None

        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

        if self.layer_norm:
            out = self.layer_norm(out)

        out = self.out_embed(out)                   # [B, T, V]
        return out

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        outputs: List[Tensor] = []
        inputs = torch.full((memory.size(0), 1), self.sos_idx, dtype=torch.long, device=memory.device)
        end_flag = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(max_length):
            output = self.forward(memory, inputs, memory_key_padding_mask)  # [B, T, V]
            output = F.softmax(output[:, [-1]], dim=-1)                     # [B, 1, V]
            outputs.append(output)                                          # [[B, 1, V]]
            output = output.argmax(-1, keepdim=False)                       # [B, 1]
            inputs = torch.cat((inputs, output), dim=1)                     # [B, T + 1]

            # set flag for early break
            output = output.squeeze(1)                                      # [B]
            current_end = output == self.eos_idx                            # [B]
            end_flag |= current_end
            if end_flag.all():
                break

        return torch.cat(outputs, dim=1)                                    # [B, T, V]


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


def _get_clones(layer: nn.Module, num_layers: int) -> nn.ModuleList:
    from copy import deepcopy
    layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    return layers
