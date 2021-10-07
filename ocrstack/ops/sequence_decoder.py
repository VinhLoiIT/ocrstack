from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import generate_square_subsequent_mask


class IS2SDecode:

    r"""This is a common interface for all models based on sequence-to-sequence approach.

    The difference to :class:`ICTCDecode` is the `max_length` parameter in decode functions to
    avoid infinitely decoding.
    """

    def decode_greedy(self, images, max_length, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        r"""Greedy Sequence-To-Sequence decoding

        In fact, it behaves like beamsearch decoding where :code:`beamsize=1` but faster since it does
        not populate a queue to store temporal decoding steps.

        Args:
            images: a tensor of shape :math:`(B, C, H, W)` containing the images
            max_length: a maximum length :math:`L` to decode
            image_mask: a tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a 2-element tuple containing prediction indices and probabilities.

            - **indices**: a tensor of shape :math:`(B, L + 2)` where :math:`B` is the batch size, :math:`L` is
              the `max_length`. It should contain both `sos` and `eos` signals.
            - **probs**: a tensor of shape :math:`(B,)` where :math:`B` is the batch size.

        """
        raise NotImplementedError()


class ICTCDecode:

    r"""This is a common interface for all models based on CTC approach.

    The difference to :class:`IS2SDecode` is `max_length` parameter in decode functions to
    avoid infinitely decoding.
    """

    def decode_greedy(self, images, image_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        r"""Greedy CTC decoding

        In fact, it behaves like beamsearch decoding where :code:`beamsize=1` but faster since it does
        not populate a queue to store temporal decoding steps.

        Args:
            images: a tensor of shape :math:`(B, C, H, W)` containing the images
            image_mask: a tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a 2-element tuple containing prediction indices and probabilities. `1` is for conventional to beamsearch
            decoding's outputs.

            - **indices**: a tensor of shape :math:`(B, 1, L + 2)` where :math:`B` is the batch size, :math:`L` is
              the `max_length`. It should contain both `sos` and `eos` signals.
            - **probs**: a tensor of shape :math:`(B, 1)` where :math:`B` is the batch size.

        """
        raise NotImplementedError()

    def decode_beamsearch(self, images, beamsize, image_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        r"""Beamsearch CTC decoding

        Args:
            images: a Tensor of shape :math:`(B, C, H, W)` containing the images
            beamsize: the number of beam for beamsearch algorithms
            image_mask: a Tensor of shape :math:`(B, H, W)` to indicate images content within a batch.

        Return:
            a 2-element tuple containing prediction indices and probabilities.

            - **indices**: a tensor of shape :math:`(B, K, L + 2)` where :math:`B` is the batch size, :math:`K` is the
              beamsize, and :math:`L` is the `max_length`. It should contain both `sos` and `eos` signals.
            - **probs**: a tensor of shape :math:`(B, K)` where :math:`B` is the batch size, :math:`K` is the beamsize.

        """
        raise NotImplementedError()


class TransformerDecoder(nn.Module, IS2SDecode):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self,
                 text_embed: nn.Module,
                 fc: nn.Module,
                 transformer_layer,
                 sos_idx: int,
                 eos_idx: int,
                 pad_idx: int,
                 num_layers: int = 1,
                 layer_norm: Optional[nn.LayerNorm] = None):
        super(TransformerDecoder, self).__init__()
        self.text_embed = text_embed
        self.fc = fc
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
        tgt = self.text_embed(tgt)                    # [B, T, E]
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).unsqueeze(0).to(memory.device)
        memory_mask = None

        out = tgt
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)

        if self.layer_norm is not None:
            out = self.layer_norm(out)

        out = self.fc(out)                   # [B, T, V]
        return out

    @torch.jit.export
    def decode_greedy(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = memory.size(0)
        sos_inputs = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=memory.device)  # [B, 1]
        scores = torch.zeros(batch_size, device=memory.device)                                          # [B]

        inputs = sos_inputs                                                                             # [B, T=1]
        end_flag = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)                      # [B]
        for _ in range(max_length + 1):
            output = self.forward(memory, inputs, memory_key_padding_mask)                              # [B, T, V]
            output = F.log_softmax(output[:, [-1]], dim=-1)                                             # [B, 1, V]
            score, index = output.max(dim=-1)                                                           # [B, 1]
            scores = scores + score.squeeze(1)                                                          # [B]
            inputs = torch.cat((inputs, index), dim=1)                                                  # [B, T+1]

            # early break
            end_flag = end_flag | (index == self.eos_idx)                                               # [B]
            if end_flag.all():
                break

        return inputs, torch.exp(scores)                                                                # [B, T], [B]


class AttentionRecurrentDecoder(nn.Module):

    def __init__(self,
                 text_embed: nn.Module,
                 fc: nn.Module,
                 recurrent_layer,
                 sos_idx: int,
                 eos_idx: int,
                 num_layers: int = 1,
                 p_teacher_forcing: float = 1.):
        super(AttentionRecurrentDecoder, self).__init__()

        self.text_embed = text_embed
        self.fc = fc

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
        prev_predict = self.text_embed(prev_predict)
        prev_attn = None
        prev_state = None

        for t in range(tgt.size(1)):
            out, context, state = self.recurrent_layer(memory, prev_predict, prev_attn,
                                                       prev_state, memory_key_padding_mask)
            output = self.fc(out)                                            # B, V
            outputs.append(output)                                                  # [[B, V]]

            teacher_forcing = (torch.rand(1) < self.p_teacher_forcing).item()
            if teacher_forcing:
                prev_predict = self.text_embed(tgt[:, t])
            else:
                prev_predict = self.text_embed(output.argmax(-1))
            prev_attn = context
            prev_state = state

        return torch.stack(outputs, dim=1)

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        '''
        memory: (B, T, E)
        '''

        sos = torch.full((memory.size(0),), fill_value=self.sos_idx, dtype=torch.long, device=memory.device)
        prev_predict = self.text_embed(sos)
        prev_attn = None
        prev_state = None

        outputs: List[Tensor] = []

        end_flag = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for t in range(max_length + 1):
            out, context, state = self.recurrent_layer(memory, prev_predict, prev_attn,
                                                       prev_state, memory_key_padding_mask)
            output = self.fc(out)                                # B, V

            prev_predict = self.text_embed(output.argmax(-1))             # B, E
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


class VisualLSTMDecoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size: int,
                 vocab_size: int,
                 num_layers: int,
                 bias: bool = True,
                 dropout: Optional[float] = None,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        NUM_DIRECTIONS = 2 if bidirectional else 1
        self.out = nn.Linear(
            NUM_DIRECTIONS * hidden_size,
            vocab_size,
            bias=bias,
        )
        self.decode = self.forward

    def forward(self, images):
        r"""
        Args:
            images: a tensor of (T, B, E)
        """
        outputs, _ = self.lstm(images)                          # T, B, H*num_direction
        outputs = self.out(outputs)                             # T, B, V or B, T, V
        B, T = outputs.size(0), outputs.size(1)

        lengths = torch.empty(B, device=images.device, dtype=torch.long).fill_(T)

        return outputs, lengths


def _get_clones(layer: nn.Module, num_layers: int) -> nn.ModuleList:
    from copy import deepcopy
    layers = nn.ModuleList([deepcopy(layer) for _ in range(num_layers)])
    return layers
