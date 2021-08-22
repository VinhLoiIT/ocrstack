from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from ocrstack.models.base import IS2SDecode
from torch import Tensor

from ..utils import generate_square_subsequent_mask


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

    # @torch.jit.export
    def decode_beamsearch(self, memory, max_length, beamsize, memory_key_padding_mask=None):
        # type: (Tensor, int, int, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        batch_size = memory.size(0)
        sos_inputs = torch.full((batch_size, 1), self.sos_idx,
                                dtype=torch.long, device=memory.device)                 # [B, L=1]

        end_flag = torch.zeros(batch_size, dtype=torch.bool,
                               device=memory.device)                                    # [B]

        inputs = (sos_inputs, torch.zeros(batch_size))                                                             # [B, K, L=1]
        batch_decoded = []
        all_candidates = [inputs]

        for _ in range(max_length + 1):
            next_candidate = []
            for (inputs, score) in all_candidates:
                # inputs: [B, L]
                # score: [B]
                end_flag = end_flag | (inputs[:, -1] == self.eos_idx)
                if end_flag.all():
                    batch_decoded.append((inputs, score))
                    continue

                outputs = self.forward(memory, inputs, memory_key_padding_mask)         # [B, L, V]
                outputs = outputs[:, -1]                                                # [B, V]
                log_probs = F.log_softmax(outputs, dim=-1)                              # [B, V]
                scores, indices = log_probs.topk(k=beamsize, dim=-1)                    # [B, K]

                scores = scores.masked_fill(end_flag.unsqueeze(-1), 0.0)
                indices = indices.masked_fill(end_flag.unsqueeze(-1), self.pad_idx)

                inputs = inputs.unsqueeze(-1)       # [B, L, 1]
                indices = indices.unsqueeze(1)      # [B, 1, K]
                inputs = torch.cat((inputs.repeat(1, 1, beamsize), indices), dim=1)     # [B, L, K]

                scores = score + scores                                                 # [B, K]

                for beam in range(beamsize):
                    next_candidate.append((inputs[..., beam], scores[..., beam]))

            next_candidate = sorted(next_candidate, key=lambda x: x[1], reverse=True)
            all_candidates = next_candidate[:beamsize]

        batch_decoded = batch_decoded + all_candidates
        batch_decoded = sorted(batch_decoded, key=lambda x: x[1], reverse=True)[:beamsize]
        batch, scores = zip(*batch_decoded)
        batch = [x.transpose(0, 1) for x in batch]  # [L, 1]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=self.pad_idx)  # [K, L, 1]
        batch = batch.permute(2, 0, 1)
        scores = torch.stack(scores, dim=-1)
        return batch, torch.exp(scores)


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
