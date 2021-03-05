from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class BaseDecoder(nn.Module):

    '''
    Base class for the Decoder component in Seq2Seq architecture

    All derivated classes from this class should perform:
    - Embedding text string from sequence of token indexes to the corresponding Tensor
    - Classify embedded tensor from embedded dimension to the size of vocabulary
    - Decoding a sequence from the source sequence
    '''

    def __init__(self, text_embedding: nn.Module, text_classifier: nn.Module,
                 sos_token_idx: Tensor, eos_token_idx: Tensor):
        super(BaseDecoder, self).__init__()
        self.text_embedding = text_embedding
        self.text_classifier = text_classifier
        self.register_buffer('sos_token_idx', sos_token_idx)
        self.register_buffer('eos_token_idx', eos_token_idx)


class TransformerDecoderAdapter(BaseDecoder):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''
    def __init__(self, text_embedding: nn.Module, text_classifier: nn.Module,
                 sos_token_idx: Tensor, eos_token_idx: Tensor, decoder: nn.TransformerDecoder):
        super(TransformerDecoderAdapter, self).__init__(text_embedding, text_classifier, sos_token_idx, eos_token_idx)
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

    def decode(self, memory, max_length):
        # type: (Tensor, int) -> Tuple[Tensor, Tensor]
        '''
        Arguments:
        ----------
        - memory: (B, S, E)

        Returns:
        --------
        - logits: (B, 1, V)
        '''
        batch_size = memory.size(0)

        assert isinstance(self.sos_token_idx, torch.Tensor)
        assert isinstance(self.eos_token_idx, torch.Tensor)

        predicts = self.sos_token_idx.unsqueeze(0).repeat(batch_size, 1, 1)     # [B, 1, V]
        ends = self.eos_token_idx.argmax(-1).repeat(batch_size).squeeze(-1)     # [B]

        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        lengths = torch.ones(batch_size, dtype=torch.long).fill_(max_length)
        for t in range(max_length):
            text = self.text_embedding(predicts)                # [B, T, E]
            text = self.forward(memory, text)                   # [B, T, V]
            output = F.softmax(text, dim=-1)                    # [B, 1, V]
            predicts = torch.cat([predicts, output], dim=1)     # [B, T + 1, V]

            # set flag for early break
            output = output.squeeze(1).argmax(-1)               # [B]
            current_end = output == ends                        # [B]
            current_end = current_end.cpu()
            lengths.masked_fill_(~end_flag & current_end, t + 1)
            end_flag |= current_end
            if end_flag.all():
                break

        return predicts, lengths


def _generate_square_subsequent_mask(sz: int) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
