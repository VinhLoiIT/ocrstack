from typing import List, Mapping

import torch
import torch.nn.functional as F
from ocrstack.data.vocab import Seq2SeqVocab

__all__ = ['Replace', 'ToCharList',
           'OneHotEncoding', 'AddSeq2SeqTokens']


class Replace(object):
    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    def __call__(self, text: str) -> str:
        return ''.join([self.mapping[c] if c in self.mapping.keys() else c for c in list(text)])


class ToCharList(object):
    def __call__(self, text: str) -> List[str]:
        return list(text)


class OneHotEncoding(object):
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.one_hot(tensor, self.vocab_size).to(tensor.device)


class AddSeq2SeqTokens(object):
    def __init__(self, vocab: Seq2SeqVocab):
        self.sos_token = vocab.SOS
        self.eos_token = vocab.EOS

    def __call__(self, tokens: List[str]) -> List[str]:
        return [self.sos_token] + tokens + [self.eos_token]
