from typing import Callable, List, Mapping

import torch
import torch.nn.functional as F
from ocrstack.data.vocab import Seq2SeqVocab, Vocab

__all__ = ['LabelDecoder', 'Replace', 'TextToTensor', 'ToCharList',
           'OneHotEncoding', 'AddSeq2SeqTokens']


class Replace(object):
    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    def __call__(self, text: str) -> str:
        return ''.join([self.mapping[c] if c in self.mapping.keys() else c for c in list(text)])


class ToCharList(object):
    def __call__(self, text: str) -> List[str]:
        return list(text)


class TextToTensor(object):
    def __init__(self, char2int: Callable):
        self.char2int = char2int

    def __call__(self, text: str):
        indexes = list(map(self.char2int, text))
        return torch.tensor(indexes, dtype=torch.long)


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


class LabelDecoder(object):
    def __init__(self, vocab: Vocab):
        self.int2char = vocab.int2char

    def decode_to_index(self, tensor: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        indexes: List[List[int]] = []
        for i, length in enumerate(lengths.cpu().tolist()):
            indexes.append(tensor[i, :length].tolist())
        return indexes

    def decode_to_tokens(self, tensor: torch.Tensor, lengths: torch.Tensor) -> List[List[str]]:
        indexes = self.decode_to_index(tensor, lengths)
        tokens = [list(map(self.int2char, sample)) for sample in indexes]
        return tokens

    def decode_to_string(self, tensor: torch.Tensor, lengths: torch.Tensor, join_char: str = '') -> List[str]:
        tokens_samples = self.decode_to_tokens(tensor, lengths)
        samples = [join_char.join(tokens) for tokens in tokens_samples]
        return samples
