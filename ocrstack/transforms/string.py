from typing import Callable, List, Mapping

import torch
import torch.nn.functional as F
from ocrstack.core.typing import Tokens
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab, Vocab
from torch import Tensor

__all__ = ['LabelDecoder', 'Replace', 'CTCDecoder', 'TextToTensor']


class Replace(object):
    def __init__(self, mapping: Mapping):
        self.mapping = mapping

    def __call__(self, text: str) -> str:
        return ''.join([self.mapping[c] if c in self.mapping.keys() else c for c in list(text)])


class ToCharList(object):
    def __call__(self, text: str) -> Tokens:
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

    def __call__(self, tokens: Tokens) -> Tokens:
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


class CTCDecoder(object):
    def __init__(self, vocab: CTCVocab):
        self.vocab = vocab

    def decode_to_index(self, tensor: torch.Tensor) -> List[List[int]]:
        '''
        Convert a Tensor to a list of token indexes if return_string is False, otherwise string.
        '''
        indexes = ctc_decode(tensor, self.vocab.BLANK_IDX)
        return indexes

    def decode_to_tokens(self, tensor: torch.Tensor) -> List[List[str]]:
        indexes = self.decode_to_index(tensor)
        tokens = [list(map(self.vocab.int2char, sample)) for sample in indexes]
        return tokens

    def decode_to_string(self, tensor: torch.Tensor, join_char: str = '') -> List[str]:
        tokens_samples = self.decode_to_tokens(tensor)
        samples = [join_char.join(tokens) for tokens in tokens_samples]
        return samples


def ctc_decode(tensor: Tensor, blank_index: int) -> List[List[int]]:
    results: List[List[int]] = []
    for sample in tensor.cpu().tolist():
        # remove duplications
        sample = [sample[0]] + [c for i, c in enumerate(sample[1:]) if c != sample[i]]
        # remove 'blank'
        sample = list(filter(lambda i: i != blank_index, sample))
        results.append(sample)
    return results
