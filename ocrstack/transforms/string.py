from typing import Callable, List, Mapping, Optional

import torch
import torch.nn.functional as F
from ocrstack.data.vocab import Seq2SeqVocab, Vocab

__all__ = ['LabelDecoder', 'Replace', 'TextToTensor', 'ToCharList',
           'OneHotEncoding', 'AddSeq2SeqTokens', 'BatchPadTexts']


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


class BatchPadTexts:
    def __init__(self, pad_value, max_length=None):
        # type: (torch.Tensor, Optional[int]) -> None
        assert max_length is None or max_length > 0
        self.max_length = max_length or 0
        self.pad_value = torch.as_tensor(pad_value)

    def __call__(self, texts: List[torch.Tensor]):
        assert len(texts) > 0
        lengths: List[int] = [text.size(0) for text in texts]
        max_length = self.max_length or max(*lengths, self.max_length)

        if len(texts) == 1:
            return texts[0].unsqueeze(0)

        batch_shape = [len(texts), max_length] + list(texts[0].shape[1:])
        batched_text = torch.empty(batch_shape, dtype=torch.long)
        for i, t in enumerate(texts):
            batched_text[i, :t.shape[0], ...].copy_(t)
            batched_text[i, t.shape[0]:, ...].copy_(self.pad_value)

        return batched_text
