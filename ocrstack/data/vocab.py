from collections import Counter
from os import PathLike
from typing import List, Union

import torch
import torch.nn.functional as F


class VocabAdapter():
    def __init__(self):
        pass

    def char2int(self, char: str) -> int:
        pass

    def int2char(self, index: int) -> str:
        pass

    def __len__(self):
        pass


class _ListVocabAdapter(VocabAdapter):
    def __init__(self, alphabets: List[str], specials: List[str] = []):
        self.itos = specials + alphabets
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def char2int(self, char: str) -> int:
        return self.stoi[char]

    def int2char(self, index: int) -> str:
        return self.itos[index]

    def __len__(self) -> int:
        return len(self.itos)


class Vocab():
    def __init__(self,
                 vocab: Union[Counter, List, VocabAdapter],
                 specials: List[str] = []):
        self.adapter: VocabAdapter
        if isinstance(vocab, VocabAdapter):
            self.adapter = vocab
        elif isinstance(vocab, list):
            self.adapter = _ListVocabAdapter(vocab, specials)
        else:
            raise ValueError(f'Unknow type of vocab. type = {type(vocab)}')

        self.char2int = self.adapter.char2int
        self.int2char = self.adapter.int2char

    def __len__(self):
        return len(self.adapter)

    def onehot(self, tokens: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(tokens, str):
            tokens = [tokens]
        token_idx = list(map(self.char2int, tokens))
        return F.one_hot(torch.tensor(token_idx), len(self))  # [T, V]

    @classmethod
    def from_file(cls, vocab_path: PathLike):
        with open(vocab_path, 'rt', encoding='utf8') as f:
            alphabets = f.readline().strip()

        return cls(list(alphabets))

    @classmethod
    def from_csv(cls, vocab_path: PathLike):
        import csv
        with open(vocab_path, 'rt', encoding='utf8') as f:
            reader = csv.reader(f)
            alphabets = [line[0] for line in reader]

        return cls(alphabets)

    @classmethod
    def from_dataset(cls, dataset):
        char_set = set()
        for item in dataset:
            raw_text = item.get('text_str', '')
            char_set = char_set.union(set(raw_text))
        return cls(list(char_set))

    def __str__(self) -> str:
        return f'{self.adapter.stoi}'


class CTCVocab(Vocab):
    def __init__(self,
                 vocab: Union[Counter, List, VocabAdapter],
                 blank: str = '~'):
        super(CTCVocab, self).__init__(vocab, [blank])
        self.blank = blank

    @property
    def BLANK(self):
        return self.blank

    @property
    def BLANK_IDX(self):
        return self.char2int(self.blank)


class Seq2SeqVocab(Vocab):
    def __init__(self,
                 vocab: Union[Counter, List, VocabAdapter],
                 sos: str = '<sos>',
                 eos: str = '<eos>',
                 pad: str = '<pad>'):
        super(Seq2SeqVocab, self).__init__(vocab, [sos, eos, pad])
        self.__sos = sos
        self.__eos = eos
        self.__pad = pad

    @property
    def SOS(self):
        return self.__sos

    @property
    def SOS_IDX(self):
        return self.char2int(self.__sos)

    @property
    def EOS(self):
        return self.__eos

    @property
    def EOS_IDX(self):
        return self.char2int(self.__eos)

    @property
    def PAD(self):
        return self.__eos

    @property
    def PAD_IDX(self):
        return self.char2int(self.__pad)
