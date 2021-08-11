import json
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F


class Vocab:

    r'''
    Base class for vocabulary representation.

    Arguments:
    - stoi: string-to-index dict.
    - unk: default for out-of-vocabulary token. It must be contained in `stoi`.
    If `None`, `KeyError` exception will be raised. Default is `None`
    '''

    def __init__(self,
                 stoi: Dict[str, int],
                 unk: Optional[str] = None):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        if unk is not None and unk not in stoi.keys():
            raise ValueError(f'{unk} must be contained in stoi')
        self.unk = unk

    def __len__(self):
        return len(self.stoi)

    def char2int(self, s: str) -> int:
        try:
            return self.stoi[s]
        except KeyError as e:
            if self.unk is not None:
                return self.stoi[self.unk]
            raise e

    def int2char(self, i: int) -> str:
        try:
            return self.itos[i]
        except KeyError as e:
            if self.unk is not None:
                return self.unk
            raise e

    def onehot(self, tokens: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(tokens, str):
            tokens = [tokens]
        token_idx = list(map(self.char2int, tokens))
        return F.one_hot(torch.tensor(token_idx), len(self))  # [T, V]

    @classmethod
    def from_json_stoi(cls, f, *args, **kwargs):
        stoi = json.load(f)
        return cls(stoi, *args, **kwargs)

    def __str__(self) -> str:
        return f'{self.stoi}'


class CTCVocab(Vocab):
    def __init__(self,
                 stoi: Dict[str, int],
                 unk: Optional[str] = None,
                 blank: str = '~'):
        super(CTCVocab, self).__init__(stoi, unk)
        if blank not in stoi.keys():
            raise ValueError(f'{blank} must be contained in stoi')
        self.blank = blank

    @property
    def BLANK(self):
        return self.blank

    @property
    def BLANK_IDX(self):
        return self.char2int(self.blank)


class Seq2SeqVocab(Vocab):
    def __init__(self,
                 stoi: Dict[str, int],
                 unk: Optional[str] = None,
                 sos: str = '<s>',
                 eos: str = '</s>',
                 pad: str = '<p>'):
        super(Seq2SeqVocab, self).__init__(stoi, unk)
        if sos not in stoi.keys():
            raise ValueError(f'{sos} must be contained in stoi')
        if eos not in stoi.keys():
            raise ValueError(f'{eos} must be contained in stoi')
        if pad not in stoi.keys():
            raise ValueError(f'{pad} must be contained in stoi')
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
        return self.__pad

    @property
    def PAD_IDX(self):
        return self.char2int(self.__pad)
