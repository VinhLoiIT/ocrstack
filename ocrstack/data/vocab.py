import json
from typing import Dict, List, Optional


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

    def lookup_index(self, token: str) -> int:
        r"""
        Convert a token (str) to a corresponding index (int). AKA char-to-int or stoi

        Unknow token (unk) will returned if token is not in this vocab

        Exceptions:
            KeyError: raised when token is not in this vocab and `unk` token is not set.
        """
        try:
            return self.stoi[token]
        except KeyError as e:
            if self.unk is not None:
                return self.stoi[self.unk]
            raise e

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return list(map(self.lookup_index, tokens))

    def lookup_token(self, index: int) -> str:
        try:
            return self.itos[index]
        except KeyError as e:
            if self.unk is not None:
                return self.unk
            raise e

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return list(map(self.lookup_token, indices))

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
        return self.lookup_index(self.blank)


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
        return self.lookup_index(self.__sos)

    @property
    def EOS(self):
        return self.__eos

    @property
    def EOS_IDX(self):
        return self.lookup_index(self.__eos)

    @property
    def PAD(self):
        return self.__pad

    @property
    def PAD_IDX(self):
        return self.lookup_index(self.__pad)
