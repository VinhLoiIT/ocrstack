import json
from typing import Dict, List, Optional, Tuple

import torch


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

    def translate(self,
                  predicts: torch.Tensor,
                  return_raw: bool = False
                  ) -> Tuple[List[str], List[List[float]]]:
        if return_raw:
            predicts = predicts.cpu()
            probs_, indices_ = predicts.max(dim=-1)
            strs = [''.join(self.lookup_tokens(indices)) for indices in indices_]
            return strs, probs_.tolist()

        predicts = predicts.argmax(dim=-1)  # [B, T]
        blank_idx = self.BLANK_IDX
        results: List[List[int]] = []
        probs: List[List[float]] = []
        for predict in predicts.cpu().tolist():
            # remove duplications
            predict = [predict[0]] + [c for i, c in enumerate(predict[1:]) if c != predict[i]]
            # remove 'blank'
            predict = list(filter(lambda i: i != blank_idx, predict))
            results.append(predict)

            # TODO: calculate prob here
            prob = [0.]
            probs.append(prob)

        samples = [''.join(self.lookup_tokens(result)) for result in results]
        return samples, probs


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

    def translate(self,
                  predicts: torch.Tensor,
                  reduce_token: str = '',
                  keep_sos: bool = False,
                  keep_eos: bool = False,
                  keep_pad: bool = False
                  ) -> Tuple[List[str], List[List[float]]]:
        '''
        Arguments:
        ----------
        predicts: [B, T, V]
        '''
        predicts = predicts.cpu()

        probs, indices = predicts.max(dim=-1)                   # [B, T]

        sos_mask = indices == self.SOS_IDX                     # [B, T]
        eos_mask = indices == self.EOS_IDX                     # [B, T]

        sos_pos = sos_mask.max(-1)[1]                           # [B]
        eos_pos = eos_mask.max(-1)[1]                           # [B]

        # TODO: implement keep_pad

        if not keep_sos:
            sos_pos += 1
        sos_pos.masked_fill_(torch.bitwise_not(torch.any(sos_mask, dim=-1)), 0)

        if keep_eos:
            eos_pos += 1
        eos_pos.masked_fill_(torch.bitwise_not(torch.any(eos_mask, dim=-1)), predicts.size(1))

        char_probs: List[List[float]] = []
        strings: List[str] = []

        for probs_, indices_, start, end in zip(probs.tolist(), indices.tolist(), sos_pos, eos_pos):
            s = reduce_token.join(self.lookup_tokens(indices_[start:end]))
            p = probs_[start:end]
            strings.append(s)
            char_probs.append(p)

        return strings, char_probs


class ITranslator:

    def translate(self, predicts: torch.Tensor) -> Tuple[List[str], List[List[float]]]:
        raise NotImplementedError()


class CTCTranslator(ITranslator):

    def __init__(self, vocab, return_raw=False):
        # type: (CTCVocab, bool) -> None
        self.vocab = vocab
        self.return_raw = return_raw

    def translate(self, predicts):
        # type: (torch.Tensor,) -> Tuple[List[str], List[List[float]]]
        '''
        Shapes:
        -------
        - predicts: (B, T, V)
        '''
        return self.vocab.translate(predicts, self.return_raw)


class Seq2SeqTranslator(ITranslator):

    def __init__(self, vocab, reduce_token, keep_sos=True, keep_eos=True, keep_pad=False):
        # type: (Seq2SeqVocab, str, bool, bool, bool) -> None
        self.vocab = vocab
        self.reduce_token = reduce_token
        self.keep_sos = keep_sos
        self.keep_eos = keep_eos
        self.keep_pad = keep_pad

    def translate(self, predicts):
        # type: (torch.Tensor,) -> Tuple[List[str], List[List[float]]]
        '''
        Shapes:
        -------
        - predicts: (B, T, V)
        '''
        return self.vocab.translate(predicts, self.reduce_token, self.keep_sos, self.keep_eos, self.keep_pad)
