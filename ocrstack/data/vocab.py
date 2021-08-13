import json
from typing import Dict, List, Optional, Tuple

import torch


class Vocab:

    r"""Base class for vocabulary representation.

    Arguments:
        stoi: string-to-index dict.
        unk: default token for the out-of-vocabulary token. It must be contained in `stoi`.
    """

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
        r"""Convert a token to its index.

        Args:
            token: a token to be converted.

        Returns:
            the index of `token`, or the index of `unk` if token is not in vocab.

        Raise:
            KeyError: raised when `token` is not in this vocab and `unk` token is not set.
        """
        try:
            return self.stoi[token]
        except KeyError as e:
            if self.unk is not None:
                return self.stoi[self.unk]
            raise e

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""A convenient method to convert a list of tokens to their index.

        Args:
            tokens: list of tokens to be converted.

        Returns:
            the indices of tokens
        """
        return list(map(self.lookup_index, tokens))

    def lookup_token(self, index: int) -> str:
        r"""Convert an index to the corresponding token.

        Args:
            index: an index to be converted.

        Returns:
            the `token` corresponding to `index` or `unk` if token not in vocab.

        Raise:
            KeyError: raised when `index` is not in this vocab and `unk` is not set.
        """
        try:
            return self.itos[index]
        except KeyError as e:
            if self.unk is not None:
                return self.unk
            raise e

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""A convenient method to convert indices to corresponding tokens.

        Args:
            indices: indices to be converted.

        Returns:
            the tokens of indices
        """
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

    r"""
    This class provides convenient methods and special token access for sequence-to-sequence training approach.

    To use this class, `sos`, `eos`, `pad` must be set and contained in `stoi`.

    Args:
        unk: Unknow token. This is used for out-of-vocabulary tokens. Default is `None`
        sos: Start of sequence token. Default is `<s>`
        eos: End of sequence token. Default is `</s>`
        pad: Padding token. Default is `<p>`
    """

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
    def SOS(self) -> str:
        r"""
        Shortcut for accessing the start of sequence token.
        """
        return self.__sos

    @property
    def SOS_IDX(self) -> int:
        r"""
        Shortcut for accessing the start of sequence index.
        """
        return self.lookup_index(self.__sos)

    @property
    def EOS(self) -> str:
        r"""
        Shortcut for accessing the end of sequence token.
        """
        return self.__eos

    @property
    def EOS_IDX(self) -> int:
        r"""
        Shortcut for accessing the end of sequence index.
        """
        return self.lookup_index(self.__eos)

    @property
    def PAD(self) -> str:
        r"""
        Shortcut for accessing the padding token.
        """
        return self.__pad

    @property
    def PAD_IDX(self) -> int:
        r"""
        Shortcut for accessing the padding index.
        """
        return self.lookup_index(self.__pad)

    def translate(self,
                  predicts: torch.Tensor,
                  reduce_token: str = '',
                  keep_sos: bool = False,
                  keep_eos: bool = False,
                  keep_pad: bool = False
                  ) -> Tuple[List[str], List[List[float]]]:
        r"""
        Translate a prediction tensor to its corresponding strings

        Args:
            predicts: a tensor of shape :math:`(B, L, V)` where :math:`B` is the batch size,
                :math:`L` is the sequence length, and :math:`V` is the vocab size.
            reduct_token: a token to concatenate over :math:`L` dim. Default is empty.
            keep_sos: whether to keep the start of sequence token after translation. Default is False.
            keep_eos: whether to keep the end of sequence token after translation. Default is False.
            keep_pad: whether to keep the padding token after translation. Default is False.

        Outputs:
            - strings: a list of :math:`B` strings
            - char_probs: a list of probabilities for each token in each string.
        """
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
        r"""Translate a prediction tensor arcording to CTC approach.

        Args:
            predicts: a tensor of shape :math:`(B, L, V)` where :math:`B` is the batch size,
                :math:`L` is the sequence length, and :math:`V` is the vocab size.
        """
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
        r"""Translate a prediction tensor arcording to sequence-to-sequence approach.

        Args:
            predicts: a tensor of shape :math:`(B, L, V)` where :math:`B` is the batch size,
                :math:`L` is the sequence length, and :math:`V` is the vocab size.
        """
        return self.vocab.translate(predicts, self.reduce_token, self.keep_sos, self.keep_eos, self.keep_pad)
