from typing import List, Tuple

import torch
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab, Vocab


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
        if self.return_raw:
            return ctc_translate_raw(predicts, self.vocab)
        else:
            return ctc_translate(predicts, self.vocab)


class Seq2SeqTranslator(ITranslator):

    def __init__(self, vocab, keep_eos=True, keep_sos=True, keep_pad=False):
        # type: (Seq2SeqVocab, bool, bool, bool) -> None
        self.vocab = vocab
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
        return seq2seq_translate(predicts, self.vocab, self.keep_sos, self.keep_eos, self.keep_pad)


def ctc_translate_raw(predicts, vocab):
    # type: (torch.Tensor, CTCVocab) -> Tuple[List[str], List[List[float]]]
    predicts = predicts.cpu()
    probs_, indices_ = predicts.max(dim=-1)
    strs = [''.join(map(vocab.int2char, indices)) for indices in indices_]
    return strs, probs_.tolist()


def ctc_translate(predicts, vocab):
    # type: (torch.Tensor, CTCVocab) -> Tuple[List[str], List[List[float]]]
    predicts = predicts.argmax(dim=-1)  # [B, T]
    blank_idx = vocab.BLANK_IDX
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

    samples = [''.join(map(vocab.int2char, result)) for result in results]
    return samples, probs


def seq2seq_translate(predicts: torch.Tensor,
                      vocab: Seq2SeqVocab,
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

    sos_mask = indices == vocab.SOS_IDX                     # [B, T]
    eos_mask = indices == vocab.EOS_IDX                     # [B, T]

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
        s = ''.join(map(vocab.int2char, indices_[start:end]))
        p = probs_[start:end]
        strings.append(s)
        char_probs.append(p)

    return strings, char_probs
