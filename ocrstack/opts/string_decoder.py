import warnings
from typing import List, Tuple

import torch
import torch.nn as nn
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab


class StringDecoder(nn.Module):
    def __init__(self, vocab):
        super(StringDecoder, self).__init__()
        self.vocab = vocab


class CTCGreedyDecoder(StringDecoder):

    def forward(self, predicts):
        # type: (torch.Tensor,) -> Tuple[List[str], List[float]]
        '''
        Shapes:
        -------
        - predicts: (B, T, V)
        '''
        return ctc_greedy_decoder(predicts, self.vocab)


class CTCBeamsearchDecoder(StringDecoder):
    def __init__(self, vocab: CTCVocab, beamsize: int):
        super(CTCBeamsearchDecoder, self).__init__(vocab)
        self.beamsize = beamsize

    def forward(self, predicts):
        # type: (torch.Tensor,) -> Tuple[List[str], List[float]]
        return ctc_beamsearch_decoder(predicts, self.vocab, self.beamsize)


class Seq2SeqGreedyDecoder(StringDecoder):

    def forward(self, predicts, lengths):
        # type: (torch.Tensor, List[int]) -> Tuple[List[str], List[float]]
        '''
        Shapes:
        -------
        - predicts: (B, T, V)
        '''
        return seq2seq_greedy_decoder(predicts, lengths, self.vocab)


class Seq2SeqBeamsearchDecoder(StringDecoder):

    def __init__(self, model: nn.Module, vocab: Seq2SeqVocab, beamsize: int):
        super(Seq2SeqBeamsearchDecoder, self).__init__(vocab)
        self.beamsize = beamsize
        self.model = model

    def forward(self, predicts, lengths):
        # type: (torch.Tensor, List[int]) -> Tuple[List[str], List[float]]
        '''
        Shapes:
        -------
        - predicts: (B, T, V)
        '''
        raise NotImplementedError()


def ctc_greedy_decoder(predicts: torch.Tensor, vocab: CTCVocab):
    predicts = predicts.argmax(dim=-1)  # [B, T]
    blank_idx = vocab.BLANK_IDX
    results: List[List[int]] = []
    probs: List[float] = []
    for predict in predicts.cpu().tolist():
        # remove duplications
        predict = [predict[0]] + [c for i, c in enumerate(predict[1:]) if c != predict[i]]
        # remove 'blank'
        predict = list(filter(lambda i: i != blank_idx, predict))
        results.append(predict)

        # TODO: calculate prob here
        prob = 0.
        probs.append(prob)

    samples = [''.join(map(vocab.int2char, result)) for result in results]
    return samples, probs


def ctc_beamsearch_decoder(predicts: torch.Tensor, vocab: CTCVocab, beamsize: int):
    if beamsize == 1:
        warnings.warn('If beamsize is set to 1, it is prefered to use ctc_greedy_decoder because it is faster.')
    raise NotImplementedError()


def seq2seq_greedy_decoder(predicts: torch.Tensor, lengths: List[int], vocab: Seq2SeqVocab):
    probs, indices = predicts.max(dim=-1)  # [B, T]
    char_probs: List[List[float]] = []
    strings: List[str] = []
    for probs_, indices_, length in zip(probs.cpu().tolist(), indices.cpu().tolist(), lengths):
        strings.append(''.join(map(vocab.int2char, indices_[:length])))
        char_probs.append(probs_[:length])

    return strings, char_probs
