from ocrstack.data.collate import Batch
import editdistance as ed
from .metric import AverageMeter
from typing import List, Tuple


__all__ = [
    'CERMeter',
    'WERMeter',
    'ACCMeter',
    'compute_norm_cer',
    'compute_norm_wer',
    'compute_global_cer',
    'compute_global_wer',
    'compute_acc',
]


def compute_norm_cer(predicts, targets):
    # type: (List[str], List[str]) -> List[float]
    cers = [ed.distance(list(pred), list(tgt)) / len(tgt) for pred, tgt in zip(predicts, targets)]
    return cers


def compute_global_cer(predicts, targets):
    # type: (List[str], List[str]) -> Tuple[List[int], List[int]]
    dist = [ed.distance(list(pred), list(tgt)) for pred, tgt in zip(predicts, targets)]
    num_refs = [len(tgt) for tgt in targets]
    return dist, num_refs


def compute_norm_wer(predicts, targets):
    # type: (List[str], List[str]) -> List[float]
    wers = [ed.distance(pred.split(' '), tgt.split(' ')) / len(tgt.split(' '))
            for pred, tgt in zip(predicts, targets)]
    return wers


def compute_global_wer(predicts, targets):
    # type: (List[str], List[str]) -> Tuple[List[int], List[int]]
    dist = [ed.distance(pred.split(' '), tgt.split(' ')) for pred, tgt in zip(predicts, targets)]
    num_refs = [len(tgt.split(' ')) for tgt in targets]
    return dist, num_refs


def compute_acc(predicts, targets):
    # type: (List[str], List[str]) -> List[float]
    accs = [1 if pred == tgt else 0 for pred, tgt in zip(predicts, targets)]
    return accs


class CERMeter(AverageMeter):

    def __init__(self, norm: bool = False):
        super(CERMeter, self).__init__()
        self.norm = norm

    def update(self, predicts, batch):
        # type: (Tuple[List[str], List[float]], Batch) -> None
        '''
        Calculate CER distance between two lists of strings
        Params:
        -------
        - predicts: List of predicted characters
        - targets: List of target characters
        '''
        predicted_strings = predicts[0]
        target_strings = batch.text_str
        if self.norm:
            cers = compute_norm_cer(predicted_strings, target_strings)
            self.add(sum(cers), len(cers))
        else:
            dist, num_refs = compute_global_cer(predicted_strings, target_strings)
            self.add(sum(dist), sum(num_refs))


class WERMeter(AverageMeter):
    def __init__(self, spec_tokens: List[str] = [], split_word_token: str = ' ', norm: bool = False):
        super(WERMeter, self).__init__()
        self.split_word_token = split_word_token
        self.spec_tokens = spec_tokens
        self.norm = norm

    def update(self, predicts, batch):
        # type: (Tuple[List[str], List[float]], Batch) -> None
        '''
        Calculate WER distance between two lists of strings
        Params:
        -------
        - predicts: List of predicted characters
        - targets: List of target characters
        Returns:
        --------
        - distances: List of distances
        - n_references: List of the number of characters of targets
        '''
        predicted_strings = predicts[0]
        target_strings = batch.text_str
        if self.norm:
            wers = compute_norm_wer(predicted_strings, target_strings)
            self.add(sum(wers), len(wers))
        else:
            dist, num_refs = compute_global_wer(predicted_strings, target_strings)
            self.add(sum(dist), sum(num_refs))


class ACCMeter(AverageMeter):
    def __init__(self):
        super(ACCMeter, self).__init__()

    def update(self, predicts, batch):
        # type: (Tuple[List[str], List[float]], Batch) -> None
        '''
        Calculate Accuracy between two lists of strings
        Params:
        -------
        - predicts: List of predicted characters
        - targets: List of target characters
        '''
        predicted_strings = predicts[0]
        target_strings = batch.text_str
        accs = compute_acc(predicted_strings, target_strings)
        self.add(sum(accs), len(accs))
