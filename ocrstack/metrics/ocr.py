from typing import List, Tuple

import editdistance as ed

from .metric import AverageMeter

__all__ = [
    'GlobalCERMeter',
    'NormCERMeter',
    'GlobalWERMeter',
    'NormWERMeter',
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


def compute_norm_wer(predicts, targets, split_token=' '):
    # type: (List[str], List[str], str) -> List[float]
    wers = [ed.distance(pred.split(split_token), tgt.split(split_token)) / len(tgt.split(split_token))
            for pred, tgt in zip(predicts, targets)]
    return wers


def compute_global_wer(predicts, targets, split_token=' '):
    # type: (List[str], List[str], str) -> Tuple[List[int], List[int]]
    dist = [ed.distance(pred.split(split_token), tgt.split(split_token)) for pred, tgt in zip(predicts, targets)]
    num_refs = [len(tgt.split(split_token)) for tgt in targets]
    return dist, num_refs


def compute_acc(predicts, targets):
    # type: (List[str], List[str]) -> List[float]
    accs = [1.0 if pred == tgt else 0.0 for pred, tgt in zip(predicts, targets)]
    return accs


class GlobalCERMeter(AverageMeter):

    r"""Class handles Global CER Metric computation overtime.

    .. math::

        GlobalCER = \frac{ \sum_{i=1}^{N} ED(\hat{y}_{i},y_{i}) } { \sum_{i=1}^{N} |y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|` is the length of target tokens.
    """

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update Global CER

        Args:
            predicts: List of predicted strings
            targets: List of target strings
        """
        dist, num_refs = compute_global_cer(predicts, targets)
        self.add(sum(dist), sum(num_refs))


class NormCERMeter(AverageMeter):

    r"""Class handles Normalized CER Metric computation overtime.

    .. math::

        NormCER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|` is the length of target tokens.
    """

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update Normalized CER

        Args:
            predicts: List of predicted strings
            targets: List of target strings
        """
        cers = compute_norm_cer(predicts, targets)
        self.add(sum(cers), len(cers))


class GlobalWERMeter(AverageMeter):

    r"""Class handles Global WER Metric computation overtime.

    .. math::

        NormWER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|` is the length of target tokens.

    Args:
        split_word_token: a token to split words in a text string.
    """

    def __init__(self, split_word_token: str = ' '):
        super(GlobalWERMeter, self).__init__()
        self.split_word_token = split_word_token

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update Global WER

        Args:
            predicts: List of predicted strings
            targets: List of target strings
        """
        dist, num_refs = compute_global_wer(predicts, targets, self.split_word_token)
        self.add(sum(dist), sum(num_refs))


class NormWERMeter(AverageMeter):

    r"""Class handles Normalized WER Metric computation overtime.

    .. math::

        NormWER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|` is the length of target tokens.

    Args:
        split_word_token: a token to split words in a text string.
    """

    def __init__(self, split_word_token: str = ' '):
        super(NormWERMeter, self).__init__()
        self.split_word_token = split_word_token

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update Normalized WER

        Args:
            predicts: List of predicted strings
            targets: List of target strings
        """
        wers = compute_norm_wer(predicts, targets, self.split_word_token)
        self.add(sum(wers), len(wers))


class ACCMeter(AverageMeter):

    r"""Class handles Accuracy Metric computation overtime.

    .. math::

        ACC = \frac{ 1 } {N} \times \sum_{i=1}^{N} (\hat{y}_{i} \equiv y_{i})

    where :math:`N` is the number of samples of the dataset, :math:`\hat{y}` and :math:`y` are predicted
    and target tokens, respectively.
    """

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update ACC Metric

        Args:
            predicts: List of predicted strings
            targets: List of target strings
        """
        accs = compute_acc(predicts, targets)
        self.add(sum(accs), len(accs))
