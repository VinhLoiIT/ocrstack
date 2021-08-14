from typing import List, Tuple

import editdistance as ed

from .metric import AverageMeter

__all__ = [
    'GlobalCERMeter',
    'NormCERMeter',
    'GlobalWERMeter',
    'NormWERMeter',
    'ACCMeter',
    'split_by_token',
    'compute_norm_cer',
    'compute_norm_wer',
    'compute_global_cer',
    'compute_global_wer',
    'compute_acc',
]


def split_by_token(tokens, split_token_index):
    # type: (List[str], str) -> List[List[str]]
    result = []
    start_pos = None
    for pos, token_index in enumerate(tokens):
        if start_pos is None and token_index != split_token_index:
            start_pos = pos
            continue
        if token_index == split_token_index and start_pos is not None:
            result.append(tokens[start_pos: pos])
            start_pos = None

    if start_pos is not None:
        result.append(tokens[start_pos:])

    return result


def compute_norm_cer(pred_tokens, tgt_tokens):
    # type: (List[str], List[str]) -> float
    cers = ed.distance(pred_tokens, tgt_tokens) / len(tgt_tokens)
    return cers


def compute_global_cer(pred_tokens, tgt_tokens):
    # type: (List[str], List[str]) -> Tuple[int, int]
    dist = ed.distance(pred_tokens, tgt_tokens)
    num_refs = len(tgt_tokens)
    return dist, num_refs


def compute_norm_wer(pred_tokens, tgt_tokens, split_token=' '):
    # type: (List[str], List[str], str) -> float
    pred_words = [''.join(word_tokens) for word_tokens in split_by_token(pred_tokens, split_token)]
    tgt_words = [''.join(word_tokens) for word_tokens in split_by_token(tgt_tokens, split_token)]
    wer = ed.distance(pred_words, tgt_words) / len(tgt_words)
    return wer


def compute_global_wer(pred_tokens, tgt_tokens, split_token=' '):
    # type: (List[str], List[str], str) -> Tuple[int, int]
    pred_words = [''.join(word_tokens) for word_tokens in split_by_token(pred_tokens, split_token)]
    tgt_words = [''.join(word_tokens) for word_tokens in split_by_token(tgt_tokens, split_token)]

    dist = ed.distance(pred_words, tgt_words)
    num_refs = len(tgt_words)
    return dist, num_refs


def compute_acc(pred_tokens, tgt_tokens):
    # type: (List[str], List[str]) -> float
    return 1.0 if pred_tokens == tgt_tokens else 0.0


class GlobalCERMeter(AverageMeter):

    r"""Class handles Global CER Metric computation overtime.

    .. math::

        GlobalCER = \frac{ \sum_{i=1}^{N} ED(\hat{y}_{i},y_{i}) } { \sum_{i=1}^{N} |y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|`
    is the length of target tokens.
    """

    def update(self, predicts, targets):
        # type: (List[List[str]], List[List[str]]) -> None
        r"""Update Global CER

        Args:
            predicts: List of predicted tokens
            targets: List of target tokens
        """
        dist, num_refs = zip(*[compute_global_cer(predict, target) for predict, target in zip(predicts, targets)])
        self.add(sum(dist), sum(num_refs))


class NormCERMeter(AverageMeter):

    r"""Class handles Normalized CER Metric computation overtime.

    .. math::

        NormCER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|`
    is the length of target tokens.
    """

    def update(self, predicts, targets):
        # type: (List[str], List[str]) -> None
        r"""Update Normalized CER

        Args:
            predicts: List of predicted tokens
            targets: List of target tokens
        """
        cers = [compute_norm_cer(predict, target) for predict, target in zip(predicts, targets)]
        self.add(sum(cers), len(cers))


class GlobalWERMeter(AverageMeter):

    r"""Class handles Global WER Metric computation overtime.

    .. math::

        NormWER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|`
    is the length of target tokens.

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
            predicts: List of predicted tokens
            targets: List of target tokens
        """
        dist, num_refs = zip(*[compute_global_wer(predict, target, self.split_word_token)
                               for predict, target in zip(predicts, targets)])
        self.add(sum(dist), sum(num_refs))


class NormWERMeter(AverageMeter):

    r"""Class handles Normalized WER Metric computation overtime.

    .. math::

        NormWER = \frac{ 1 } {N} \times \sum_{i=1}^{N} \frac {ED(\hat{y}_{i},y_{i}) } {|y_{i}|}

    where :math:`N` is the number of samples of the dataset, :math:`ED` is the Edit Distance (or Levenshtein Distance)
    of each predict :math:`\hat{y_{i}}` and target :math:`y_{i}` pair, and :math:`|y_{i}|`
    is the length of target tokens.

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
            predicts: List of predicted tokens
            targets: List of target tokens
        """
        wers = [compute_norm_wer(predict, target, self.split_word_token)
                for predict, target in zip(predicts, targets)]
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
            predicts: List of predicted tokens
            targets: List of target tokens
        """
        accs = [compute_acc(predict, target) for predict, target in zip(predicts, targets)]
        self.add(sum(accs), len(accs))
