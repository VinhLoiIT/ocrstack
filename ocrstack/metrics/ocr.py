import editdistance as ed
from .metric import AverageMeter
from typing import List


__all__ = [
    'CERMeter',
    'WERMeter',
    'ACCMeter',
]


class CERMeter(AverageMeter):

    def __init__(self):
        super(CERMeter, self).__init__()

    def update(self, predicts: List[List[str]], targets: List[List[str]]):
        '''
        Calculate CER distance between two lists of strings
        Params:
        -------
        - predicts: List of predicted characters
        - targets: List of target characters
        '''
        total_distance = 0
        total_refs = 0
        for predict, target in zip(predicts, targets):
            total_distance += ed.distance(predict, target)
            total_refs += len(target)
        self.add(total_distance, total_refs)


class WERMeter(AverageMeter):
    def __init__(self, spec_tokens: List[str] = [], split_word_token: str = ' '):
        super(WERMeter, self).__init__()
        self.split_word_token = split_word_token
        self.spec_tokens = spec_tokens

    def update(self, predicts: List[List[str]], targets: List[List[str]]):
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
        total_distance = 0
        total_refs = 0
        for predict, target in zip(predicts, targets):
            for i in range(len(predict) - 1, -1, -1):
                if predict[i] in self.spec_tokens:
                    predict.insert(i + 1, self.split_word_token)
            predict_word_tokens = ''.join(predict).split(self.split_word_token)

            for i in range(len(target) - 1, -1, -1):
                if target[i] in self.spec_tokens:
                    target.insert(i + 1, self.split_word_token)
            target_word_tokens = ''.join(target).split(self.split_word_token)

            total_distance += ed.distance(predict_word_tokens, target_word_tokens)
            total_refs += len(target_word_tokens)
        self.add(total_distance, total_refs)


class ACCMeter(AverageMeter):
    def __init__(self):
        super(ACCMeter, self).__init__()

    def update(self, predicts: List[List[str]], targets: List[List[str]]):
        '''
        Calculate Accuracy between two lists of strings
        Params:
        -------
        - predicts: List of predicted characters
        - targets: List of target characters
        Returns:
        --------
        - distances: List of distances
        - n_references: List of the number of characters of targets
        '''
        true_predicts = 0
        total_lines = len(targets)
        for predict, target in zip(predicts, targets):
            predict_str = ''.join(predict)
            target_str = ''.join(target)
            true_predicts += 1 if predict_str == target_str else 0
        self.add(true_predicts, total_lines)
