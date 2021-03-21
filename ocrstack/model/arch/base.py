from ocrstack.metrics.metric import AverageMeter
from typing import Dict
from ocrstack.data.collate import CollateBatch
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, batch: CollateBatch):
        if self.training:
            return self.forward_train(batch)
        else:
            return self.forward_eval(batch)

    def forward_train(self, batch: CollateBatch):
        pass

    def forward_eval(self, batch: CollateBatch):
        pass

    def compute_batch_loss(self, batch, outputs):
        '''
        Computing loss
        :param outputs: outputs return by `forward_train` method
        :return: loss value as tensor. Please return pure tensor, not .item()
        '''
        pass

    def compute_batch_metrics(self, batch, outputs):
        '''
        Computing metrics
        :param outputs: outputs return by `forward_eval` method
        :return: metric dictionary. Please return pure tensor, not .item()
        '''
        pass

    def update_metrics(self, current_metrics, batch_metrics):
        # type: (Dict[str, AverageMeter], Dict[str, torch.Tensor]) -> None
        pass
