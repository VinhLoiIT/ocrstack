import logging
from typing import Dict, Optional, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.metrics.ocr import ACCMeter, CERMeter, WERMeter
from ocrstack.models.base import BaseModel
from torch.utils.data.dataloader import DataLoader


class Evaluator:
    def __init__(self,
                 model: BaseModel,
                 data_loader: DataLoader,
                 device,
                 metrics: Optional[Union[Dict, str]] = 'all',
                 log_interval: Optional[int] = None,
                 num_iter_eval: Optional[int] = None):
        if metrics is None:
            self.metrics = {}
        elif isinstance(metrics, Dict):
            self.metrics = metrics
        elif isinstance(metrics, str) and metrics == 'all':
            self.metrics = {
                'CER': CERMeter(),
                'NormCER': CERMeter(norm=True),
                'WER': WERMeter(),
                'NormWER': WERMeter(norm=True),
                'ACC': ACCMeter(),
            }
        else:
            raise ValueError(f'Unknow metrics = "{metrics}". You should pass a dict or "all" or set to None.')
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.log_interval = log_interval
        self.num_iter_eval = num_iter_eval or float('inf')

    @torch.no_grad()
    def eval(self):
        for metric in self.metrics.values():
            metric.reset()

        self.model.eval()
        batch: Batch
        for i, batch in enumerate(self.data_loader):
            batch = batch.to(self.device)
            predicts = self.model.predict(batch)
            for metric in self.metrics.values():
                metric.update(predicts, batch)

            if self.log_interval is not None and len(self.metrics) > 0 \
                    and ((i + 1) % self.log_interval == 0 or (i + 1) == len(self.data_loader)):
                logging.info('Metric at {:5d}/{}: {}'.format(
                    i + 1,
                    len(self.data_loader),
                    ' - '.join([f'{name}: {metric.compute():.04f}'
                                for name, metric in self.metrics.items()]),
                ))

            if (i + 1) >= self.num_iter_eval:
                break

        return {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }
