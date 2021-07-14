from ocrstack.engine.logger import ConsoleLogger, NoLogger
from typing import Dict, Optional, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.metrics.ocr import ACCMeter, CERMeter, WERMeter
from ocrstack.models.base import BaseModel
from torch.utils.data.dataloader import DataLoader


class EvaluateInterface:

    def eval(self):
        raise NotImplementedError()


class Evaluator(EvaluateInterface):
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
        self.num_iter_eval = num_iter_eval or float('inf')
        self.logger = NoLogger()
        if log_interval is not None:
            self.logger = ConsoleLogger(log_interval, 'Evaluator')

    def eval(self):
        for metric in self.metrics.values():
            metric.reset()

        self.model.eval()
        batch: Batch
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                batch = batch.to(self.device)
                predicts = self.model.predict(batch)
                for metric in self.metrics.values():
                    metric.update(predicts, batch)

                self.logger.log_scalars('Evaluation', {
                    name: metric.compute()
                    for name, metric in self.metrics.items()
                })

                if (i + 1) >= self.num_iter_eval:
                    break

        return {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }


class ComposeEvaluator(EvaluateInterface):
    def __init__(self, evaluators: Dict[str, EvaluateInterface]) -> None:
        self.evaluators = evaluators

    def eval(self):
        val_metrics = {}
        for name, evaluator in self.evaluators.items():
            metrics = evaluator.eval()
            val_metrics[name] = metrics
        return val_metrics
