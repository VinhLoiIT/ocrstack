from typing import Dict, Optional, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.metrics.ocr import ACCMeter, CERMeter, WERMeter
from ocrstack.models.base import BaseModel
from ocrstack.models.layers.translator import ITranslator
from torch.utils.data.dataloader import DataLoader


class EvaluateInterface:

    def eval(self, model, num_iter_eval, device):
        # type: (BaseModel, int, torch.Device) -> Dict[str, float]
        raise NotImplementedError()


class Evaluator(EvaluateInterface):
    def __init__(self,
                 data_loader: DataLoader,
                 translator: Optional[ITranslator] = None,
                 metrics: Optional[Union[Dict, str]] = 'all'):

        self.translator = translator
        self.data_loader = data_loader

        if isinstance(metrics, Dict):
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
            raise ValueError(f'Unknow metrics = "{metrics}". You should pass a dict or "all".')

    def eval(self, model, num_iter_eval, device):
        # type: (BaseModel, int, torch.Device) -> Dict[str, float]
        for metric in self.metrics.values():
            metric.reset()

        model.eval()
        batch: Batch
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                batch = batch.to(device)
                predicts = model.predict(batch)
                if self.translator is not None:
                    predicts = self.translator.translate(predicts)
                for metric in self.metrics.values():
                    metric.update(predicts, batch)

                if (i + 1) >= num_iter_eval:
                    break

        return {
            name: metric.compute()
            for name, metric in self.metrics.items()
        }


class ComposeEvaluator(EvaluateInterface):
    def __init__(self, evaluators: Dict[str, EvaluateInterface]) -> None:
        self.evaluators = evaluators

    def eval(self, model, num_iter_eval, device):
        # type: (BaseModel, int, torch.Device) -> Dict[str, float]
        val_metrics = {}
        for name, evaluator in self.evaluators.items():
            metrics = evaluator.eval(model, num_iter_eval, device)
            for metric_name, metric_value in metrics:
                val_metrics[name + '/' + metric_name] = metric_value
        return val_metrics
