import logging
from typing import Dict, List, Optional, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.metrics.metric import AverageMeter
from ocrstack.metrics.ocr import ACCMeter, CERMeter, WERMeter
from ocrstack.models.base import ITrainableModel
from ocrstack.models.layers.translator import ITranslator
from torch.utils.data.dataloader import DataLoader


class IEvaluator:

    def get_name(self) -> str:
        raise NotImplementedError()

    def eval(self, model, device=None):
        # type: (ITrainableModel, Optional[torch.Device]) -> Optional[Dict[str, float]]
        raise NotImplementedError()


class BaseEvaluator(IEvaluator):

    def __init__(self,
                 data_loader: DataLoader,
                 translator: Optional[ITranslator] = None,
                 num_iterations: Optional[int] = None) -> None:
        super().__init__()
        self.translator = translator
        self.data_loader = data_loader

        self.num_iterations = num_iterations
        if self.num_iterations is None:
            self.num_iterations = len(self.data_loader)

        self.logger = logging.getLogger(self.get_name())

    def get_name(self) -> str:
        raise NotImplementedError()

    def eval(self, model, device=None):
        # type: (ITrainableModel, Optional[torch.Device]) -> Optional[Dict[str, float]]
        raise NotImplementedError()


class LossEvaluator(IEvaluator):

    def __init__(self,
                 data_loader: DataLoader,
                 prefix: str = 'Validate',
                 log_interval: Optional[int] = None) -> None:
        super().__init__()
        self.prefix = prefix
        self.data_loader = data_loader

        self.log_interval = log_interval
        if log_interval is None:
            self.log_interval = len(self.data_loader)

    def get_name(self) -> str:
        return f'{self.prefix}/Loss'

    def eval(self, model, device=None):
        # type: (ITrainableModel, Optional[torch.Device]) -> Optional[Dict[str, float]]
        total_loss = AverageMeter()

        for i, batch in self.data_loader:
            loss = model.forward_batch(batch)
            total_loss.add(loss.item() * len(batch), len(batch))

            if (i + 1) % self.log_interval == 0:
                pass


class MetricsEvaluator(BaseEvaluator):
    def __init__(self,
                 name: str,
                 data_loader: DataLoader,
                 translator: Optional[ITranslator] = None,
                 num_iterations: Optional[int] = None,
                 metrics: Optional[Union[Dict, str]] = 'all',
                 log_interval: Optional[int] = None):
        self.name = name
        super().__init__(data_loader, translator, num_iterations)

        self.log_interval = log_interval
        if log_interval is None:
            self.log_interval = len(self.data_loader)

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

    def get_name(self):
        return self.name

    def eval(self, model, device=None):
        # type: (ITrainableModel, torch.Device) -> Optional[Dict[str, float]]
        for metric in self.metrics.values():
            metric.reset()

        model.eval()
        batch: Batch
        with torch.no_grad():
            for i, batch in enumerate(self.data_loader):
                batch = batch.to(device)
                predicts = model.predict_batch(batch)
                if self.translator is not None:
                    predicts = self.translator.translate(predicts)
                for metric in self.metrics.values():
                    metric.update(predicts, batch)

                if (i + 1) % self.log_interval == 0:
                    self.logger.info({
                        f'{metric_name}': metric.compute()
                        for metric_name, metric in self.metrics.items()
                    })

                if (i + 1) >= self.num_iterations:
                    break

        return {
            f'{metric_name}': metric.compute()
            for metric_name, metric in self.metrics.items()
        }


class VisualizeEvaluator(BaseEvaluator):
    def __init__(self,
                 data_loader: DataLoader,
                 translator: Optional[ITranslator] = None,
                 num_iterations: Optional[int] = None,
                 meta_fields: List[str] = []):
        super().__init__(data_loader, translator, num_iterations)
        self.meta_fields = meta_fields

    def get_name(self) -> str:
        return 'Visualizer'

    @torch.no_grad()
    def eval(self, model, device=None):
        # type: (ITrainableModel, torch.Device) -> Optional[Dict[str, float]]
        self.logger.info(f'Visualizing some predictions for {self.num_iterations} iteration(s)')
        self.logger.info('-' * 120)

        batch: Batch
        for i, batch in enumerate(self.data_loader):

            batch = batch.to(device)
            model_outputs = model.predict_batch(batch)
            predicts, _ = self.translator.translate(model_outputs)

            for metadata, text_str, predict in zip(batch.metadata, batch.text_str, predicts):
                s = ''
                for field in self.meta_fields:
                    s += f'{field}: {metadata[field]}, '
                s += f'Text: {text_str}, '
                s += f'Predict: {predict}'
                self.logger.info(s)

            if (i + 1) >= self.num_iterations:
                break
