import logging
from io import StringIO
from typing import Dict, List, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from ocrstack.models.base import ITrainableModel


class ILogger:

    def open(self, root_dir: str):
        pass

    def close(self):
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        raise NotImplementedError()

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        raise NotImplementedError()

    def log_model(self, model: ITrainableModel, device: str):
        raise NotImplementedError()

    def log_metrics(self, metrics, step=None):
        # type: (Dict[str, Dict[str, float]], str, Optional[int]) -> None
        raise NotImplementedError()

    def log_info(self, info: str):
        raise NotImplementedError()


class BaseLogger(ILogger):

    def log_metrics(self, metrics, prefix='', step=None):
        # type: (Dict[str, Dict[str, float]], str, Optional[int]) -> None
        for prefix, metric in metrics.items():
            for k, v in metric.items():
                self.log_scalar(f'{prefix}/{k}', v, step)


class NoLogger(BaseLogger):

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        pass

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        pass

    def log_model(self, model: ITrainableModel, device: str):
        pass

    def log_metrics(self, metrics, step=None):
        # type: (Dict[str, Dict[str, float]], Optional[int]) -> None
        pass

    def log_info(self, info: str):
        pass


class ConsoleLogger(BaseLogger):
    def __init__(self, log_interval: int, name: Optional[str] = None):
        self.logger = logging.getLogger(name or self.__class__.__name__)
        self.log_interval = log_interval
        self._internal_step = 0

    def open(self, root_dir: str):
        logging.basicConfig(format='[%(levelname)s] %(name)s: %(message)s', level=logging.INFO)
        self._internal_step = 0

    def _log(self, string: str, step):
        self._internal_step = step or self._internal_step
        if self._internal_step % self.log_interval == 0:
            self.logger.info(string)
        self._internal_step += 1

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self._log(f'Step {self._internal_step:5d}: {name} = {value:.04f}', step)

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        with StringIO() as writer:
            writer.write(f'Step {self._internal_step:5d}: ')
            writer.write(' - '.join([f'{name}: {value:.04f}' for name, value in scalar_dict.items()]))
            self._log(writer.getvalue(), step)

    def log_model(self, model: ITrainableModel, device: str):
        try:
            import torchinfo
            torchinfo.summary(model, input_data=model.example_inputs(), device=device)
        except ImportError:
            self.logger.info(model)

    def log_info(self, info: str):
        self.logger.info(info)


class TensorboardLogger(BaseLogger):

    def open(self, root_dir: str):
        from pathlib import Path
        log_dir = Path(root_dir, 'tb_logs')
        self.logger = SummaryWriter(str(log_dir))

    def close(self):
        self.logger.close()

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self.logger.add_scalar(name, value, step)

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        self.logger.add_scalars(name, scalar_dict, step)

    def log_model(self, model: ITrainableModel, device: str):
        inputs = model.example_inputs()
        if inputs is None:
            pass
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(device)
        elif isinstance(inputs, (list, tuple)):
            inputs = [x.to(device) for x in inputs]
        elif isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            raise RuntimeError('Unsupported example_inputs')
        self.logger.add_graph(model, input_to_model=inputs)

    def log_info(self, info: str):
        pass


class ComposeLogger(BaseLogger):

    def __init__(self, loggers: List[ILogger]) -> None:
        self.loggers = loggers

    def open(self, root_dir: str):
        for logger in self.loggers:
            logger.open(root_dir)

    def close(self):
        for logger in self.loggers:
            logger.close()

    def log_scalar(self, name: str, value: float, step: Optional[int]):
        for logger in self.loggers:
            logger.log_scalar(name, value, step=step)

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_scalars(name, scalar_dict, step=step)

    def log_model(self, model: ITrainableModel, device: str):
        for logger in self.loggers:
            logger.log_model(model, device)

    def log_metrics(self, metrics, step=None):
        # type: (Dict[str, Dict[str, float]], Optional[int]) -> None
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def log_info(self, info: str):
        for logger in self.loggers:
            logger.log_info(info)
