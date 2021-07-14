import logging
from io import StringIO
from typing import Dict, List, Optional

from torch.utils.tensorboard.writer import SummaryWriter


class LoggerInterface:

    def open(self):
        pass

    def close(self):
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        raise NotImplementedError()

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        raise NotImplementedError()


class NoLogger(LoggerInterface):

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        pass

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        pass


class ConsoleLogger(LoggerInterface):
    def __init__(self, log_interval: int, name: Optional[str] = None):
        self.logger = logging.getLogger(name or self.__name__)
        self.log_interval = log_interval
        self._internal_step = 0

    def open(self):
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


class TensorboardLogger(LoggerInterface):

    def __init__(self, log_dir: str = 'runs'):
        self._log_dir = log_dir

    def open(self):
        from datetime import datetime
        from pathlib import Path
        log_dir = Path(self._log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.logger = SummaryWriter(str(log_dir))

    def close(self):
        self.logger.close()

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self.logger.add_scalar(name, value, step)

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        self.logger.add_scalars(name, scalar_dict, step)


class ComposeLogger(LoggerInterface):

    def __init__(self, loggers: List[LoggerInterface]) -> None:
        self.loggers = loggers

    def open(self):
        for logger in self.loggers:
            logger.open()

    def close(self):
        for logger in self.loggers:
            logger.close()

    def log_scalar(self, name: str, value: float, step: Optional[int]):
        for logger in self.loggers:
            logger.log_scalar(name, value, step=step)

    def log_scalars(self, name: str, scalar_dict: Dict[str, float], step: Optional[int] = None):
        for logger in self.loggers:
            logger.log_scalars(name, scalar_dict, step=step)
