import logging
import sys
from typing import List, Optional

from torch.utils.tensorboard.writer import SummaryWriter


class LoggerInterface:

    def open(self):
        pass

    def close(self):
        pass

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        raise NotImplementedError()


class NoLogger(LoggerInterface):

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        pass


class ConsoleLogger(LoggerInterface):
    def __init__(self, log_interval: int, name: Optional[str] = None):
        self.logger = logging.getLogger(name or self.__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.log_interval = log_interval
        self._internal_step = 0

    def open(self):
        self._internal_step = 0

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self._internal_step = step or self._internal_step
        if self._internal_step % self.log_interval == 0:
            self.logger.info(f'Step {self._internal_step:5d}: {name} = {value:.04f}')
        self._internal_step += 1


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
