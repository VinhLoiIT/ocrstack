import logging
import sys
from typing import Dict


class BaseLogger:
    def __init__(self):
        self.logger = logging.getLogger(self.__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handler = self.logger.addHandler(handler)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def log_metrics(self, metric_dict: Dict[str, float]):
        for name, value in metric_dict.items():
            self.info(f'{name} : {value:.04f}')
