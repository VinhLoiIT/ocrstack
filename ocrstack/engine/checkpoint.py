from pathlib import Path
from typing import Dict

import torch


class ICkptSaver:

    def is_better(self, eval_metrics: Dict[str, float]) -> bool:
        pass

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, float]):
        raise NotImplementedError()


class NullCkpt(ICkptSaver):

    def is_better(self, eval_metrics: Dict[str, float]) -> bool:
        return False

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, float]):
        pass


class MonitorCkpt(ICkptSaver):

    def __init__(self, metric_monitor: str, type_monitor: str) -> None:
        super().__init__()
        self.metric_monitor = metric_monitor
        self.type_monitor = type_monitor

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, float]):
        filename = f'{self.metric_monitor}={eval_metrics[self.metric_monitor]:.04f}.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)


class LastCkpt(ICkptSaver):

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, float]):
        filename = 'last.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)
