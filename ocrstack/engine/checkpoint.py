from pathlib import Path
from typing import Dict

import torch
import queue


class ICkptSaver:

    def is_better(self, eval_metrics: Dict[str, float]) -> bool:
        pass

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, float]) -> None
        raise NotImplementedError()


class NullCkpt(ICkptSaver):

    def is_better(self, eval_metrics: Dict[str, float]) -> bool:
        return False

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, float]) -> None
        pass


class MonitorCkpt(ICkptSaver):

    def __init__(self, metric_monitor: str, type_monitor: str, top: int) -> None:
        super().__init__()
        self._history: queue.Queue = queue.Queue(top)
        self.metric_monitor = metric_monitor
        self.type_monitor = type_monitor

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, float]) -> None
        filename = f'{self.metric_monitor}={eval_metrics[self.metric_monitor]:.04f}.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)

        try:
            oldest_checkpoint = self._history.get_nowait()
            oldest_checkpoint.unlink()
        except queue.Empty:
            pass

        self._history.put_nowait(checkpoint_path)


class LastCkpt(ICkptSaver):

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, float]):
        filename = 'last.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)
