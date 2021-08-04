from pathlib import Path
from typing import Dict

import torch
import queue


class ICkptSaver:

    def get_metric_value(self, metrics: Dict[str, Dict[str, float]]) -> float:
        raise NotImplementedError()

    def get_last_metric_value(self) -> float:
        raise NotImplementedError()

    def is_better(self, eval_metrics: Dict[str, Dict[str, float]]) -> bool:
        pass

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, Dict[str, float]]) -> None
        raise NotImplementedError()


class NullCkpt(ICkptSaver):

    def get_metric_value(self, metrics: Dict[str, Dict[str, float]]) -> float:
        pass

    def get_last_metric_value(self) -> float:
        pass

    def is_better(self, eval_metrics: Dict[str, Dict[str, float]]) -> bool:
        return False

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, Dict[str, float]]) -> None
        pass


class MonitorCkpt(ICkptSaver):

    def __init__(self, evaluator_name: str, metric_monitor: str, type_monitor: str, top: int) -> None:
        super().__init__()
        self._history: queue.Queue = queue.Queue(top)
        self.metric_monitor = metric_monitor
        self.type_monitor = type_monitor
        self.__last_metric_value = float('inf') if type_monitor == 'lower' else float('-inf')
        self.evaluator_name = evaluator_name

    def get_metric_value(self, metrics: Dict[str, Dict[str, float]]) -> float:
        return metrics[self.evaluator_name][self.metric_monitor]

    def get_last_metric_value(self) -> float:
        return self.__last_metric_value

    def is_better(self, eval_metrics: Dict[str, Dict[str, float]]) -> bool:
        if self.type_monitor == 'lower' and self.get_metric_value(eval_metrics) < self.get_last_metric_value():
            return True

        if self.type_monitor == 'higher' and self.get_metric_value(eval_metrics) > self.get_last_metric_value():
            return True

        return False

    def save(self, session_dir, trainer_state, eval_metrics):
        # type: (str, Dict, Dict[str, Dict[str, float]]) -> None
        current_metric_value = self.get_metric_value(eval_metrics)
        self.__last_metric_value = current_metric_value

        filename = f'{self.evaluator_name}_{self.metric_monitor}={current_metric_value:.04f}.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)

        try:
            self._history.put_nowait(checkpoint_path)
        except queue.Full:
            oldest_checkpoint = self._history.get_nowait()
            oldest_checkpoint.unlink()
            self._history.put_nowait(checkpoint_path)


class LastCkpt(ICkptSaver):

    def get_metric_value(self, metrics: Dict[str, Dict[str, float]]) -> float:
        return None

    def get_last_metric_value(self) -> float:
        return None

    def is_better(self, eval_metrics: Dict[str, Dict[str, float]]) -> bool:
        return True

    def save(self, session_dir: str, trainer_state: Dict, eval_metrics: Dict[str, Dict[str, float]]):
        filename = 'last.pth'
        checkpoint_path = Path(session_dir, 'ckpt', filename)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trainer_state, checkpoint_path)
