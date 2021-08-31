import logging
import queue
from pathlib import Path
from typing import Any

import torch


class CheckpointSaver:

    def __init__(
        self,
        save_dir: Path,
        metric_monitor: str,
        type_monitor: str,
        top: int,
        prefix: str = '',
    ) -> None:

        self._history: queue.Queue = queue.Queue(top)
        self.metric_monitor = metric_monitor
        self.type_monitor = type_monitor

        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float('inf') if type_monitor == 'lower' else float('-inf')
        self.prefix = prefix
        self.logger = logging.getLogger('CheckpointSaver')

    def update(self, new_value, to_save):
        if self.is_better(new_value):
            self.logger.info('Found better value. %s improved from %.4f to %.4f',
                             self.metric_monitor,
                             self.best_value,
                             new_value)
            self.best_value = new_value

            filename = f'{self.prefix}{self.metric_monitor}={self.best_value:.06f}.pth'
            checkpoint_path = self.save(filename, to_save)
            self.logger.debug('Saved checkpoint to %s', str(checkpoint_path))

            best_ckpt_path = self.save_dir / f'best_{self.metric_monitor}.pth'
            best_ckpt_path.unlink(missing_ok=True)
            best_ckpt_path.symlink_to(filename, target_is_directory=False)

            try:
                self._history.put_nowait(checkpoint_path)
            except queue.Full:
                oldest_checkpoint: Path = self._history.get_nowait()
                oldest_checkpoint.unlink(missing_ok=True)
                self._history.put_nowait(checkpoint_path)

    def is_better(self, new_value: float) -> bool:
        if self.type_monitor == 'lower' and new_value < self.best_value:
            return True

        if self.type_monitor == 'higher' and new_value > self.best_value:
            return True

        return False

    def save(self, filename, to_save):
        # type: (str, Any) -> Path
        save_path = self.save_dir / filename
        self.logger.debug('Saving checkpoint to %s', str(save_path))
        torch.save(to_save, save_path)
        return save_path
