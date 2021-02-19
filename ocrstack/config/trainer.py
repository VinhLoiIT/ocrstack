from pathlib import Path
from typing import Optional
from .config import Config
from dataclasses import dataclass


@dataclass
class TrainerConfig(Config):
    seed: int
    num_workers: int
    batch_size: int
    lr: float
    device: str
    iter_train: int
    iter_eval: int
    iter_visualize: int
    num_iter_visualize: int
    checkpoint_dir: str
    log_file: str = 'log.txt'
    log_interval: int = 10
    monitor_metric: str = 'CER'
    monitor_metric_type: str = 'lower'
    pretrained_weight: Optional[Path] = None
    pretrained_config: Optional[Path] = None
    resume_checkpoint: Optional[Path] = None
    continue_training: bool = False
    use_amp: bool = False
    clip_grad_value: float = 0.5
