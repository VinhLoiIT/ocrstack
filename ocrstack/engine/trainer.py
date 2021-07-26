import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from ocrstack.engine.logger import ConsoleLogger, ILogger, ComposeLogger, TensorboardLogger
from ocrstack.engine.utils import set_seed
from ocrstack.metrics.metric import AverageMeter
from ocrstack.models.base import BaseModel
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.data.dataloader import DataLoader

from .checkpoint import ICkptSaver, LastCkpt, MonitorCkpt
from .evaluator import Evaluator
from .visualizer import Visualizer


class Trainer(object):

    '''
    Base class for OCR Trainer
    '''

    def __init__(self,
                 model: BaseModel,
                 optimizer: optim.Optimizer,
                 cfg: Config,
                 lr_scheduler=None,
                 evaluator: Optional[Evaluator] = None,
                 visualizer: Visualizer = None,
                 checkpoint_callback: Optional[ICkptSaver] = None,
                 logger: Optional[ILogger] = None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAINER.USE_AMP)
        self.evaluator = evaluator
        self.train_metrics = {
            'Loss': AverageMeter(),
            'Time': AverageMeter(),
        }

        if cfg.TRAINER.ITER_CHECKPOINT is None:
            checkpoint_callback = None
        elif checkpoint_callback is None:
            if cfg.TRAINER.MONITOR_METRIC is None:
                self.checkpoint_callback = LastCkpt()
            else:
                self.checkpoint_callback = MonitorCkpt(cfg.TRAINER.MONITOR_METRIC,
                                                       cfg.TRAINER.MONITOR_METRIC_TYPE)
        else:
            self.checkpoint_callback = checkpoint_callback

        self.visualizer = visualizer
        if logger is None:
            self.logger = ComposeLogger([
                ConsoleLogger(cfg.TRAINER.LOG_INTERVAL, name='Trainer'),
                TensorboardLogger(),
            ])
        else:
            self.logger = logger

    def train_step(self, batch: Batch):
        batch = batch.to(self.cfg.TRAINER.DEVICE)
        loss = self.model.train_batch(batch)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        if self.cfg.TRAINER.CLIP_GRAD_VALUE is not None:
            clip_grad_value_(self.model.parameters(), self.cfg.TRAINER.CLIP_GRAD_VALUE)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return loss.item()

    def train(self, train_loader: DataLoader):
        set_seed(self.cfg.TRAINER.SEED)

        self.model.to(self.cfg.TRAINER.DEVICE)
        self.model.train()

        session_dir = create_session_dir(self.cfg.TRAINER.LOG_DIR)
        self.logger.open(session_dir)
        self.logger.log_model(self.model, self.cfg.TRAINER.DEVICE)

        self._save_config(session_dir)
        self._warmup(train_loader)

        self.num_iteration = 0
        self.epoch = 0
        logging.info(f'Start training for {self.cfg.TRAINER.ITER_TRAIN} iteration(s)')
        while self.num_iteration < self.cfg.TRAINER.ITER_TRAIN:
            loss_meter = self.train_metrics['Loss']
            for i, batch in enumerate(train_loader):
                with torch.cuda.amp.autocast(enabled=self.cfg.TRAINER.USE_AMP):
                    loss = self.train_step(batch)
                loss_meter.update(loss, len(batch))
                self.num_iteration += 1

                self.logger.log_scalar('Train/Loss', loss, self.num_iteration)
                if self.num_iteration % self.cfg.TRAINER.LOG_INTERVAL == 0:
                    loss_meter.reset()

                if self.visualizer is not None and self.num_iteration % self.cfg.TRAINER.ITER_VISUALIZE == 0:
                    self.model.eval()
                    logging.info('Visualizing training process')
                    self.visualizer.visualize(self.model, self.cfg.TRAINER.DEVICE,
                                              self.cfg.TRAINER.NUM_ITER_VISUALIZE)
                    self.model.train()

                metrics: Dict[str, float] = {name: m.compute() for name, m in self.train_metrics.items()}

                if self.evaluator is not None and self.num_iteration % self.cfg.TRAINER.ITER_EVAL == 0:
                    metrics = self.evaluator.eval(self.model, self.cfg.TRAINER.NUM_ITER_EVAL,
                                                  self.cfg.TRAINER.DEVICE)
                    self.logger.log_metrics(metrics, False, self.num_iteration)
                    self.model.train()

                if self.checkpoint_callback is not None and self.num_iteration % self.cfg.TRAINER.ITER_CHECKPOINT == 0:
                    self.checkpoint_callback.save(session_dir, self.state_dict(), metrics)

                if self.num_iteration >= self.cfg.TRAINER.ITER_TRAIN:
                    break

            self.epoch += 1

        self.logger.close()

    def state_dict(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'num_iteration': self.num_iteration,
        }
        return state

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if state_dict['lr_scheduler'] is not None:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.epoch = state_dict['epoch']
        self.num_iteration = state_dict['num_iteration']

    def _save_config(self, session_dir: str):
        config_path = Path(session_dir, 'config.yaml')
        logging.info(f'Save config to {config_path}')
        self.cfg.to_yaml(config_path)

    def _warmup(self, train_loader):
        self.model.train()
        if self.cfg.TRAINER.NUM_ITER_WARMUP > 0:
            logging.info(f'Warmup trainer for {self.cfg.TRAINER.NUM_ITER_WARMUP} iteration(s)')
            for i, batch in enumerate(train_loader):
                self.train_step(batch)
                logging.debug(f'Warmed {i + 1} iteration(s)')
                if i + 1 == self.cfg.TRAINER.NUM_ITER_WARMUP:
                    break
            logging.info('Warmup trainer finished')


def create_session_dir(root_dir: str, name: Optional[str] = None, exist_ok: bool = False) -> str:
    from datetime import datetime
    from pathlib import Path
    if name is None:
        name = datetime.now().strftime('%Y%m%d-%H%M%S')

    log_dir = Path(root_dir, name)
    log_dir.mkdir(parents=True, exist_ok=exist_ok)
    return str(log_dir)
