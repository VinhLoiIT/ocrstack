import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.optim as optim
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from ocrstack.engine.logger import ConsoleLogger, LoggerInterface
from ocrstack.metrics.metric import AverageMeter
from ocrstack.models.base import BaseModel
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.utils.data.dataloader import DataLoader

from .checkpoint import CkptSaver
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
                 evaluator: Union[Evaluator, List[Evaluator]] = [],
                 visualizer: Visualizer = None,
                 checkpoint_callback: Optional[CkptSaver] = None,
                 logger: Optional[LoggerInterface] = None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAINER.USE_AMP)
        if isinstance(evaluator, Evaluator):
            self.evaluators = [evaluator]
        else:
            self.evaluators = evaluator
        self.train_metrics = {
            'Loss': AverageMeter(),
            'Time': AverageMeter(),
        }

        # self.checkpoint_saver = CkptSaver(Path(config.checkpoint_dir), exist_ok=True)
        self.checkpoint_callback = checkpoint_callback
        self.visualizer = visualizer
        if logger is None:
            self.logger = ConsoleLogger(cfg.TRAINER.LOG_INTERVAL, name='Trainer')
        else:
            self.logger = logger

    def train_step(self, batch: Batch):
        batch = batch.to(self.cfg.TRAINER.DEVICE)
        loss = self.model.train_batch(batch)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        clip_grad_value_(self.model.parameters(), self.cfg.TRAINER.CLIP_GRAD_VALUE)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return loss.item()

    def train(self, train_loader: DataLoader):
        self.model.to(self.cfg.TRAINER.DEVICE)
        self.model.train()

        self.logger.open()
        self.logger.log_model(self.model, self.cfg.TRAINER.DEVICE)

        self._save_config()
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
                    self.visualizer.visualize()
                    self.model.train()

                train_metrics = {name: m.compute() for name, m in self.train_metrics.items()}
                if len(self.evaluators) > 0 and self.num_iteration % self.cfg.TRAINER.ITER_EVAL == 0:
                    val_metrics = [evaluator.eval() for evaluator in self.evaluators]
                    if self.checkpoint_callback is not None:
                        self.checkpoint_callback(self.state_dict(), train_metrics, val_metrics)
                    self.model.train()
                if self.num_iteration >= self.cfg.TRAINER.ITER_TRAIN:
                    break

            self.epoch += 1

        self.logger.close()

    def state_dict(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'num_iteration': self.num_iteration,
        }
        return state

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.epoch = state_dict['epoch']
        self.num_iteration = state_dict['num_iteration']

    def _save_config(self):
        config_path = Path(self.cfg.TRAINER.LOG_DIR, 'trainer_config.yaml')
        logging.info(f'Save config to {config_path}')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.TRAINER.to_yaml(config_path)

    def _warmup(self, train_loader):
        logging.info(f'Warmup trainer for {self.cfg.TRAINER.NUM_ITER_WARMUP} iteration(s)')
        self.model.train()
        if self.cfg.TRAINER.NUM_ITER_WARMUP > 0:
            for i, batch in enumerate(train_loader):
                self.train_step(batch)
                logging.debug(f'Warmed {i + 1} iteration(s)')
                if i + 1 == self.cfg.TRAINER.NUM_ITER_WARMUP:
                    break
        logging.info('Warmup trainer finished')
