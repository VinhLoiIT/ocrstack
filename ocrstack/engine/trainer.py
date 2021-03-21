import logging
from ocrstack.model.arch.base import BaseModel
from ocrstack.data.collate import CollateBatch
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.nn.utils.clip_grad import clip_grad_value_
import torch.optim as optim
from ocrstack.metrics import ACCMeter, CERMeter, WERMeter
from ocrstack.metrics.metric import AverageMeter
from torch.utils.data.dataloader import DataLoader
from ocrstack.config import TrainerConfig


class Trainer(object):

    '''
    Base class for OCR Trainer
    '''

    def __init__(self,
                 model: BaseModel,
                 optimizer: optim.Optimizer,
                 config: TrainerConfig,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)
        self.config = config
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        self.eval_metrics = {
            'CER': CERMeter(),
            'WER': WERMeter(),
            'ACC': ACCMeter(),
        }

        self.train_metrics = {
            'Loss': AverageMeter(),
            'Time': AverageMeter(),
        }

        self.checkpoint_saver = CheckpointSaver(Path(config.checkpoint_dir), config.monitor_metric_type)
        self.state = {}

    def train_step(self, batch: CollateBatch):
        inputs, targets = batch.images, batch.text
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.model(batch)
            loss = self.model.compute_batch_loss(batch, outputs)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        return loss.item()

    def val_step(self, batch):
        inputs, targets = batch.images, batch.text
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)

        with torch.cuda.amp.autocast(enabled=self.config.use_amp), torch.no_grad():
            outputs = self.model(batch)
            metrics = self.model.compute_batch_metrics(batch, outputs)

        self.model.update_metrics(self.state.get('metrics', {}), metrics)

    def visualize_step(self, batch):
        pass

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None):

        self.model.to(self.config.device)
        self.model.train()

        self._save_config()
        self._warmup(train_loader)

        num_iteration = 0
        epoch = 0
        logging.info(f'Start training for {self.config.iter_train} iteration(s)')
        while num_iteration < self.config.iter_train:
            loss_meter = self.train_metrics['Loss']
            # time_meter = self.train_metrics['Time']
            for i, batch in enumerate(train_loader):
                loss = self.train_step(batch)
                loss_meter.update(loss, len(batch))
                num_iteration += 1

                if num_iteration % self.config.log_interval == 0:
                    logging.info(f'Iteration {num_iteration}: Loss {loss:.4f}')
                    loss_meter.reset()

                if num_iteration % self.config.iter_visualize == 0:
                    self.model.eval()
                    logging.info('Visualizing training process')
                    with torch.cuda.amp.autocast(enabled=self.config.use_amp), torch.no_grad():
                        loader = val_loader or train_loader
                        for i, batch in enumerate(loader):
                            self.visualize_step(batch)

                            if (i + 1) % self.config.num_iter_visualize == 0:
                                logging.info('Visualize done')
                                break
                    self.model.train()

                if val_loader is not None and num_iteration % self.config.iter_eval == 0:
                    logging.info(f'Evaluating at iteration = {num_iteration}:')
                    for metric in self.eval_metrics.values():
                        metric.reset()

                    self.model.eval()
                    with torch.cuda.amp.autocast(enabled=self.config.use_amp), torch.no_grad():
                        for i, batch in enumerate(val_loader):
                            self.val_step(batch)

                            if (i + 1) % self.config.log_interval == 0 or i == len(train_loader) - 1:
                                log_str = f'Iteration {i+1:5d}/{len(val_loader)}:'
                                for name, metric in self.eval_metrics.items():
                                    log_str += f' {name}: {metric.compute():.3f}'
                                logging.info(log_str)

                    logging.info('Evaluation metrics:')
                    for name, metric in self.eval_metrics.items():
                        logging.info(f'{name}: {metric.compute():.3f}')

                    metric_value = self.eval_metrics[self.config.monitor_metric].compute()
                    if self.checkpoint_saver.is_better(metric_value):
                        to_save = {
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'lr_scheduler': self.lr_scheduler.state_dict(),
                            'epoch': epoch,
                        }
                        self.checkpoint_saver.save(epoch, metric_value, to_save)
                        logging.info('Found better checkpoint.')
                        logging.info(f'{self.config.monitor_metric} is updated to {metric_value}')

                    self.model.train()

                if num_iteration >= self.config.iter_train:
                    break

            epoch += 1

    def _save_config(self):
        config_path = Path(self.config.checkpoint_dir, 'trainer_config.yaml')
        logging.info(f'Save config to {config_path}')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.to_yaml(config_path)

    def _warmup(self, train_loader):
        logging.info(f'Warmup trainer for {self.config.num_iter_warmup} iteration(s)')
        self.model.train()
        if self.config.num_iter_warmup > 0:
            for i, batch in enumerate(train_loader):
                self.train_step(batch)
                logging.debug(f'Warmed {i + 1} iteration(s)')
                if i + 1 == self.config.num_iter_warmup:
                    break
        logging.info('Warmup trainer finished')


class CheckpointSaver(object):
    '''
    Checkpointer class
    '''

    MODE_HIGHER = 'higher'
    MODE_LOWER = 'lower'

    def __init__(self, checkpoint_dir: Path, mode: str = 'greater'):
        self._prev_name: Optional[Path] = None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.mode = mode
        if mode == CheckpointSaver.MODE_HIGHER:
            self.best = float('-inf')
        elif mode == CheckpointSaver.MODE_LOWER:
            self.best = float('inf')
        else:
            raise ValueError('Unknow mode = {mode}. It should be "higher" or "lower"')

    def save(self, epoch: int, metric_value: float, to_save: Dict):
        if self.is_better(metric_value):
            self.best = metric_value
            if self._prev_name is not None:
                self._prev_name.unlink(True)
            checkpoint_path = self.checkpoint_dir.joinpath(f'epoch={epoch}_{metric_value:.5f}.pth')
            torch.save(to_save, checkpoint_path)
            self._prev_name = checkpoint_path

    def is_better(self, value: float) -> bool:
        return ((self.mode == CheckpointSaver.MODE_HIGHER and value > self.best)
                or (self.mode == CheckpointSaver.MODE_LOWER and value < self.best))
