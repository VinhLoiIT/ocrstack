import logging
import queue
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from ocrstack.data.vocab import Seq2SeqVocab
from ocrstack.metrics.metric import AverageMeter
from ocrstack.metrics.ocr import (ACCMeter, GlobalCERMeter, GlobalWERMeter,
                                  NormCERMeter, NormWERMeter)
from ocrstack.models.base import ITrainableS2S
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import set_seed

__all__ = [
    'S2STrainCfg',
    'S2STrainer',
    'setup_logging',
    'create_session_dir',
]


class S2STrainCfg(Config):

    def __init__(
        self,
        n_epochs: int = 1000,
        learning_rate: int = 1e-4,
        batch_size: int = 2,
        num_workers: int = 2,
        device: str = 'cpu',
        max_length: int = 1,
        num_iter_visualize: Union[int, float] = 0.05,
        log_interval: Union[int, float] = 0.1,
        validate_steps: int = 1,
        save_by: Optional[str] = 'val_loss',
        save_top_k: int = 3,
        log_dir: str = 'runs',
        seed: Optional[int] = None,
        reduction_char_visualize: Optional[str] = None,
        is_debug: bool = False,
        enable_early_stopping: bool = False,
        num_val_early_stopping: int = 10,
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.max_length = max_length
        self.num_iter_visualize = num_iter_visualize
        self.log_interval = log_interval
        self.validate_steps = validate_steps
        self.save_by = save_by
        self.save_top_k = save_top_k
        self.log_dir = log_dir
        self.seed = seed
        self.reduction_char_visualize = reduction_char_visualize
        self.is_debug = is_debug
        self.enable_early_stopping = enable_early_stopping
        self.num_val_early_stopping = num_val_early_stopping


class S2STrainer:
    def __init__(self,
                 cfg: S2STrainCfg,
                 vocab: Seq2SeqVocab,
                 model: ITrainableS2S,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.lr_scheduler = lr_scheduler

        setup_logging()
        self.logger = logging.getLogger('Trainer')

    def state_dict(self, epoch):
        state_dict = {}
        state_dict['model'] = self.model.state_dict()
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        state_dict['epoch'] = epoch
        return state_dict

    def train(self):
        best_loss = float('inf')
        best_metric = 0.0
        count_early_stopping = 0

        self.model.train()
        self.model.to(self.cfg.device)

        session_dir = Path(create_session_dir(self.cfg.log_dir))

        self.cfg.to_yaml(session_dir.joinpath('trainer_config.yaml'))
        self.vocab.to_json(session_dir / "vocab.json")

        tensorboard_dir = session_dir.joinpath('tb_logs')
        tb_writer = SummaryWriter(tensorboard_dir)

        ckpt_dir = session_dir.joinpath('ckpt')
        ckpt_dir.mkdir(parents=True)
        ckpt_history: queue.Queue = queue.Queue(self.cfg.save_top_k)

        num_iter = len(self.train_loader)
        log_interval = _normalize_interval(self.train_loader, self.cfg.log_interval)

        self.logger.info('Start training')

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)

        for epoch in range(self.cfg.n_epochs):
            total_loss_meter = AverageMeter()
            running_loss_meter = AverageMeter()

            if not self.model.training:
                self.model.train()

            batch: Batch
            for i, batch in enumerate(self.train_loader):
                batch = batch.to(self.cfg.device)
                self.optimizer.zero_grad()
                loss = self.model.forward_batch(batch)
                loss.backward()
                self.optimizer.step()
                loss = loss.item()

                tb_writer.add_scalar('Train/Loss', loss, num_iter * epoch + i)

                running_loss_meter.add(loss * len(batch), len(batch))
                total_loss_meter.add(loss * len(batch), len(batch))

                if (i + 1) % log_interval == 0:
                    self.logger.info(
                        'Epoch [%3d/%3d] - [%6.2f] Running Loss = %.4f. Total loss = %.4f.',
                        epoch + 1,
                        self.cfg.n_epochs,
                        (i + 1) * 100 / num_iter,
                        running_loss_meter.compute(),
                        total_loss_meter.compute()
                    )
                    running_loss_meter.reset()

                if self.cfg.is_debug and i == 2:
                    break

            train_loss = total_loss_meter.compute()
            train_running_loss = running_loss_meter.compute()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            torch.save(self.state_dict(epoch), ckpt_dir.joinpath('latest.pth'))

            if (epoch + 1) % self.cfg.validate_steps == 0:
                self.model.eval()
                with torch.no_grad():
                    self._visualize_epoch(epoch + 1)
                    val_loss, val_metrics = self._validate_epoch(epoch + 1, tb_writer)
                self.model.train()

                if val_loss < best_loss:
                    self.logger.info(
                        'Found better validation loss. Improved from %.4f to %.4f',
                        best_loss, val_loss
                    )
                    count_early_stopping = 0
                    best_loss = val_loss
                else:
                    self.logger.info('Loss does not improve. Best loss = %.4f.', best_loss)
                    count_early_stopping += 1
                    if count_early_stopping >= self.cfg.num_val_early_stopping:
                        self.logger.info('Loss does not improve for %d times. Early stop.', count_early_stopping)
                        break

                if self.cfg.save_by is None:
                    continue

                if self.cfg.save_by in val_metrics.keys():
                    metric_val = val_metrics[self.cfg.save_by]
                elif self.cfg.save_by == 'val_loss':
                    metric_val = val_loss
                elif self.cfg.save_by == 'train_loss':
                    metric_val = train_loss
                elif self.cfg.save_by == 'train_running_loss':
                    metric_val = train_running_loss
                else:
                    raise ValueError(f'Unknow save_by={self.cfg.save_by}')

                best_metric = max(metric_val, best_metric)
                ckpt_path = ckpt_dir.joinpath(f'{self.cfg.save_by}={metric_val}.pth')
                torch.save(self.state_dict(epoch + 1), ckpt_path)

                try:
                    ckpt_history.put_nowait(ckpt_path)
                except queue.Full:
                    oldest_checkpoint = ckpt_history.get()
                    oldest_checkpoint.unlink()
                    ckpt_history.put(ckpt_path)

            if self.cfg.is_debug and (epoch + 1) == 2:
                break

        tb_writer.close()

    def _visualize_epoch(self, epoch):
        if self.cfg.is_debug:
            num_iter = 1
        else:
            num_iter = _normalize_interval(self.val_loader, self.cfg.num_iter_visualize)
        self.logger.info('Visualize Epoch [{:3d}]'.format(epoch))

        batch: Batch
        for i, batch in enumerate(self.val_loader):
            if i == num_iter:
                break
            batch = batch.to(self.cfg.device)
            indices = self.model.decode_greedy(batch.images, self.cfg.max_length, batch.image_mask)[0]
            predict_strs = self.vocab.translate(indices, self.cfg.reduction_char_visualize)
            for predict_str, tgt in zip(predict_strs, batch.text_str):
                self.logger.info(f'Predict: {predict_str} ; Target: {tgt}')

    def _validate_epoch(self, epoch: int, tb_writer: SummaryWriter) -> Tuple[float, Dict[str, float]]:
        logger = logging.getLogger('Validation')

        total_loss_meter = AverageMeter()
        metrics = {
            'CER': GlobalCERMeter(),
            'NormCER': NormCERMeter(),
            'WER': GlobalWERMeter(),
            'NormWER': NormWERMeter(),
            'ACC': ACCMeter(),
        }  # type: Dict[str, AverageMeter]

        num_iter = len(self.val_loader)
        log_interval = _normalize_interval(self.val_loader, self.cfg.log_interval)

        batch: Batch
        for i, batch in enumerate(self.val_loader):
            batch = batch.to(self.cfg.device)
            loss = self.model.forward_batch(batch)
            indices = self.model.decode_greedy(batch.images, self.cfg.max_length, batch.image_mask)[0]

            pred_tokens = self.vocab.translate(indices)
            tgt_tokens = self.vocab.translate(batch.text)

            total_loss_meter.add(loss.item() * len(batch), len(batch))
            for metric in metrics.values():
                metric.update(pred_tokens, tgt_tokens)

            if (i + 1) % log_interval == 0:
                logger.info('Epoch [{:3d}] - [{:6.2f}] val_loss = {:.4f} - {}'.format(
                    epoch,
                    (i + 1) * 100 / num_iter,
                    total_loss_meter.compute(),
                    ' - '.join([f'{k}: {v.compute():.4f}' for k, v in metrics.items()])
                ))

            if self.cfg.is_debug and i == 2:
                break

        val_loss = total_loss_meter.compute()
        out_metrics = {k: v.compute() for k, v in metrics.items()}

        tb_writer.add_scalar('Validation/Loss', val_loss, epoch)
        for k, v in out_metrics.items():
            tb_writer.add_scalar(f'Validation/{k}', v, epoch)

        logger.info('Epoch [{:3d}] - val_loss = {:.4f} - {}'.format(
            epoch,
            val_loss,
            ' - '.join([f'{k}: {v:.4f}' for k, v in out_metrics.items()])
        ))

        for k, v in out_metrics.items():
            if k != 'ACC':
                out_metrics[k] = -v

        return val_loss, out_metrics


def setup_logging():
    logging.basicConfig(format='[%(levelname)s] %(name)s: %(message)s', level=logging.INFO)


def create_session_dir(root_dir: str, name: Optional[str] = None, exist_ok: bool = False) -> str:
    from datetime import datetime
    from pathlib import Path
    if name is None:
        name = datetime.now().strftime('%Y%m%d-%H%M%S')

    log_dir = Path(root_dir, name)
    log_dir.mkdir(parents=True, exist_ok=exist_ok)
    return str(log_dir)


def _normalize_interval(loader: DataLoader, interval: Union[int, float]) -> int:
    num_iter = len(loader)
    interval = interval
    if isinstance(interval, float):
        assert 0 <= interval <= 1
        interval = int(interval * num_iter)

    return interval
