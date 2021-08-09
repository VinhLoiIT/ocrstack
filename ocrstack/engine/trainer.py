import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.metrics.metric import AverageMeter
from ocrstack.metrics.ocr import ACCMeter, CERMeter, WERMeter
from ocrstack.models.base import IS2SModel
from ocrstack.models.layers.translator import ITranslator
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

__all__ = [
    'S2STrainConfig',
    'validate_s2s',
    'train_s2s',
    'train_s2s_epoch',
    'setup_logging',
    'create_session_dir',
]


@dataclass()
class S2STrainConfig:
    n_epochs: int = 1000
    learning_rate: int = 1e-4
    batch_size: int = 2
    num_workers: int = 2
    device: str = 'cpu'
    max_length: int = 1
    print_prediction: bool = False
    log_interval: Union[int, float] = 0.1
    validate_steps: int = 1
    save_by: Optional[str] = 'val_loss'
    save_top_k: int = 3
    log_dir: str = 'runs'
    seed: int = 0


def _train_s2s_iteration(model: IS2SModel, optimizer: torch.optim.Optimizer, batch: Batch) -> float:
    optimizer.zero_grad()
    loss = model.forward_batch(batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def _validate_s2s_iteration(cfg: S2STrainConfig,
                            model: IS2SModel,
                            translator: ITranslator,
                            batch: Batch
                            ) -> Tuple[float, List[str]]:
    loss = model.forward_batch(batch)
    predicts = model.decode_greedy(batch.images, batch.image_mask, cfg.max_length)
    predict_strs = translator.translate(predicts)[0]
    return loss.item(), predict_strs


@torch.no_grad()
def validate_s2s(cfg: S2STrainConfig,
                 epoch: int,
                 model: IS2SModel,
                 translator: ITranslator,
                 val_loader: DataLoader,
                 tb_writer: Optional[SummaryWriter] = None) -> Tuple[float, Dict[str, float]]:
    logger = logging.getLogger('Validation')

    total_loss = AverageMeter()
    metrics = {
        'CER': CERMeter(),
        'NormCER': CERMeter(norm=True),
        'WER': WERMeter(),
        'NormWER': WERMeter(norm=True),
        'ACC': ACCMeter(),
    }  # type: Dict[str, AverageMeter]

    if model.training:
        model.eval()

    batch: Batch
    for batch in val_loader:
        batch = batch.to(cfg.device)
        loss, predict_strs = _validate_s2s_iteration(cfg, model, translator, batch)

        total_loss.add(loss, len(batch))
        for metric in metrics.values():
            metric.update(predict_strs, batch.text_str)

        if cfg.print_prediction:
            for predict_str in predict_strs:
                logger.info(predict_str)

    val_loss = total_loss.compute()
    out_metrics = {k: v.compute() for k, v in metrics.items()}

    if tb_writer is not None:
        tb_writer.add_scalar('Val/Loss', val_loss, epoch)
        for k, v in out_metrics.items():
            tb_writer.add_scalar(f'Val/{k}', v, epoch)

    logger.info(f'Epoch [{epoch:3d}] val_loss = {val_loss:.4f}')
    for k, v in out_metrics.items():
        logger.info(f'{k} = {v:.04f}')

    for k, v in out_metrics.items():
        if k != 'ACC':
            out_metrics[k] = -v

    return val_loss, out_metrics


def train_s2s_epoch(cfg: S2STrainConfig,
                    epoch: int,
                    model: IS2SModel,
                    optimizer: torch.optim.Optimizer,
                    train_loader: DataLoader,
                    tb_writer: Optional[SummaryWriter] = None):
    r"""
    Training a model for one epoch
    """
    logger = logging.getLogger('Trainer')

    total_loss = AverageMeter()
    running_loss = AverageMeter()

    num_iter = len(train_loader)
    log_interval = cfg.log_interval
    if isinstance(log_interval, float):
        assert 0 <= log_interval <= 1
        log_interval = int(log_interval * num_iter)

    if not model.training:
        model.train()

    batch: Batch
    for i, batch in enumerate(train_loader):
        batch = batch.to(cfg.device)
        loss = _train_s2s_iteration(model, optimizer, batch)

        with torch.no_grad():

            if tb_writer is not None:
                tb_writer.add_scalar('Loss', loss, num_iter * epoch + i)

            running_loss.add(loss, len(batch))
            total_loss.add(loss, len(batch))

            if (i + 1) % log_interval == 0:
                logger.info('Epoch [{:3d}/{:3d}] - [{:6.2f}%] Running Loss = {:.4f}. Total loss = {:.4f}.'.format(
                    epoch + 1,
                    cfg.n_epochs,
                    (i + 1) * 100 / num_iter,
                    running_loss.compute(),
                    total_loss.compute()
                ))
                running_loss.reset()

    return total_loss.compute(), running_loss.compute()


def train_s2s(cfg: S2STrainConfig,
              model: IS2SModel,
              optimizer: torch.optim.Optimizer,
              train_loader: DataLoader,
              val_loader: DataLoader,
              translator: ITranslator,
              lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              is_debug: bool = False,
              ):

    def state(epoch):
        state_dict = {}
        state_dict['model'] = model.state_dict()
        state_dict['optimizer'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler'] = lr_scheduler.state_dict()
        state_dict['epoch'] = epoch

    setup_logging()
    logger = logging.getLogger('Trainer')

    best_loss = float('inf')
    best_metric = 0.0
    model.train()
    model.to(cfg.device)

    if is_debug:
        logger.info('Start training in debug mode')
        logger.info('Run training for 1 iteration')
        batch = next(iter(train_loader))
        batch = batch.to(cfg.device)
        _train_s2s_iteration(model, optimizer, batch)

        if lr_scheduler is not None:
            lr_scheduler.step()

        model.eval()
        logger.info('Run validation for 1 iteration')
        batch = next(iter(val_loader))
        batch = batch.to(cfg.device)
        _validate_s2s_iteration(cfg, model, translator, batch)
        logger.info('Training DONE')
        return

    session_dir = Path(create_session_dir(cfg.log_dir))

    tensorboard_dir = session_dir.joinpath('tb_logs')
    tb_writer = SummaryWriter(tensorboard_dir)

    ckpt_dir = session_dir.joinpath('ckpt')
    ckpt_dir.mkdir(parents=True)
    ckpt_history: queue.Queue = queue.Queue(cfg.save_top_k)

    logger.info('Start training')

    for epoch in range(cfg.n_epochs):
        train_loss, train_running_loss = train_s2s_epoch(cfg, epoch, model, optimizer, train_loader, tb_writer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        torch.save(state(epoch), ckpt_dir.joinpath('latest.pth'))

        if (epoch + 1) % cfg.validate_steps == 0:
            model.eval()
            val_loss, val_metrics = validate_s2s(cfg, epoch + 1, model, translator, val_loader)
            model.train()

            if val_loss < best_loss:
                logger.info('Found better validation loss. Improved from {:.4f} to {:.4f}'.format(
                    best_loss, val_loss
                ))
                best_loss = val_loss

            if cfg.save_by is None:
                continue

            if cfg.save_by in val_metrics.keys():
                metric_val = val_metrics[cfg.save_by]
            elif cfg.save_by == 'val_loss':
                metric_val = val_loss
            elif cfg.save_by == 'train_loss':
                metric_val = train_loss
            elif cfg.save_by == 'train_running_loss':
                metric_val = train_running_loss
            else:
                raise ValueError(f'Unknow save_by={cfg.save_by}')

            best_metric = max(metric_val, best_metric)
            ckpt_path = ckpt_dir.joinpath(f'{cfg.save_by}={metric_val}.pth')
            torch.save(state(epoch + 1), ckpt_path)

            try:
                ckpt_history.put_nowait(ckpt_path)
            except queue.Full:
                oldest_checkpoint = ckpt_history.get()
                oldest_checkpoint.unlink()
                ckpt_history.put(ckpt_path)

    tb_writer.close()


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
