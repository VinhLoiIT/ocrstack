import pytest
import torch
import torchinfo
from ocrstack.config.config import Config
from ocrstack.data.collate import BatchCollator
from ocrstack.data.dataset import DummyDataset
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab
from ocrstack.engine.evaluator import Evaluator
from ocrstack.engine.trainer import Trainer
from ocrstack.models import (resnet18_attn_lstm, resnet18_lstm_ctc,
                             resnet18_transformer)
from ocrstack.models.layers.translator import CTCTranslator, Seq2SeqTranslator
from ocrstack.transforms.image import BatchPadImages
from ocrstack.transforms.string import BatchPadTexts
from torch import optim
from torch.utils.data.dataloader import DataLoader


def test_log_info():
    vocab = CTCVocab(list('12345678'))
    model = resnet18_lstm_ctc(False, vocab)
    torchinfo.summary(model, input_size=[
        [1, 3, 64, 256],
    ])

    vocab = Seq2SeqVocab(list('12345678'))
    model = resnet18_transformer(False, vocab)
    torchinfo.summary(model,
                      col_names=('input_size', 'output_size', 'num_params'),
                      row_settings=('depth', 'var_names'),
                      input_data=[
                          torch.rand([1, 3, 64, 256]),
                          torch.rand([1, 5, 100]),
                          torch.tensor([5], dtype=torch.int32),
                      ])


def simple_trainer_config(device):
    cfg = Config()
    cfg.TRAINER.BATCH_SIZE = 2
    cfg.TRAINER.LEARNING_RATE = 1e-4
    cfg.TRAINER.DEVICE = device
    cfg.TRAINER.CLIP_GRAD_VALUE = 0.5

    cfg.TRAINER.ITER_TRAIN = 2
    cfg.TRAINER.ITER_EVAL = 1
    cfg.TRAINER.ITER_VISUALIZE = 1
    cfg.TRAINER.NUM_ITER_VISUALIZE = 1
    cfg.TRAINER.NUM_WORKERS = 2
    cfg.TRAINER.NUM_ITER_WARMUP = 2

    cfg.TRAINER.SEED = 0
    cfg.TRAINER.USE_AMP = False
    cfg.TRAINER.LOG_DIR = 'runs'
    cfg.TRAINER.LOG_INTERVAL = 10
    cfg.TRAINER.MONITOR_METRIC = 'CER'
    cfg.TRAINER.MONITOR_METRIC_TYPE = 'lower'
    cfg.TRAINER.PRETRAINED_WEIGHT = None
    cfg.TRAINER.PRETRAINED_CONFIG = None
    cfg.TRAINER.RESUME_CHECKPOINT = None
    cfg.TRAINER.CONTINUE_TRAINING = False

    return cfg


def trainer_ctc(device):
    vocab = CTCVocab(list('12345678'))
    vocab_size = len(vocab)
    model = resnet18_lstm_ctc(pretrained=False, vocab=vocab)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    cfg = simple_trainer_config(device)

    dataset = DummyDataset(10, 3, 64, 256, 5, vocab_size)

    batch_collator = BatchCollator(
        BatchPadImages(0.),
        BatchPadTexts(0.),
    )

    train_loader = DataLoader(dataset, cfg.TRAINER.BATCH_SIZE, num_workers=cfg.TRAINER.NUM_WORKERS,
                              collate_fn=batch_collator)

    val_loader = DataLoader(dataset, cfg.TRAINER.BATCH_SIZE, num_workers=cfg.TRAINER.NUM_WORKERS,
                            collate_fn=batch_collator)

    translator = CTCTranslator(vocab, True)
    evaluator = Evaluator(model, translator, val_loader, cfg.TRAINER.DEVICE)
    trainer = Trainer(model, optimizer, cfg, evaluator=evaluator)
    trainer.train(train_loader)


def trainer_seq2seq(device, model, *args, **kwargs):
    vocab = Seq2SeqVocab(list('12345678'))
    vocab_size = len(vocab)

    model = model(vocab=vocab, *args, **kwargs)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    cfg = simple_trainer_config(device)

    dataset = DummyDataset(10, 3, 64, 256, 5, vocab_size, seq2seq=True)

    batch_collator = BatchCollator(
        BatchPadImages(0.),
        BatchPadTexts(vocab.PAD_IDX),
    )

    train_loader = DataLoader(dataset, cfg.TRAINER.BATCH_SIZE, num_workers=cfg.TRAINER.NUM_WORKERS,
                              collate_fn=batch_collator)

    val_loader = DataLoader(dataset, cfg.TRAINER.BATCH_SIZE, num_workers=cfg.TRAINER.NUM_WORKERS,
                            collate_fn=batch_collator)

    translator = Seq2SeqTranslator(vocab, False, False, False)
    evaluator = Evaluator(model, translator, val_loader, cfg.TRAINER.DEVICE)
    trainer = Trainer(model, optimizer, cfg, evaluator=evaluator)
    trainer.train(train_loader)


def test_trainer_ctc_cpu():
    trainer_ctc('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_ctc_gpu():
    trainer_ctc('cuda')


def test_trainer_resnet18_transformer_cpu():
    trainer_seq2seq('cpu', resnet18_transformer, pretrained=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_seq2seq_gpu():
    trainer_seq2seq('cuda', resnet18_transformer, pretrained=False)


def test_trainer_conv_attn_rnn_cpu():
    trainer_seq2seq('cpu', resnet18_attn_lstm, pretrained=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_conv_attn_rnn_gpu():
    trainer_seq2seq('cuda', resnet18_attn_lstm, pretrained=False)
