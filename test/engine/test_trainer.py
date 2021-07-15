import pytest
import torch
import torchinfo
from ocrstack.data.collate import BatchCollator
from ocrstack.data.dataset import DummyDataset
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab
from ocrstack.engine.evaluator import Evaluator
from ocrstack.engine.trainer import Trainer, TrainerConfig
from ocrstack.models import resnet18_attn_lstm, resnet18_lstm_ctc, resnet18_transformer
from ocrstack.transforms.image import BatchPadImages
from ocrstack.transforms.string import BatchPadTexts
from torch import optim
from torch.utils.data.dataloader import DataLoader


def test_log_info():
    vocab = CTCVocab(list('12345678'))
    model = resnet18_lstm_ctc(pretrained=False, vocab=vocab, hidden_size=128)
    torchinfo.summary(model, input_size=[
        [1, 3, 64, 256],
    ])

    vocab = Seq2SeqVocab(list('12345678'))
    model = resnet18_transformer(False, vocab, d_model=128, nhead=8, num_layers=2, max_length=20)
    torchinfo.summary(model,
                      col_names=('input_size', 'output_size', 'num_params'),
                      row_settings=('depth', 'var_names'),
                      input_data=[
                          torch.rand([1, 3, 64, 256]),
                          torch.rand([1, 5, 100]),
                          torch.tensor([5], dtype=torch.int32),
                      ])


def trainer_ctc(device):
    vocab = CTCVocab(list('12345678'))
    vocab_size = len(vocab)
    model = resnet18_lstm_ctc(pretrained=False, vocab=vocab, hidden_size=128)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        batch_size=2,
        lr=1e-4,
        device=device,
        iter_train=2,
        iter_eval=1,
        iter_visualize=1,
        num_iter_visualize=1,
    )

    dataset = DummyDataset(10, 3, 64, 256, 5, vocab_size)

    batch_collator = BatchCollator(
        BatchPadImages(0.),
        BatchPadTexts(0.),
    )

    train_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                              collate_fn=batch_collator)

    val_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                            collate_fn=batch_collator)

    evaluator = Evaluator(model, val_loader, config.device)
    trainer = Trainer(model, optimizer, config, evaluator=evaluator)
    trainer.train(train_loader)


def trainer_seq2seq(device):
    vocab = Seq2SeqVocab(list('12345678'))
    vocab_size = len(vocab)

    model = resnet18_transformer(False, vocab, d_model=128, nhead=8, num_layers=1, max_length=20)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        batch_size=2,
        lr=1e-4,
        device=device,
        iter_train=2,
        iter_eval=1,
        iter_visualize=1,
        num_iter_visualize=1,
    )

    dataset = DummyDataset(10, 3, 64, 256, 5, vocab_size, seq2seq=True)

    batch_collator = BatchCollator(
        BatchPadImages(0.),
        BatchPadTexts(0.),
    )

    train_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                              collate_fn=batch_collator)

    val_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                            collate_fn=batch_collator)

    evaluator = Evaluator(model, val_loader, config.device)
    trainer = Trainer(model, optimizer, config, evaluator=evaluator)
    trainer.train(train_loader)


def trainer_conv_attn_rnn(device):
    vocab = Seq2SeqVocab(list('12345678'))
    model = resnet18_attn_lstm(False, vocab, hidden_size=256, max_length=5)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        batch_size=2,
        lr=1e-4,
        device=device,
        iter_train=2,
        iter_eval=1,
        iter_visualize=1,
        num_iter_visualize=1,
    )

    dataset = DummyDataset(10, 3, 64, 256, 5, len(vocab), seq2seq=True)

    batch_collator = BatchCollator(
        BatchPadImages(0.),
        BatchPadTexts(0.),
    )

    train_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                              collate_fn=batch_collator)

    val_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                            collate_fn=batch_collator)

    evaluator = Evaluator(model, val_loader, config.device)
    trainer = Trainer(model, optimizer, config, evaluator=evaluator)
    trainer.train(train_loader)


def test_trainer_ctc_cpu():
    trainer_ctc('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_ctc_gpu():
    trainer_ctc('cuda')


def test_trainer_seq2seq_cpu():
    trainer_seq2seq('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_seq2seq_gpu():
    trainer_seq2seq('cuda')


def test_trainer_conv_attn_rnn_cpu():
    trainer_conv_attn_rnn('cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuda is not available')
def test_trainer_conv_attn_rnn_gpu():
    trainer_conv_attn_rnn('cuda')
