import torch
import torchinfo
from ocrstack.config.trainer import TrainerConfig
from ocrstack.data.collate import BatchCollator
from ocrstack.data.dataset import DummyDataset
from ocrstack.data.vocab import CTCVocab, Seq2SeqVocab
from ocrstack.engine.evaluator import Evaluator
from ocrstack.engine.trainer import Trainer
from ocrstack.loss import CrossEntropyLoss, CTCLoss
from ocrstack.models import resnet18_lstm_ctc, resnet18_transformer
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


def test_trainer_ctc():
    vocab = CTCVocab(list('12345678'))
    vocab_size = len(vocab)
    model = resnet18_lstm_ctc(pretrained=False, vocab=vocab, hidden_size=128)
    criterion = CTCLoss(vocab.BLANK_IDX)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        batch_size=2,
        lr=1e-4,
        device='cpu',
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
    trainer = Trainer(model, criterion, optimizer, config, evaluator=evaluator)
    trainer.train(train_loader)


def test_trainer_seq2seq():
    vocab = Seq2SeqVocab(list('12345678'))
    vocab_size = len(vocab)

    model = resnet18_transformer(False, vocab, d_model=128, nhead=8, num_layers=1, max_length=20)

    criterion = CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    config = TrainerConfig(
        batch_size=2,
        lr=1e-4,
        device='cpu',
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
    trainer = Trainer(model, criterion, optimizer, config, evaluator=evaluator)
    trainer.train(train_loader)
