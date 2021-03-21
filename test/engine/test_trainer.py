import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.trainer import TrainerConfig
from ocrstack.data.collate import CollateBatch
from ocrstack.data.dataset import DummyDataset
from ocrstack.engine.trainer import Trainer
from ocrstack.model.arch.base import BaseModel
from ocrstack.model.arch.seq2seq import Seq2Seq
from ocrstack.model.component.conv_adapter import ResNetAdapter
from ocrstack.model.component.sequence_decoder import TransformerDecoderAdapter
from ocrstack.model.component.sequence_encoder import TransformerEncoderAdapter
from torch import optim
from torch.utils.data.dataloader import DataLoader


class DummyModel(BaseModel):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = ResNetAdapter('resnet18', False, 0)
        d_model = 512
        nhead = 8
        dim_feedforward = 128
        vocab_size = 10

        tf_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        tf_encoder = nn.TransformerEncoder(tf_encoder_layer, 1)
        encoder = TransformerEncoderAdapter(tf_encoder)

        tf_decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        tf_decoder = nn.TransformerDecoder(tf_decoder_layer, 1)
        text_embedding = nn.Linear(vocab_size, d_model)
        text_classifier = nn.Linear(d_model, vocab_size)

        sos_token_idx = F.one_hot(torch.tensor([0], dtype=torch.long), vocab_size)
        eos_token_idx = F.one_hot(torch.tensor([1], dtype=torch.long), vocab_size)
        decoder = TransformerDecoderAdapter(text_embedding, text_classifier, sos_token_idx, eos_token_idx, tf_decoder)

        self.seq2seq = Seq2Seq(decoder, encoder)

    def forward_train(self, batch: CollateBatch):
        images = self.conv(batch.images.tensor)
        B, C, H, W = images.shape
        images = images.reshape(B, C, H*W)
        images = images.transpose(1, 2)
        targets = batch.text.tensor[:, :-1].float()
        seq = self.seq2seq(images, targets)
        return seq

    def compute_batch_loss(self, batch: CollateBatch, outputs):
        B, T, E = outputs.shape
        outputs = outputs.reshape(B * T, E)
        targets = batch.text.tensor[:, 1:].argmax(-1)
        targets = targets.view(-1)
        return F.cross_entropy(outputs, targets)


def test_trainer_train():
    model = DummyModel()
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

    dataset = DummyDataset(10, 3, 64, 256, 5, 10)
    train_loader = DataLoader(dataset, config.batch_size, num_workers=config.num_workers,
                              collate_fn=CollateBatch.collate)

    trainer = Trainer(model, optimizer, config)
    trainer.train(train_loader)
