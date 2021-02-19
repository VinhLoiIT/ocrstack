from ocrstack.config import TrainerConfig

import torch
import torch.nn as nn
import torch.optim as optim
from ocrstack.data.collate import CollateBatch, TextList
from ocrstack.data.vocab import Vocab
from ocrstack.transforms import LabelDecoder

from .trainer import Trainer


class Seq2SeqTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: TrainerConfig,
                 vocab: Vocab,
                 ):
        super(Seq2SeqTrainer, self).__init__(model,
                                             optimizer,
                                             config)
        self.string_tf = LabelDecoder(vocab)

    def val_step(self, batch: CollateBatch):
        inputs, targets = batch.images, batch.text
        inputs = inputs.to(self.config.device)
        with torch.cuda.amp.autocast(enabled=self.config.use_amp), torch.no_grad():
            outputs: TextList = self.model(inputs, max_length=max(targets.lengths))

        predicts = self.string_tf.decode_to_tokens(outputs.max_probs_idx, outputs.lengths_tensor)
        labels = self.string_tf.decode_to_tokens(targets.max_probs_idx[:, 1:], targets.lengths_tensor - 1)

        for metric in self.eval_metrics.values():
            metric.update(predicts, labels)

    def visualize_step(self, batch: CollateBatch):
        inputs, targets = batch.images, batch.text
        inputs = inputs.to(self.config.device)
        with torch.cuda.amp.autocast(enabled=self.config.use_amp), torch.no_grad():
            outputs: TextList = self.model(inputs, max_length=max(targets.lengths))

        predicts = self.string_tf.decode_to_string(outputs.max_probs_idx, outputs.lengths_tensor)
        labels = self.string_tf.decode_to_string(targets.max_probs_idx[:, 1:], targets.lengths_tensor - 1)

        for pred, label in zip(predicts, labels):
            print('-' * 20)
            print(f'Predicts: {pred}')
            print(f'Labels  : {label}')
