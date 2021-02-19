from ocrstack.config import TrainerConfig
from ocrstack.transforms.string import CTCDecoder, LabelDecoder

import torch.nn as nn
import torch.optim as optim
from ocrstack.data.vocab import CTCVocab

from .trainer import Trainer


class VisualSeqTrainer(Trainer):
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 config: TrainerConfig,
                 vocab: CTCVocab,
                 ):
        super(VisualSeqTrainer, self).__init__(model,
                                               optimizer,
                                               config)
        self.ctc_decoder = CTCDecoder(vocab)
        self.label_decoder = LabelDecoder(vocab)

    def val_step(self, batch):
        inputs, targets = batch.images, batch.text
        inputs = inputs.to(self.config.device)
        targets = targets.to(self.config.device)

        outputs = self.model(inputs)

        predicts = self.ctc_decoder.decode_to_tokens(outputs)
        labels = self.label_decoder.decode_to_tokens(targets.tensor.argmax(-1), targets.lengths_tensor)

        for metric in self.eval_metrics.values():
            metric.update(predicts, labels)
