from typing import Callable

import torch
import torch.nn as nn
from ocrstack.data.collate import ImageList
from ocrstack.data.vocab import Vocab
from ocrstack.transforms import LabelDecoder
from PIL import Image


class VisualSeqInfer():
    def __init__(self,
                 vocab: Vocab,
                 model: nn.Module,
                 image_transform: Callable,
                 text_transform: Callable,
                 use_amp: bool = False,
                 device: str = 'cpu',
                 ):
        self.vocab = vocab
        self.model = model
        self.string_tf = LabelDecoder(vocab)
        self.use_amp = use_amp
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self.image_transform = image_transform
        self.text_transform = text_transform

    def single_run(self, image: Image.Image):
        inputs = self._prepare_input(image)
        with torch.cuda.amp.autocast(enabled=self.use_amp), torch.no_grad():
            outputs = self.model(inputs, max_length=max(targets.lengths))  # TODO: Test for now

        predicts = self.string_tf.decode_to_tokens(outputs.tensor.argmax(-1), outputs.lengths_tensor)
        labels = self.string_tf.decode_to_tokens(targets.tensor.argmax(-1), targets.lengths_tensor)

        for metric in self.eval_metrics.values():
            metric.update(predicts, labels)

    def _prepare_input(self, image: Image.Image) -> ImageList:
        image_tensor = self.image_transform(image)

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        inputs = Seq2SeqInput(inputs, targets)
