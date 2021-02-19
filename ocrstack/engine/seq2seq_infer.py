from ocrstack.engine.attention_visualizer import AttentionVisualizer
from typing import Callable, Optional

import torch.nn as nn
from ocrstack.data.collate import ImageList, TextList
from ocrstack.data.vocab import Seq2SeqVocab
from ocrstack.transforms import LabelDecoder
from PIL import Image


class Seq2SeqInfer():
    def __init__(self,
                 vocab: Seq2SeqVocab,
                 model: nn.Module,
                 image_transform: Callable,
                 text_transform: Callable,
                 use_amp: bool = False,
                 device: str = 'cpu',
                 attn_visualizer: Optional[AttentionVisualizer] = None
                 ):
        self.vocab = vocab
        self.string_tf = LabelDecoder(vocab)
        self.use_amp = use_amp
        self.device = device

        self.model = model
        self.model = self.model.to(device)
        self.model.eval()

        self.image_transform = image_transform
        self.text_transform = text_transform
        if attn_visualizer is not None:
            attn_visualizer.setup(self.model)

    def run(self, image: Image.Image, max_length: int) -> str:
        image_tensor = self.image_transform(image)
        images = ImageList.from_tensors([image_tensor])

        text_tensor = self.text_transform('')
        text = TextList.from_tensors([text_tensor])

        output: TextList = self.model(inputs, max_length=max_length)

        string = self.string_tf.decode_to_string(output.max_probs_idx, output.lengths_tensor - 1)[0]
        return string
