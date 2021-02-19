from typing import Optional

import torch
import torch.nn.functional as F
from ocrstack.data.collate import ImageList, TextList
from ocrstack.model.component import Classifier, ImageEmbedding
from torch.nn import Module

__all__ = ['CRNN']


class CTCOutputLayer(Module):
    def __init__(self, blank_idx: int, mode='greedy'):
        super(CTCOutputLayer, self).__init__()
        self.mode = mode
        self.blank_idx = blank_idx

    def forward(self, logits, targets=None):
        if self.training:
            assert targets is not None
            return self.losses(logits, targets)

        if self.mode == 'greedy':
            probs = F.softmax(logits, dim=-1)
            predicts = probs.argmax(-1)
            return predicts

        raise NotImplementedError()

    def losses(self, logits, targets=None):
        input_lengths = torch.tensor([logits.size(1)] * logits.size(0), dtype=torch.int, device=logits.device)
        log_probs = F.log_softmax(logits, dim=-1)                      # [B, T, V]
        loss = F.ctc_loss(log_probs.transpose(0, 1),                   # [T, B, V]
                          targets.tensor.argmax(-1),                   # [B, S]
                          input_lengths=input_lengths,
                          target_lengths=targets.lengths_tensor,
                          blank=self.blank_idx, reduction='mean',
                          zero_infinity=False)
        loss = {
            'ctc_loss': loss
        }
        return loss


class CRNN(Module):

    def __init__(self, blank_idx: int, image_embed: ImageEmbedding, classifier: Classifier):
        super(CRNN, self).__init__()
        self.image_embed = image_embed
        self.classifier = classifier
        self.output = CTCOutputLayer(blank_idx)

    def forward(self, inputs: ImageList, targets: Optional[TextList] = None):
        '''
        Shapes:
        -------
        - images: [B,C,H,W]
        - padding_mask: [B,H,W]

        Returns:
        ----
        - outputs: [B,T,V]
        '''

        if not self.training:
            return self.inference(inputs)

        assert targets is not None
        images = self.image_embed(inputs.tensor)                    # [B, T, E]
        logits = self.classifier(images)                            # [B, T, V]
        output = self.output(logits, targets)
        loss = {}
        loss.update(output)
        return loss

    def inference(self, inputs: ImageList):
        images = self.image_embed(inputs.tensor)
        images = self.classifier(images)
        images = self.output(images)
        return images
