from typing import Any, Callable, Dict, List, Optional

import torch


class Batch:

    def __init__(self, images, text, lengths, text_str, metadata):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[Dict]) -> None
        self.images = images
        self.text = text
        self.text_str = text_str
        self.metadata = metadata
        self.lengths = lengths

    def __len__(self):
        return len(self.images)

    def to(self, device):
        return Batch(self.images.to(device), self.text.to(device),
                     self.lengths, self.text_str, self.metadata)


class BatchCollator:
    def __init__(self, batch_image_transform=None, batch_text_transform=None):
        # type: (Optional[Callable], Optional[Callable]) -> None
        self.batch_image_transform = batch_image_transform or torch.stack
        self.batch_text_transform = batch_text_transform or torch.stack

    def __call__(self, batch: List[Dict[str, Any]]):
        batch.sort(key=lambda sample: len(sample['text']), reverse=True)
        text_str = [sample['text_str'] for sample in batch]
        lengths = torch.tensor([len(sample['text']) for sample in batch])
        images = self.batch_image_transform([x['image'] for x in batch])
        text = self.batch_text_transform([x['text'] for x in batch])
        metadata = [x.get('metadata', {}) for x in batch]
        return Batch(images, text, lengths, text_str, metadata)
