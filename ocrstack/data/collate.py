from typing import Any, Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


class Batch:

    def __init__(self, images, image_mask, text, lengths, text_str, metadata):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[Dict]) -> None
        self.images = images
        self.image_mask = image_mask
        self.text = text
        self.text_str = text_str
        self.metadata = metadata
        self.lengths = lengths

    def __len__(self):
        return len(self.images)

    def to(self, device):
        return Batch(self.images.to(device), self.image_mask.to(device), self.text.to(device),
                     self.lengths, self.text_str, self.metadata)


class BatchCollator:
    def __init__(self, text_padding_value=0, image_padding_value=0., sort_length=False, batch_first=True):
        # type: (int, float, bool, bool) -> None
        r"""
        Args:
            text_padding_value: Value which will be used for text padding since text might have different sizes.
                Default is 0
            image_padding_value: Value which will be used for image padding since images might have different sizes.
                Default is 0.
            sort_length: Sort samples by its groundtruth length.
                Default is False.
            batch_first: returned padded text is (B, T) if batch_first is True, otherwise (T, B)
                Default is True
        """
        self.text_padding_value = text_padding_value
        self.image_padding_value = image_padding_value
        self.sort_length = sort_length
        self.batch_first = batch_first

    def __call__(self, batch: List[Dict[str, Any]]):
        if self.sort_length:
            batch.sort(key=lambda sample: len(sample['text']), reverse=True)

        text_str = [sample['text_str'] for sample in batch]
        metadata = [x.get('metadata', {}) for x in batch]

        images, image_mask = self.collate_images([x['image'] for x in batch])
        text, lengths = self.collate_text([x['text'] for x in batch])

        return Batch(images, image_mask, text, lengths, text_str, metadata)

    def collate_images(self, images: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_shapes = torch.tensor([im.shape for im in images])
        C, H, W = image_shapes.max(dim=0)[0]
        B = len(images)

        batch_image = torch.full((B, C, H, W), fill_value=self.image_padding_value)
        image_mask = torch.full((B, H, W), fill_value=0.0)
        for img, pad_img in zip(images, batch_image):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
            image_mask[..., : img.shape[-2], : img.shape[-2]] = 1.0

        return batch_image, image_mask

    def collate_text(self, texts: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        lengths: List[int] = torch.tensor([len(text) for text in texts])
        padded_text = pad_sequence(texts, self.batch_first, self.text_padding_value)
        return padded_text, lengths
