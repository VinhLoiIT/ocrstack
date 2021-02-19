from typing import Any, Dict, List, Tuple

import torch


class CollateBatch(object):

    def __init__(self, images, text):
        self.images = images
        self.text = text

    def __len__(self):
        return len(self.images.sizes)

    @staticmethod
    def collate(batch: List[Dict[str, Any]]):
        batch.sort(key=lambda sample: len(sample['text']), reverse=True)
        images = ImageList.from_tensors([x['image'] for x in batch])
        text = TextList.from_tensors([x['text'] for x in batch])
        return CollateBatch(images, text)


class ImageList(object):
    def __init__(self, tensor: torch.Tensor, sizes: List[Tuple[int, int]]):
        self.tensor = tensor
        self.sizes = sizes

    def __getitem__(self, idx):
        size = self.sizes[idx]
        tensor = self.tensor[idx, ..., :size[0], :size[1]]
        return tensor, size

    def __len__(self):
        return len(self.sizes)

    def to(self, device) -> 'ImageList':
        return ImageList(self.tensor.to(device), self.sizes)

    @staticmethod
    def from_tensors(tensors: List[torch.Tensor], pad_value: float = 0.) -> 'ImageList':
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes: List[Tuple[int, int]] = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [torch.as_tensor(x, dtype=torch.int) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if len(tensors) == 1:
            return ImageList(tensors[0].unsqueeze(0).contiguous(), image_sizes)

        # max_size can be a tensor in tracing mode, therefore convert to list
        batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
        batched_imgs = tensors[0].new_full(batch_shape, pad_value)
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)


class TextList(object):
    def __init__(self, tensor: torch.Tensor, lengths: List[int]):
        assert tensor.ndim == 3, tensor.shape
        self.lengths = lengths
        self.tensor = tensor
        _max = self.tensor.max(-1)
        self.max_probs_idx, self.max_probs_val = _max.indices, _max.values
        self.lengths_tensor = torch.as_tensor(lengths, device=tensor.device)
        self.device = tensor.device

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        length = self.lengths[index]
        tensor = self.tensor[index, :length, ...]
        return tensor, length

    def to(self, device) -> 'TextList':
        return TextList(self.tensor.to(device), self.lengths)

    @staticmethod
    def from_tensors(tensors: List[torch.Tensor], pad_value: int = 0) -> 'TextList':
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)

        lengths: List[int] = [text.size(0) for text in tensors]
        max_length = max(lengths)

        if len(tensors) == 1:
            return TextList(tensors[0].unsqueeze(0), lengths)

        batch_shape = [len(tensors), max_length] + list(tensors[0].shape[1:])
        batched_text = tensors[0].new_full(size=batch_shape, fill_value=pad_value)
        for i, t in enumerate(tensors):
            batched_text[i, :t.shape[0], ...].copy_(t)

        return TextList(batched_text.contiguous(), lengths)
