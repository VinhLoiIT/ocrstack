from typing import List

import numpy as np
import torch
from PIL import Image
from torch.types import Number

__all__ = [
    'ScaleHeight',
    'RGBA2RGB',
    'BatchPadImages',
]


class ScaleHeight(object):
    def __init__(self, height: int):
        self.height = height

    def __call__(self, image: Image.Image):
        w, h = image.size
        factor = self.height / h
        w, h = int(w * factor), int(h * factor)
        image = image.resize((w, h))
        return image


class RGBA2RGB:
    def __init__(self, background_color):
        self.background_color = background_color

    def __call__(self, image: Image.Image):
        background = Image.new('RGBA', image.size, self.background_color)
        alpha_composite = Image.alpha_composite(background, image)
        return alpha_composite.convert('RGB')


class BatchPadImages:
    def __init__(self, pad_value: Number = 0.):
        self.pad_value = pad_value

    def __call__(self, images: List[torch.Tensor]):
        assert len(images) > 0
        B = len(images)

        image_shapes = np.array([im.shape for im in images])
        C, H, W = image_shapes.max(axis=0)

        batched_imgs = torch.full((B, C, H, W), fill_value=self.pad_value)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return batched_imgs
