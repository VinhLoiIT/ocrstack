from typing import List, Optional, Tuple, Union

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


class RandomPadding:
    def __init__(self, pad_width, pad_height, pad_color=None):
        # type: (Union[Tuple[int, int], int], Union[Tuple[int, int], int], Optional[Tuple[int, int, int]])-> None
        self.pad_color = pad_color
        if isinstance(pad_width, int):
            self.min_pad_w, self.max_pad_w = 0, pad_width
        else:
            self.min_pad_w, self.max_pad_w = pad_width

        if isinstance(pad_width, int):
            self.min_pad_h, self.max_pad_h = 0, pad_height
        else:
            self.min_pad_h, self.max_pad_h = pad_height

    def __call__(self, image: Image.Image):
        width, height = image.size
        pad_w = np.random.randint(self.min_pad_w, self.max_pad_w)
        pad_h = np.random.randint(self.min_pad_h, self.max_pad_h)
        new_width = width + pad_w
        new_height = height + pad_h

        if self.pad_color is None:
            # sum corner pixels' colors
            tl = image.getpixel((0, 0))
            tr = image.getpixel((width - 1, 0))
            bl = image.getpixel((0, height - 1))
            br = image.getpixel((width - 1, height - 1))
            pad_color = np.array([p for p in [tl, tr, br, bl]]).mean(axis=0).astype(np.uint8)
            pad_color = tuple(pad_color)
        else:
            pad_color = self.pad_color

        result = Image.new(image.mode, (new_width, new_height), pad_color)
        result.paste(image, (pad_w // 2, pad_h // 2))
        return result


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
