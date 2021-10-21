from typing import Union

import torch
import torch.nn as nn


class CollapsePool(nn.Module):
    def __init__(self, mode: str, keepdim: bool = False):
        super(CollapsePool, self).__init__()
        self.pool: Union[nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d]
        if mode == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, None))
        elif mode == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, None))
        else:
            raise ValueError(f'mode should be one of "avg", "max". mode = {mode}')
        self.keepdim = keepdim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        '''
        images: [B, C, H, W]

        Outputs:
        - images: [B, T, E]
        '''
        images = self.pool(images)          # [B, C, 1, W]
        if not self.keepdim:
            images = images.squeeze(-2)
        return images
