import torch.nn as nn
from torch import Tensor


class CollapseConcat(nn.Module):
    def __init__(self, in_channels: int, height: int):
        super(CollapseConcat, self).__init__()
        self.in_channels = in_channels
        self.out_channels = height * self.in_channels

    def forward(self, images: Tensor) -> Tensor:
        '''
        Shapes:
        -------
        - images: (B, C, H, W)
        - padding_mask: (B, H, W). Bool tensor

        Outputs:
        --------
        - sequence: (B, T, C*H) where T = W
        - padding_mask: (B, T) where T = W
        '''
        B, C, H, W = images.shape
        images = images.reshape(B, C * H, W)
        images = images.transpose(-2, -1)
        return images
