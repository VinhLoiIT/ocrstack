import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextModeling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, (1, 1))

    def forward(self, images: torch.Tensor):
        '''
        Shapes:
        -------
            - images: [B, C, H, W]
            - out: [B, C, 1, 1]
        '''
        B, C, H, W = images.shape
        prob = self.conv(images)                # [B, 1, H, W]
        prob = prob.reshape(B, H*W, 1, 1)       # [B, H*W, 1, 1]
        prob = F.softmax(prob, 1)               # [B, H*W, 1, 1]

        images = images.reshape(B, C, H*W)      # [B, C, H*W]
        ctx = torch.bmm(images, prob)           # [B, C, 1, 1]
        return ctx


class Transform(nn.Module):
    def __init__(self, in_channels, ratio):
        super().__init__()
        self.ratio = ratio
        self.in_channels = in_channels
        self.out_channels = in_channels // ratio

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, (1, 1))
        self.norm = nn.LayerNorm(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.in_channels, (1, 1))

    def forward(self, images: torch.Tensor):
        '''
        Shapes:
        -------
            - images: [B, C, 1, 1]
            - out: [B, C, 1, 1]
        '''
        images = self.conv1(images)         # [B, C/r, 1, 1]
        images = F.relu(self.norm(images))  # [B, C/r, 1, 1]
        images = self.conv2(images)         # [B, C, 1, 1]
        return images


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.context_modeling = ContextModeling(in_channels)
        self.transform = Transform(in_channels, ratio)

    def forward(self, images: torch.Tensor):
        '''
        Shapes:
        -------
            - images: [B, C, H, W]
            - out: [B, C, H, W]
        '''
        out = self.context_modeling(images)     # [B, C, 1, 1]
        out = self.transform(out)               # [B, C, 1, 1]
        images = images + out                   # [B, C, H, W]
        return images
