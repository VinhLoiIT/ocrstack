from ocrstack.models.backbone.unet import UNet
import torch
import pytest


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('channels', [[64, 128, 256, 512]])
@pytest.mark.parametrize('inner_channels', [64, 128])
def test_forward_unet(batch_size, channels, inner_channels):
    feats = [
        torch.rand(batch_size, channels[0], 128, 128),
        torch.rand(batch_size, channels[1], 64, 64),
        torch.rand(batch_size, channels[2], 32, 32),
        torch.rand(batch_size, channels[3], 16, 16)
    ]
    unet = UNet(channels, inner_channels)
    outs = unet.forward(feats)
    assert outs.shape == torch.Size([batch_size, inner_channels, 128, 128])
