import torch
from common_utils import _create_dummy_image, _create_dummy_image_mask

from ocrstack.core.builder import build_loss


def test_cross_entropy():
    cfg = {
        'name': 'CrossEntropyLoss',
        'args': {
        }
    }
    loss = build_loss(cfg)

    inputs = _create_dummy_image(1, 3, 7, 7, dtype=torch.float)
    outputs = _create_dummy_image_mask(1, 3, 7, 7)  # 1, 7, 7
    assert inputs.shape == torch.Size([1, 3, 7, 7])
    assert outputs.shape == torch.Size([1, 7, 7])
    loss.forward(inputs, outputs)
