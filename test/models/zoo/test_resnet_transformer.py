import pytest
import torch
from ocrstack.models.zoo import ResNetTransformerCfg, ResNetTransformer


@pytest.fixture()
def resnet_transformer():
    cfg = ResNetTransformerCfg(
        vocab_size=10,
        embed_dim=20,
        num_heads=2
    )
    model = ResNetTransformer(cfg)
    return model


@pytest.fixture()
def script_resnet_transformer(resnet_transformer):
    return torch.jit.script(resnet_transformer)


@pytest.fixture()
def inputs():
    images = torch.rand(2, 3, 64, 128)
    mask = torch.ones(2, 64, 128).float()
    targets = torch.randint(0, 10, (2, 4))
    return images, mask, targets


def test_resnet_transformer_forward(resnet_transformer, inputs):
    images, _, targets = inputs
    with torch.no_grad():
        outputs = resnet_transformer(images, targets)
    assert outputs.shape == torch.Size((2, 4, 10))


def test_script_resnet_transformer_forward(script_resnet_transformer, inputs):
    images, _, targets = inputs
    with torch.no_grad():
        outputs = script_resnet_transformer(images, targets)
    assert outputs.shape == torch.Size((2, 4, 10))


def test_resnet_transformer_decode(resnet_transformer, inputs):
    images, mask, _ = inputs
    max_length = 7
    resnet_transformer.eval()
    with torch.no_grad():
        outputs, scores = resnet_transformer.decode_greedy(images, max_length, mask)
    assert outputs.size(0) == 2
    assert outputs.size(1) <= max_length + 2
    assert scores.shape == torch.Size((2,))


def test_script_resnet_transformer_decode(script_resnet_transformer, inputs):
    images, mask, _ = inputs
    max_length = 7
    script_resnet_transformer.eval()
    with torch.no_grad():
        outputs, scores = script_resnet_transformer.decode_greedy(images, max_length, mask)
    assert outputs.size(0) == 2
    assert outputs.size(1) <= max_length + 2
    assert scores.shape == torch.Size((2,))
