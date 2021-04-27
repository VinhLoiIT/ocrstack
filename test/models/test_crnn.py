import torch
from ocrstack.models import resnet18_lstm_ctc


def test_resnet18_lstm_ctc():
    vocab_size = 10
    model = resnet18_lstm_ctc(pretrained=False, vocab_size=vocab_size, hidden_size=32)
    images = torch.rand(2, 3, 64, 256)
    outputs = model.forward(images)
    assert outputs.shape == torch.Size([256 // 32, 2, vocab_size])

    model.eval()
    outputs = model.forward(images)
    assert outputs.shape == torch.Size([2, 256 // 32, vocab_size])
