from torch import Tensor
import torch.nn as nn


class TextEmbedding(nn.Module):
    def __init__(self, *layers):
        super(TextEmbedding, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, text: Tensor) -> Tensor:
        for layer in self.layers:
            text = layer(text)

        # text = text.transpose(0, 1)

        return text


class ImageEmbedding(nn.Module):
    def __init__(self, batch_first: bool, *layers: nn.Module):
        super().__init__()
        self.batch_first = batch_first
        self.layers = nn.ModuleList(layers)

    def forward(self, images: Tensor) -> Tensor:
        for layer in self.layers:
            images = layer(images)

        if self.batch_first:
            return images

        images = images.permute(2, 0, 1)

        return images


class CharacterEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, embed_size)

    def forward(self, text: Tensor) -> Tensor:
        '''
        Shapes:
        -------
        - text: (N, T, V)
        - padding_mask: (N, T)
        '''
        text = self.linear(text.float())
        return text


class Classifier(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size


class LinearClassifier(Classifier):
    def __init__(self, embed_size, vocab_size, bias: bool = True):
        super().__init__(embed_size, vocab_size)
        self.classifier = nn.Linear(embed_size, vocab_size, bias=bias)

    def forward(self, logits: Tensor) -> Tensor:
        outputs = self.classifier(logits)
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, images: Tensor) -> Tensor:
        B, C, H, W = images.shape
        images = images.transpose(-2, -1).reshape(B, C, W * H)
        images = images.transpose(-2, -1)
        return images


class ConvEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(ConvEmbed, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, (1, 1), bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, images: Tensor) -> Tensor:
        images = self.conv(images)
        return images


class LinearEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, images: Tensor) -> Tensor:
        images = self.linear(images)
        return images
