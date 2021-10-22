import math

import torch.nn as nn

from ocrstack.core.builder import MODULE_REGISTRY

MODULE_REGISTRY._do_register('Sequential', nn.Sequential)


@MODULE_REGISTRY.register()
class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 pad_idx,
                 scale_grad_by_freq: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, pad_idx, scale_grad_by_freq=scale_grad_by_freq)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        inputs = self.embed(inputs) * math.sqrt(self.embed_dim)
        inputs = self.dropout(inputs)
        return inputs


@MODULE_REGISTRY.register()
class LinearClassifier(nn.Linear):
    def __init__(self, embed_dim: int, vocab_size: int, bias: bool = True) -> None:
        super().__init__(embed_dim, vocab_size, bias=bias)
