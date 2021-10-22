import math
from typing import Dict, List

import torch.nn as nn

from ocrstack.core.builder import EMBEDDING_REGISTRY


@EMBEDDING_REGISTRY.register()
class ListModuleEmbedding(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


@EMBEDDING_REGISTRY.register()
class DictModuleEmbedding(nn.Module):
    def __init__(self, layers: Dict[str, nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(layers)

    def forward(self, inputs):
        return self.layers(inputs)


@EMBEDDING_REGISTRY.register()
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
