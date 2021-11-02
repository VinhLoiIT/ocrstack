from torch.nn import CrossEntropyLoss

from ocrstack.core.builder import LOSS_REGISTRY

LOSS_REGISTRY.register(CrossEntropyLoss)
