import torch.nn as nn
from ocrstack.data.collate import Batch


class BaseModel(nn.Module):

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def example_inputs(self):
        pass

    def train_batch(self, batch: Batch):
        pass

    def predict(self, batch: Batch):
        pass
