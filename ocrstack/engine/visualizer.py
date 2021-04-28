from typing import List, Optional

import torch
from ocrstack.data.collate import Batch
from ocrstack.models.base import BaseModel
from torch.utils.data.dataloader import DataLoader


class Visualizer:
    def __init__(self,
                 model: BaseModel,
                 data_loader: DataLoader,
                 device,
                 writers: List['BaseWriter'] = [],
                 num_iter_visualize: Optional[int] = None):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.writers = writers
        self.num_iter_visualize = num_iter_visualize or float('inf')

    @torch.no_grad()
    def visualize(self):
        self.model.eval()

        for writer in self.writers:
            writer.start()

        batch: Batch
        for i, batch in enumerate(self.data_loader):
            batch = batch.to(self.device)
            model_outputs = self.model(batch)
            for writer in self.writers:
                writer.visualize(batch, model_outputs)
            if (i + 1) >= self.num_iter_visualize:
                break

        for writer in self.writers:
            writer.end()


class BaseWriter:

    def start(self):
        pass

    def visualize(self, batch: Batch, model_outputs):
        pass

    def end(self):
        pass


class ConsoleWriter(BaseWriter):

    def __init__(self, col_filename: str = None, col_text: str = None):
        self.col_filename = col_filename
        self.col_text = col_text

    def start(self):
        print('-' * 120)

    def visualize(self, batch: Batch, model_outputs):
        predicts, _ = model_outputs
        for metadata, predict in zip(batch.metadata, predicts):
            s = ''
            if self.col_filename:
                s += f'File: {metadata[self.col_filename]}, '
            if self.col_text:
                s += f'Text: {metadata[self.col_text]}, '
            s += f'Predict: {predict}'
            print(s)

    def end(self):
        print('-' * 120)
