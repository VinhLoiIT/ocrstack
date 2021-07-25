from typing import List, Optional, Union

import torch
from ocrstack.data.collate import Batch
from ocrstack.models.base import BaseModel
from ocrstack.models.layers.translator import ITranslator
from torch.utils.data.dataloader import DataLoader


class Visualizer:
    def __init__(self,
                 data_loader: DataLoader,
                 translator: Optional[ITranslator] = None,
                 writers: List['BaseWriter'] = []):
        self.translator = translator
        self.data_loader = data_loader
        self.writers = writers

    @torch.no_grad()
    def visualize(self, model, device, num_iter_visualize=None):
        # type: (BaseModel, torch.device, Optional[int]) -> None
        model.eval()

        for writer in self.writers:
            writer.start()

        batch: Batch
        for i, batch in enumerate(self.data_loader):
            batch = batch.to(device)
            model_outputs = model.predict(batch)

            if self.translator is not None:
                model_outputs = self.translator.translate(model_outputs)

            for writer in self.writers:
                writer.visualize(batch, model_outputs)

            if num_iter_visualize is not None and (i + 1) >= num_iter_visualize:
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

    def __init__(self, meta_fields: Union[str, List[str]] = []):
        if isinstance(meta_fields, str):
            meta_fields = [meta_fields]
        self.meta_fields = meta_fields

    def start(self):
        print('-' * 120)

    def visualize(self, batch: Batch, model_outputs):
        predicts, _ = model_outputs
        for metadata, text_str, predict in zip(batch.metadata, batch.text_str, predicts):
            s = ''
            for field in self.meta_fields:
                s += f'{field}: {metadata[field]}, '
            s += f'Text: {text_str}, '
            s += f'Predict: {predict}'
            print(s)

    def end(self):
        print('-' * 120)
