import logging
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    'PairFileDataset',
    'CSVDataset',
    'DummyDataset',
]


class PairFileDataset(Dataset):
    r"""A convenient dataset loader for loading samples by pair

    Your directory might look like this:

    .. code-block::

        data/train/00001.png
        data/train/00001.txt
        data/train/00002.png
        data/train/00002.txt
        [...]
        data/train/10000.png
        data/train/10000.txt

        data/val/00001.png
        data/val/00001.txt
        data/val/00002.png
        data/val/00002.txt
        [...]
        data/val/10000.png
        data/val/10000.txt

    Args:
        image_dir: a path to your image directory.
        image_pattern: a pattern for globing images. Default is `*.png`.
        image_transform: a callable to transform an image to tensor. Default is `None`
        text_transform: a callable to transform a text string to tensor. Default is `None`
        text_file_suffix: the corresponding text file suffix. Default is "txt"
        encoding: text file encoding. Default is "utf8"

    Hint:
        Since we use `pathlib` for path management, you might want to
        pass :code:`image_dir="**/*.png"` for recursively glob images

    Note:
        This class only support one-line text files. If your text file has multiple lines, you
        should write your own dataset class.

    Examples:
        You could use this class as follows:

        .. code-block:: python

            train_data = PairFileDataset('data/train')
            val_data = PairFileDataset('data/val')
    """
    def __init__(self,
                 image_dir: Union[Path, str],
                 image_pattern: str = '*.png',
                 image_transform: Optional[Callable] = None,
                 text_transform: Optional[Callable] = None,
                 text_file_suffix: str = 'txt',
                 encoding: str = 'utf8',
                 ):
        super(PairFileDataset, self).__init__()
        self.logger = logging.getLogger('PairFileDataset')
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(list(self.image_dir.glob(image_pattern)))
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.text_file_suffix = text_file_suffix
        self.encoding = encoding

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        if self.image_transform is not None:
            image = self.image_transform(image)

        text_path = image_path.with_suffix(f'.{self.text_file_suffix}')
        with open(text_path, 'rt', encoding=self.encoding) as f:
            raw_text = f.readline().rstrip()

        text = None
        if self.text_transform is not None:
            text = self.text_transform(raw_text)

        data = {
            'metadata': {
                'imagePath': str(image_path),
                'textPath': str(text_path),
            },
            'image': image,
            'text': text,
            'text_str': raw_text,
        }

        return data


class CSVDataset(Dataset):
    r"""A convenient CSV-like dataset loader

    Your directory might look like this:

    .. code-block::

        data/train_images/00001.png
        data/train_images/00002.png
        [...]
        data/train_images/10000.png

        data/train_split.csv

    Args:
        image_dir: a path to your image folder
        csv_path: a path to your CSV file
        image_transform: a callable to transform an image to tensor. Default is `None`
        text_transform: a callable to transform a text string to tensor. Default is `None`
        delimiter: delimiter to seperate columns in CSV file. Default is ','
        encoding: CSV file encoding. Default is 'utf8'

    Examples:
        You could use this class as follows:

        .. code-block:: python

            dataset = CSVDataset('data/train_images',
                                 'data/train_split.csv')
    """
    def __init__(self,
                 image_dir: Union[Path, str],
                 csv_path: Union[Path, str],
                 image_transform: Optional[Callable] = None,
                 text_transform: Optional[Callable] = None,
                 delimiter=',',
                 encoding: str = 'utf8'):
        super(CSVDataset, self).__init__()
        self.csv_path = csv_path
        self.image_dir = Path(image_dir)
        self.image_transform = image_transform
        self.text_transform = text_transform
        import csv
        with open(csv_path, 'rt', encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            self.rows = [row for row in reader]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        image_name, raw_text = self.rows[idx]

        image_path = self.image_dir / image_name
        image = Image.open(image_path)
        if self.image_transform is not None:
            image = self.image_transform(image)

        text = None
        if self.text_transform is not None:
            text = self.text_transform(raw_text)

        data = {
            'metadata': {
                'index': idx,
                'imagePath': str(image_path),
            },
            'image': image,
            'text': text,
            'text_str': raw_text,
        }

        return data


class DummyDataset(Dataset):

    '''
    This class provides a dummy data set for debugging and testing flow.
    '''

    def __init__(self, num_samples, image_channels, height, width, max_lengths, vocab_size, seq2seq=False):
        super(DummyDataset, self).__init__()
        self.images = torch.rand(num_samples, image_channels, height, width)
        if seq2seq:
            self.texts = torch.randint(0, vocab_size, (num_samples, max_lengths + 2))
        else:
            self.texts = torch.randint(0, vocab_size, (num_samples, max_lengths))
        self.space_idx = 0
        self.vocab_size = vocab_size
        self.seq2seq = seq2seq

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.seq2seq:
            raw_text = ''.join([' ' if item.item() == self.space_idx else '0' for item in self.texts[index][1:-1]])
        else:
            raw_text = ''.join([' ' if item.item() == self.space_idx else '0' for item in self.texts[index]])
        text = self.texts[index]
        return {
            'metadata': {
                'imagePath': f'imagePath_{index}',
                'textPath': f'textPath_{index}',
            },
            'image': self.images[index],
            'text': text,
            'text_str': raw_text,
        }
