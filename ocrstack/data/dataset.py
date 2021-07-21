import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    'OCRDataset',
    'CSVDataset',
    'DummyDataset',
    'glob'
]


def glob(folder: Union[Path, str], suffixes: List[str]) -> List[Path]:
    if os.name != 'nt':
        suffixes = suffixes + [p.upper() for p in suffixes]
    suffixes = ['**/*.{}'.format(p) for p in suffixes]

    return sum([sorted(list(Path(folder).glob(p))) for p in suffixes], [])


class OCRDataset(Dataset):
    def __init__(self,
                 image_paths: Union[Path, List[Path], List[str]],
                 image_transform: Optional[Callable] = None,
                 text_transform: Optional[Callable] = None,
                 text_file_suffix: str = 'txt',
                 encoding: str = 'utf8'
                 ):
        super(OCRDataset, self).__init__()
        self.logger = logging.getLogger('OCRDataset')
        if isinstance(image_paths, list):
            assert len(image_paths) > 0, "Image Paths should not be empty"
            image_paths = list(map(Path, image_paths))
        elif isinstance(image_paths, Path):
            # TODO: glob image in dir
            raise NotImplementedError()

        self.image_paths = image_paths
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.text_file_suffix = text_file_suffix
        self.encoding = encoding

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)

        text_path = image_path.with_suffix(f'.{self.text_file_suffix}')
        with open(text_path, 'rt', encoding=self.encoding) as f:
            raw_text = f.readline().rstrip()

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

        image_path = self.image_dir.joinpath(image_name)
        image = Image.open(image_path).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)

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
