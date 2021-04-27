import os
from pathlib import Path
from typing import List, Optional, Union, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import logging


__all__ = [
    'OCRDataset',
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
                'rawText': raw_text,
            },
            'image': image,
            'text': text,
        }

        return data


class DummyDataset(Dataset):
    def __init__(self, num_samples, image_channels, height, width, max_lengths, vocab_size, onehot: bool = True):
        super(DummyDataset, self).__init__()
        self.images = torch.rand(num_samples, image_channels, height, width)
        self.texts = torch.randint(0, vocab_size, (num_samples, max_lengths))
        if onehot:
            self.onehot_texts = F.one_hot(self.texts, vocab_size)
        else:
            self.onehot_texts = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            'metadata': {
                'imagePath': 'tempPath',
                'textPath': 'textPath',
                'rawText': 'dummyText',
            },
            'image': self.images[index],
            'text': self.onehot_texts[index] if self.onehot_texts is not None else self.texts[index],
        }
