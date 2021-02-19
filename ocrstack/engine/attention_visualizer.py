import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class AttentionVisualizer(object):
    def __init__(self, mean, std, output_ext: str = 'png'):
        self.in_out: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}
        self.output_ext = output_ext
        self.input: Optional[torch.Tensor] = None
        self.input_size: Optional[Tuple[int, int]] = None
        self.feature_size: Optional[Tuple[int, int]] = None
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def setup(self, model: nn.Module):
        for name, module in model.named_modules():
            parts = name.split('.')
            if (attn_type := parts[-1]) in ['self_attn', 'multihead_attn']:
                if 'encoder' in parts[:-1]:
                    attn_name = 'encoder'
                elif 'decoder' in parts[:-1]:
                    attn_name = 'decoder'
                else:
                    attn_name = 'unknow'
                meta = {
                    'layer': parts[-2],
                    'attn_name': attn_name,
                    'attn_type': attn_type,
                    'name': name,
                }
                self._register_hook(module, meta)

            if len(parts) == 1 and parts[0] == 'img_emb':
                self._register_hook(module, {'name': parts[0]})

            if name == 'img_emb.layers.2':
                self._register_hook(module, {'name': name})

        logging.debug('Hook infos')
        for k, v in self.in_out.items():
            logging.debug('-' * 20)
            logging.debug(k)
            logging.debug(v)

    def _register_hook(self, module, meta):
        assert 'name' in meta.keys()
        logging.debug(f'Adding hook to {meta["name"]}')
        self.in_out[module] = {
            'inputs': None,
            'outputs': None,
        }
        self.meta[module] = meta
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, inputs, outputs):
        self.in_out[module]['inputs'] = inputs
        self.in_out[module]['outputs'] = outputs

    def visualize(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        for module, meta in self.meta.items():
            in_out = self.in_out[module]

            if meta['name'] == 'img_emb':
                self.input = in_out['inputs'][0]
                B, _, H, W = in_out['inputs'][0].shape
                assert B == 1, 'Visualize only work with batch_size = 1'
                self.input_size = (H, W)
                logging.info(f'Input Size: {self.input_size}')
            elif meta['name'] == 'img_emb.layers.2':
                B, _, H, W = in_out['inputs'][0].shape
                self.feature_size = (H, W)
                logging.info(f'Feature Size: {self.feature_size}')
            elif meta['attn_name'] == 'decoder' and meta['attn_type'] == 'multihead_attn':
                self.visualize_decoder_multihead_attn(in_out, meta, output_dir)

    def visualize_decoder_multihead_attn(self, in_out, meta, output_dir: Path):
        output_name = '_'.join([meta['attn_name'], meta['layer'], meta['attn_type']])
        logging.debug(f'Name: {output_name}')
        _, weights = in_out['outputs']
        weights = weights.detach().cpu()
        logging.debug(f'Shape: {weights.shape}')

        assert self.input is not None
        assert self.input_size is not None
        assert self.feature_size is not None

        B, T, S = weights.shape
        assert B == 1, 'TODO: update!'
        assert S == (self.feature_size[0] * self.feature_size[1]), f'{S} != {self.feature_size}'

        # weights = weights.view(B, T, self.feature_size[0], self.feature_size[1])
        weights = weights.view(B, T, self.feature_size[1], self.feature_size[0])
        weights = weights.transpose(-2, -1)
        weights = F.interpolate(weights, self.input_size, mode='bilinear', align_corners=False)

        sample = self.input[0]
        sample = ((sample * self.std) + self.mean).permute(1, 2, 0)     # [H, W, C]
        sample = sample.clamp(0, 1)
        weights = weights[0]                        # [T, H, W]

        for step in range(len(weights)):
            weight = weights[step]
            weight = np.expand_dims(weight, -1)

            plt.figure(111, figsize=(15, 10))
            plt.imshow(sample, alpha=0.3)
            plt.imshow(weight, cmap='inferno', alpha=0.7)

            output_path = output_dir.joinpath(f'{output_name}_step_{step}.{self.output_ext}')
            plt.savefig(output_path)
