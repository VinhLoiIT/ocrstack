from .config import Config
import torch


def generate_padding_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    B, S = len(lengths), lengths.max()
    padding_mask = torch.arange(0, S, device=lengths.device).expand(B, S) >= lengths.unsqueeze(-1)
    return padding_mask
