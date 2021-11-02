import torch
import torch.nn.functional as F


def _create_dummy_image(batch_size: int = 1,
                        channels: int = 3,
                        height: int = 64,
                        width: int = 64,
                        dtype=torch.float,
                        device=torch.device('cpu')):
    if dtype == torch.uint8:
        return torch.randint(0, 256, (batch_size, channels, height, width), dtype=dtype, device=device)

    return torch.rand(batch_size, channels, height, width, dtype=dtype, device=device)


def _create_dummy_image_mask(batch_size: int = 1,
                             num_classes: int = 3,
                             height: int = 64,
                             width: int = 64,
                             one_hot: bool = False,
                             device=torch.device('cpu')):

    dummy = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long, device=device)
    if one_hot:
        dummy = F.one_hot(dummy, num_classes)
    return dummy


def _create_dummy_sequence(batch_size: int = 1,
                           length: int = 10,
                           channels: int = 5,
                           dtype=torch.float,
                           device=torch.device('cpu')):
    return torch.rand((batch_size, length, channels), dtype=dtype, device=device)


def _create_dummy_sequence_indices(batch_size: int = 1,
                                   length: int = 10,
                                   max_index: int = 5,
                                   one_hot: bool = False,
                                   device=torch.device('cpu')):
    dummy = torch.randint(0, max_index, (batch_size, length), device=device)
    dummy_onehot = None
    if one_hot:
        dummy_onehot = F.one_hot(dummy, max_index)
    return dummy, dummy_onehot
