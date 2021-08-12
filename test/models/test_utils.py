import torch
from ocrstack.models.utils import generate_padding_mask_from_lengths, generate_square_subsequent_mask


def test_generate_square_subsequent_mask():
    mask = generate_square_subsequent_mask(5)
    expected_mask = torch.tensor([[1, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0],
                                  [1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)

    mask = generate_square_subsequent_mask(1)
    expected_mask = torch.tensor([[True]])
    assert torch.equal(mask, expected_mask)


def test_generate_padding_mask_from_lengths():
    lengths = torch.tensor([3, 2, 5, 1])
    mask = generate_padding_mask_from_lengths(lengths)
    expected_mask = torch.tensor([[0, 0, 0, 1, 1],
                                  [0, 0, 1, 1, 1],
                                  [0, 0, 0, 0, 0],
                                  [0, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)

    lengths = torch.tensor([1])
    mask = generate_padding_mask_from_lengths(lengths)
    expected_mask = torch.tensor([[False]], dtype=torch.bool)
    assert torch.equal(mask, expected_mask)