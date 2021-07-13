import torch
from ocrstack.opts.attention import attention


def test_attention():
    torch.manual_seed(0)
    scores = torch.rand(2, 3, 4)    # [B, T, S]
    values = torch.rand(2, 4, 5)    # [B, S, E]

    q_padding_mask = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool)
    assert q_padding_mask.shape == torch.Size([2, 3])

    k_padding_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=torch.bool)
    assert k_padding_mask.shape == torch.Size([2, 4])

    padding_mask = torch.bitwise_or(q_padding_mask.unsqueeze(-1), k_padding_mask.unsqueeze(-2))

    values, weights = attention(scores, values, q_padding_mask, k_padding_mask, out_weights=True)
    assert (weights.masked_select(padding_mask) == 0).all()
