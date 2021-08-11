import pytest
import torch
from ocrstack.models.layers.attention import (AdditiveAttention, Attention,
                                              DotProductAttention, ScaledDotProductAttention,
                                              dot_product_score, multihead_attention)


@pytest.mark.parametrize('num_heads', (1, 2))
def test_multihead_attention(num_heads):
    torch.manual_seed(0)
    B, T, S = 2, 3, 4
    queries = torch.rand(B, T, 6)   # [B, T, E]
    keys = torch.rand(B, S, 6)      # [B, S, E]
    values = torch.rand(B, S, 6)    # [B, S, E]
    score_func = dot_product_score

    q_padding_mask = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.bool)
    assert q_padding_mask.shape == torch.Size([2, 3])

    key_padding_mask = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 1]], dtype=torch.bool)
    assert key_padding_mask.shape == torch.Size([2, 4])
    context, weights = multihead_attention(queries, keys, values, score_func, num_heads,
                                           q_padding_mask, key_padding_mask, out_weights=True)

    padding_mask = torch.bitwise_or(q_padding_mask.unsqueeze(-1), key_padding_mask.unsqueeze(-2))
    padding_mask = padding_mask.unsqueeze_(1).expand(B, num_heads, T, S)
    assert (weights.masked_select(padding_mask) == 0).all()
    assert context.shape == torch.Size((B, T, 6))


def test_not_divisible_numheads():
    with pytest.raises(ValueError):
        Attention(embed_dim=10, num_heads=3)


@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('src_length', (1, 2))
@pytest.mark.parametrize('tgt_length', (1, 3))
@pytest.mark.parametrize('k_dim', (10, 4))
@pytest.mark.parametrize('v_dim', (10, 4))
@pytest.mark.parametrize('num_heads', (1, 2))
@pytest.mark.parametrize('scaled', (True, False))
def test_dot_prod_attention(batch_size, src_length, tgt_length, k_dim, v_dim, num_heads, scaled):

    q = torch.rand(batch_size, tgt_length, 10)
    k = torch.rand(batch_size, src_length, k_dim)
    v = torch.rand(batch_size, src_length, v_dim)

    if scaled:
        attn = ScaledDotProductAttention(embed_dim=10, num_heads=num_heads, k_dim=k_dim, v_dim=v_dim)
    else:
        attn = DotProductAttention(embed_dim=10, num_heads=num_heads, k_dim=k_dim, v_dim=v_dim)

    context, weights = attn(q, k, v, out_weights=True)
    assert weights.shape == torch.Size((batch_size, num_heads, tgt_length, src_length))
    assert context.shape == torch.Size((batch_size, tgt_length, 10))


@pytest.mark.parametrize('batch_size', (1, 2))
@pytest.mark.parametrize('src_length', (1, 2))
@pytest.mark.parametrize('tgt_length', (1, 3))
@pytest.mark.parametrize('k_dim', (10, 4))
@pytest.mark.parametrize('v_dim', (10, 4))
@pytest.mark.parametrize('num_heads', (1, 2))
def test_additive_attention(batch_size, src_length, tgt_length, k_dim, v_dim, num_heads):

    q = torch.rand(batch_size, tgt_length, 10)
    k = torch.rand(batch_size, src_length, k_dim)
    v = torch.rand(batch_size, src_length, v_dim)

    attn = AdditiveAttention(embed_dim=10, num_heads=num_heads, k_dim=k_dim, v_dim=v_dim)
    context, weights = attn(q, k, v, out_weights=True)
    assert weights.shape == torch.Size((batch_size, num_heads, tgt_length, src_length))
    assert context.shape == torch.Size((batch_size, tgt_length, 10))
