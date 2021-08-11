import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def multihead_attention(queries: Tensor,
                        keys: Tensor,
                        values: Tensor,
                        score_func: Callable[[Tensor, Tensor], Tensor],
                        num_heads: int = 1,
                        q_padding_mask: Optional[Tensor] = None,
                        key_padding_mask: Optional[Tensor] = None,
                        attn_mask: Optional[Tensor] = None,
                        out_weights: bool = False,
                        ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    queries: (B, T, E)
    keys: (B, S, E)
    values: (B, S, E)
    score_func: a callable receive queries and keys to compute relation score
    num_heads: the number of head if multihead attention
    q_padding_mask: (B, T)
    k_padding_mask: (B, S)
    attn_mask: (B, T, S)
    out_weights: return attention weight or not
    """
    assert queries.size(-1) == keys.size(-1) and keys.size(-1) == values.size(-1)
    embed_dim = queries.size(-1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim

    B = queries.size(0)
    T = queries.size(1)
    S = keys.size(1)
    if num_heads > 1:
        queries = queries.reshape(B, T, num_heads, head_dim)                    # (B, T, N, H)
        queries = queries.transpose(1, 2).reshape(B, num_heads, T, head_dim)    # (B, N, T, H)

        keys = keys.reshape(B, S, num_heads, head_dim)                          # (B, S, N, H)
        keys = keys.transpose(1, 2).reshape(B, num_heads, S, head_dim)          # (B, N, S, H)

        values = values.reshape(B, S, num_heads, head_dim)                      # (B, S, N, H)
        values = values.transpose(1, 2).reshape(B, num_heads, S, head_dim)      # (B, N, S, H)
    else:
        queries = queries.unsqueeze(1)                                          # (B, N=1, T, H=E)
        keys = keys.unsqueeze(1)                                                # (B, N=1, S, H=E)
        values = values.unsqueeze(1)                                            # (B, N=1, S, H=E)

    scores = score_func(queries, keys)                                          # (B, N, T, S)

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)           # (B, 1, 1, S)
        scores = scores.masked_fill(key_padding_mask, float('-inf'))            # (B, N, T, S)

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(1)                                      # (B, N=1, T, S)
        scores = scores.masked_fill(~attn_mask, float('-inf'))                  # (B, N, T, S)

    weights = F.softmax(scores, dim=-1)                                         # (B, N, T, S)

    if q_padding_mask is not None:
        q_padding_mask = q_padding_mask.unsqueeze(1).unsqueeze(-1)              # (B, 1, T, 1)
        weights = weights.masked_fill(q_padding_mask, 0.0)                      # (B, N, T, S)

    context = torch.matmul(weights, values)                                     # (B, N, T, H)
    context = context.transpose(1, 2).reshape(B, T, embed_dim)                  # (B, T, E)

    if out_weights:
        return context, weights
    else:
        return context, None


def additive_score(queries: Tensor,
                   keys: Tensor,
                   W_q: Tensor,
                   W_k: Tensor,
                   v_a: Tensor,
                   bias_q: Optional[Tensor] = None,
                   bias_k: Optional[Tensor] = None,
                   bias_a: Optional[Tensor] = None,
                   ) -> Tensor:
    '''
    Input:
    - queries: [*, T, A]
    - keys: [*, S, A]
    - attn_mask: [*, T, S] - BoolTensor, value True for where T can attention at S
    Output:
    - score: [*, T, S]
    '''
    keys = F.linear(keys, W_k, bias_k)          # [*, S, A]
    queries = F.linear(queries, W_q, bias_q)    # [*, T, A]

    keys = keys.unsqueeze(-3)                    # [*, 1, S, A]
    queries = queries.unsqueeze(-2)              # [*, T, 1, A]

    score = F.linear(torch.tanh(queries + keys), v_a, bias_a)     # [*, T, S, 1]
    score = score.squeeze(-1)                                     # [*, T, S]
    return score


def dot_product_score(queries: Tensor, keys: Tensor, scaled: bool = False):
    '''
    Input:
    - queries: [*, T, A]
    - keys: [*, S, A]
    Output:
    - score: [*, T, S]
    '''
    # [*, T, A] x [*, A, S] = [*, T, S]
    if scaled:
        attn_dim = queries.size(-1)
        queries = queries / math.sqrt(attn_dim)
    score = torch.matmul(queries, keys.transpose(-1, -2))       # [*, T, S]
    return score


class Attention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 1,
                 k_dim: Optional[int] = None,
                 v_dim: Optional[int] = None,
                 bias_q: bool = True,
                 bias_k: bool = True,
                 bias_v: bool = True,
                 ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f'embed_dim={embed_dim} must be divisible by num_heads={num_heads}')

        self.embed_dim = embed_dim
        self.k_dim = embed_dim if k_dim is None else k_dim
        self.v_dim = embed_dim if v_dim is None else v_dim

        self.bias_q = bias_q
        self.bias_k = bias_k
        self.bias_v = bias_v

        self.in_proj_q = nn.Linear(embed_dim, embed_dim, bias=bias_q)
        self.in_proj_k = nn.Linear(self.k_dim, embed_dim, bias=bias_k)
        self.in_proj_v = nn.Linear(self.v_dim, embed_dim, bias=bias_v)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias_q)

    def compute_scores(self, queries, keys):
        raise NotImplementedError()

    def forward(self,
                queries,
                keys,
                values,
                q_padding_mask: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                out_weights: bool = True,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Shapes:
        -------
        - queries: (B, T, embed_dim)
        - keys: (B, S, k_dim)
        - values: (B, S, v_dim)
        - q_padding_mask: (B, T). BoolTensor where True value indicates padding locations, False otherwise.
        - key_padding_Mask: (B, S). BoolTensor where True value indicates padding locations, False otherwise.
        - attn_mask: (B, T, S). BoolTensor where True value indicates attention-able locations between queries and keys
        """
        queries = self.in_proj_q(queries)
        keys = self.in_proj_k(keys)
        values = self.in_proj_v(values)

        context, weights = multihead_attention(queries, keys, values, self.compute_scores,
                                               self.num_heads, q_padding_mask, key_padding_mask, attn_mask,
                                               out_weights)

        context = self.out_proj(context)

        return context, weights


class ScaledDotProductAttention(Attention):

    def compute_scores(self, queries, keys):
        return dot_product_score(queries, keys, True)


class DotProductAttention(Attention):

    def compute_scores(self, queries, keys):
        return dot_product_score(queries, keys, False)


class AdditiveAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.qk_proj = nn.Linear(self.head_dim, 1, bias=False)

    def compute_scores(self, queries, keys):
        return additive_score(queries, keys, self.q_proj.weight, self.k_proj.weight,
                              self.qk_proj.weight, self.q_proj.bias, self.k_proj.bias, self.qk_proj.bias)
