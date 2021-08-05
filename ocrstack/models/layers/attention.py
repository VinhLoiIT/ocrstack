from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def attention(scores: Tensor,
              values: Tensor,
              q_padding_mask: Optional[Tensor] = None,
              k_padding_mask: Optional[Tensor] = None,
              attn_mask: Optional[Tensor] = None,
              out_weights: bool = False,
              ) -> Tuple[Tensor, Optional[Tensor]]:

    if k_padding_mask is not None:
        k_padding_mask = k_padding_mask.unsqueeze(-2)       # [B, 1, S]
        scores = scores.masked_fill(k_padding_mask, float('-inf'))  # [B, T, S]

    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float('-inf'))

    weights = F.softmax(scores, dim=-1)

    if q_padding_mask is not None:
        q_padding_mask = q_padding_mask.unsqueeze(-1)       # [B, T, 1]
        weights = weights.masked_fill(q_padding_mask, 0.0)  # [B, T, S]

    values = weights.bmm(values)
    if out_weights:
        return values, weights
    else:
        return values, None


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
    - queries: [B, T, A]
    - keys: [B, S, A]
    - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
    Output:
    - score: [B, T, S]
    '''
    keys = F.linear(keys, W_k, bias_k)          # [B, S, A]
    queries = F.linear(queries, W_q, bias_q)    # [B, T, A]

    keys = keys.unsqueeze(1)                    # [B, 1, S, A]
    queries = queries.unsqueeze(2)              # [B, T, 1, A]

    score = F.linear(torch.tanh(queries + keys), v_a, bias_a)     # [B, T, S, 1]
    score = score.squeeze(-1)                                     # [B, T, S]
    return score


def dot_product_score(queries: Tensor, keys: Tensor, scaled: bool = False):
    '''
    Input:
    - queries: [B, T, A]
    - keys: [B, S, A]
    Output:
    - score: [B, T, S]
    '''
    # [B,T,A] x [B,A,S] = [B,T,S]
    if scaled:
        attn_dim = queries.size(-1)
        queries = queries / (attn_dim**0.5)
    score = queries.bmm(keys.transpose(1, 2))       # [B,T,S]
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

    def prepare_multihead(self, queries: Tensor, keys: Tensor, values: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B = queries.size(0)
        T = queries.size(1)
        S = keys.size(1)

        queries = queries.reshape(B, T, self.num_heads, self.head_dim)                      # (B, T, N, H)
        queries = queries.transpose(1, 2).reshape(B * self.num_heads, T, self.head_dim)     # (B * N, T, H)

        keys = keys.reshape(B, S, self.num_heads, self.head_dim)                      # (B, S, N, H)
        keys = keys.transpose(1, 2).reshape(B * self.num_heads, S, self.head_dim)     # (B * N, S, H)

        values = values.reshape(B, S, self.num_heads, self.head_dim)                      # (B, S, N, H)
        values = values.transpose(1, 2).reshape(B * self.num_heads, S, self.head_dim)     # (B * N, S, H)

        return queries, keys, values

    def forward(self,
                queries,
                keys,
                values,
                q_padding_mask: Optional[Tensor] = None,
                k_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                out_weights: bool = False,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Shapes:
        -------
        - queries: (B, T, embed_dim)
        - keys: (B, S, k_dim)
        - values: (B, S, v_dim)
        """
        queries = self.in_proj_q(queries)
        keys = self.in_proj_k(keys)
        values = self.in_proj_v(values)

        batch_size = queries.size(0)
        tgt_length = queries.size(1)

        if self.num_heads > 1:
            queries, keys, values = self.prepare_multihead(queries, keys, values)
            scores = self.compute_scores(queries, keys)     # (B, T, S)
            context, weights = attention(scores, values, q_padding_mask, k_padding_mask, attn_mask, out_weights)
            # context: (B * N, T, H)
            # weights: (B * N, T, H)
            context = context.reshape(batch_size, self.num_heads, tgt_length, self.head_dim)
            context = context.transpose(1, 2)                                           # (B, T, N, H)
            context = context.reshape(batch_size, tgt_length, self.embed_dim)           # (B, T, E)
            context = self.out_proj(context)                                            # (B, T, E)
        else:
            scores = self.compute_scores(queries, keys)     # (B, T, S)
            context, weights = attention(scores, values, q_padding_mask, k_padding_mask, attn_mask, out_weights)
            context = self.out_proj(context)

        return context, weights


class DotProductAttention(Attention):
    def __init__(self, scaled: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaled = scaled

    def compute_scores(self, queries, keys):
        return dot_product_score(queries, keys, self.scaled)


class AdditiveAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.qk_proj = nn.Linear(self.head_dim, 1, bias=False)

    def compute_scores(self, queries, keys):
        return additive_score(queries, keys, self.q_proj.weight, self.k_proj.weight,
                              self.qk_proj.weight, self.q_proj.bias, self.k_proj.bias, self.qk_proj.bias)
