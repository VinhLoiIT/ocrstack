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

    def compute_scores(self, queries, keys):
        raise NotImplementedError()

    def forward(self,
                queries,
                keys,
                values,
                q_padding_mask: Optional[Tensor] = None,
                k_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                out_weights: bool = False,
                ) -> Tuple[Tensor, Optional[Tensor]]:
        scores = self.compute_scores(queries, keys)
        return attention(scores, values, q_padding_mask, k_padding_mask, attn_mask, out_weights)


class DotProductAttention(Attention):
    def __init__(self, scaled: bool = False):
        super().__init__()
        self.scaled = scaled

    def compute_scores(self, queries, keys):
        return dot_product_score(queries, keys, self.scaled)


class AdditiveAttention(Attention):
    def __init__(self, attn_size: int, bias_q: bool = True, bias_k: bool = True, bias_v: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(attn_size, attn_size, bias=bias_q)
        self.k_proj = nn.Linear(attn_size, attn_size, bias=bias_k)
        self.qk_proj = nn.Linear(attn_size, 1, bias=bias_v)

    def compute_scores(self, queries, keys):
        return additive_score(queries, keys, self.q_proj.weight, self.k_proj.weight,
                              self.qk_proj.weight, self.q_proj.bias, self.k_proj.bias, self.qk_proj.bias)

# TODO: Support Multihead Attention
