import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, nhead):
        super(MultiheadAttention, self).__init__()
        self.nhead = nhead

    def score(self, queries, keys):
        raise NotImplementedError()

    def forward(self, queries, keys, values, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [B, T, A]
        :param keys: [B, S, A]
        :param values: [B, S, A]
        :param attn_mask: [B,T,S]
        Output:
        - values: [B, T, C]
        - weights: [B, T, S] if output_weights = True else None
        '''
        weights = self.score(queries, keys)  # [B,T,S]
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            weights += attn_mask
        weights = F.softmax(weights, dim=-1)
        values = weights.bmm(values)  # [B, T, A]
        if output_weights:
            return values, weights
        else:
            return values, None


class MultiheadAdditiveAttention(MultiheadAttention):
    def __init__(self, embed_dim: int, nhead: int):
        super().__init__(embed_dim)
        self.head_size = embed_dim // nhead
        self.q_proj = nn.ModuleList([nn.Linear(self.embed_dim, self.head_size) for _ in range(nhead)])

        self.Wa = nn.Linear(embed_dim, embed_dim)
        self.Ua = nn.Linear(embed_dim, embed_dim)
        self.va = nn.Linear(embed_dim, 1)

    def score(self, queries, keys):
        '''
        Input:
        - queries: [B, T, A]
        - keys: [B, S, A]
        - attn_mask: [B, T, S] - BoolTensor, value True for where T can attention at S
        Output:
        - weights: [B, T, S]
        '''
        keys = self.Wa(keys)  # [B,S,A]
        queries = self.Ua(queries)  # [B,T,A]

        keys = keys.unsqueeze(1)  # [B,1,S,A]
        queries = queries.unsqueeze(2)  # [B,T,1,A]

        weights = self.va(torch.tanh(queries + keys))  # [B,T,S,1]
        weights = weights.squeeze(-1)  # [B,T,S]
        return weights


class ScaleDotProductAttention(MultiheadAttention):
    def __init__(self, attn_size):
        super().__init__(attn_size)

    def score(self, queries, keys):
        '''
        Input:
        - queries: [B, T, A]
        - keys: [B, S, A]
        Output:
        - weights: [B, T, S]
        '''
        attn_dim = queries.size(-1)
        # [B,T,A] x [B,A,S] = [B,T,S]
        matmul = queries.bmm(keys.transpose(1, 2))
        scaled = matmul / (attn_dim**0.5)  # [B,T,S]

        return scaled


class MultiHeadAttention(nn.Module):
    def __init__(self, attn, nhead):
        super().__init__()
        self.head_size = attn.attn_size
        self.nhead = nhead
        self.attn_size = self.nhead * self.head_size

        self.heads = nn.ModuleList([attn for _ in range(nhead)])
        self.q_proj = nn.ModuleList([nn.Linear(self.attn_size, self.head_size) for _ in range(nhead)])
        self.k_proj = nn.ModuleList([nn.Linear(self.attn_size, self.head_size) for _ in range(nhead)])
        self.v_proj = nn.ModuleList([nn.Linear(self.attn_size, self.head_size) for _ in range(nhead)])
        self.o_proj = nn.Linear(self.attn_size, self.attn_size)

    def forward(self, queries, keys, values, attn_mask=None, output_weights=False):
        '''
        Input:
        :param queries: [B, T, A]
        :param keys: [B, S, A]
        Output:
        - values: [B, T, A]
        - weights: [nhead, B, T, S]
        '''
        q_projected = [q_proj(queries) for q_proj in self.q_proj]
        k_projected = [k_proj(keys) for k_proj in self.k_proj]
        v_projected = [v_proj(values) for v_proj in self.v_proj]

        head_outputs = [head(q, k, v, attn_mask, output_weights)
                        for head, q, k, v in zip(self.heads, q_projected, k_projected, v_projected)]
        values, weights = list(zip(*head_outputs))
        # values (list): nhead * [B,T,head_attn_size]
        # weights (list): nhead * [B,T,S]

        values = torch.cat(values, -1)  # [B,T,A]
        values = self.o_proj(values)  # [B,T,A]
        if output_weights:
            weights = torch.stack(weights, dim=0)  # [nhead,B,T,S]
            return values, weights
        else:
            return values, None


def attention(queries, keys, values, score_func, attn_mask, queries_padding_mask, keys_padding_mask):
    '''
    Perform attention

    Inputs:
    - queries: (B, T, E)
    - keys: (B, S, E)
    - values: (B, T, E)
    - score_func:
    '''
    pass


def multihead_attention(queries, keys, values, score_func):
    pass
