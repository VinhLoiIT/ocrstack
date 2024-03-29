import torch
import torch.nn as nn

from ocrstack.core.builder import MODULE_REGISTRY


@MODULE_REGISTRY.register()
class TransformerPE1D(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, d_model
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0).contiguous()  # [1, T, E]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@MODULE_REGISTRY.register()
class TransformerPE2D(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # [E]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [B,C,H,W]
        '''
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = x.reshape(B * H, W, C)  # (B * H, W, C)
        x = x + self.pe[:, :W, :]
        x = x.reshape(B, H, W, C)   # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        return self.dropout(x)


@MODULE_REGISTRY.register()
class PEAdaptive2D(nn.Module):
    '''
    Adaptive 2D positional encoding
    https://arxiv.org/pdf/1910.04396.pdf
    '''

    def __init__(self, cnn_features, dropout=0.1, max_len=5000):
        super().__init__()
        self.alpha = self._make_pe(cnn_features)
        self.beta = self._make_pe(cnn_features)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

        pe = torch.zeros(max_len, cnn_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze_(1)        # (T, 1)
        div_term = torch.exp(torch.arange(0, cnn_features, 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / cnn_features))    # (E,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze_(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Shapes:
        -------
            x: [B,C,H,W]
        '''
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        alpha = self.alpha(x)  # [B,H,W,C]
        alpha = alpha.transpose(1, 2)  # [B,W,H,C]
        alpha = alpha.reshape(-1, alpha.size(-2), alpha.size(-1)) * self.pe[:, :alpha.size(-2), :]  # [BxW,H,C]
        alpha = alpha.reshape(B, W, H, C).permute(0, 3, 2, 1)  # [B,C,H,W]

        beta = self.beta(x)  # [B,H,W,C]
        beta = beta.view(-1, beta.size(-2), beta.size(-1)) * self.pe[:, :beta.size(-2), :]  # [B*H,W,C]
        beta = beta.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        pe = alpha + beta  # [B,C,H,W]
        return self.dropout(pe)

    def _make_pe(self, cnn_features):
        return nn.Sequential(
            nn.Linear(cnn_features, cnn_features, bias=False),
            nn.ReLU(),
            nn.Linear(cnn_features, cnn_features, bias=False),
            nn.Sigmoid(),
        )
