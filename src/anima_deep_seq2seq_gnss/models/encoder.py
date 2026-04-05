from __future__ import annotations

import torch
from torch import nn


class TemporalSatelliteAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.temporal = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.satellite = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, S, E]
        mask: [B, T, S] bool
        """
        b, t, s, e = x.shape

        xt = x.permute(0, 2, 1, 3).reshape(b * s, t, e)
        mt = mask.permute(0, 2, 1).reshape(b * s, t)
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        yt, _ = self.temporal(
            xt,
            xt,
            xt,
            key_padding_mask=~mt,
            attn_mask=causal,
            need_weights=False,
        )
        yt = yt.reshape(b, s, t, e).permute(0, 2, 1, 3)

        xs = x.reshape(b * t, s, e)
        ms = mask.reshape(b * t, s)
        ys, _ = self.satellite(xs, xs, xs, key_padding_mask=~ms, need_weights=False)
        ys = ys.reshape(b, t, s, e)

        return 0.5 * (yt + ys)


class LSTMSatelliteEncoder(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, s, e = x.shape
        xs = x.permute(0, 2, 1, 3).reshape(b * s, t, e)
        ys, _ = self.lstm(xs)
        ys = self.dropout(ys)
        ys = ys.reshape(b, s, t, e).permute(0, 2, 1, 3)
        return ys * mask.unsqueeze(-1).float()


class EncoderModule(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_heads: int,
        encoder_type: str,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if encoder_type == "lstm":
            self.core = LSTMSatelliteEncoder(embed_dim, dropout=dropout)
        elif encoder_type == "mha":
            self.core = TemporalSatelliteAttention(embed_dim, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.core(self.norm1(x), mask)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x * mask.unsqueeze(-1).float()


class EncoderStack(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_heads: int,
        encoder_type: str,
        num_modules: int,
        dropout: float,
    ):
        super().__init__()
        self.modules_ = nn.ModuleList(
            [
                EncoderModule(
                    embed_dim=embed_dim,
                    ff_hidden_dim=ff_hidden_dim,
                    num_heads=num_heads,
                    encoder_type=encoder_type,
                    dropout=dropout,
                )
                for _ in range(num_modules)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for mod in self.modules_:
            x = mod(x, mask)
        return x
