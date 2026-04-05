from __future__ import annotations

import torch
from torch import nn

from ..config import AppConfig
from .encoder import EncoderStack
from .quantizer import SoftQuantizer


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    w = mask.float()
    denom = w.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (x * w.unsqueeze(-1)).sum(dim=dim) / denom


class _BaseDetector(nn.Module):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        mcfg = cfg.model
        dcfg = cfg.data
        e = mcfg.embedding_dim

        self.max_seq_len = dcfg.seq_len
        self.max_sats = dcfg.max_sats

        self.quantizer = SoftQuantizer(num_bins=mcfg.quant_bins, embedding_dim=e)
        self.presence_proj = nn.Linear(1, e)
        self.time_embed = nn.Embedding(dcfg.seq_len, e)
        self.sat_embed = nn.Embedding(dcfg.max_sats, e)
        self.dropout = nn.Dropout(mcfg.dropout)

        self.encoder = EncoderStack(
            embed_dim=e,
            ff_hidden_dim=mcfg.ff_hidden_dim,
            num_heads=mcfg.num_heads,
            encoder_type=mcfg.encoder_type,
            num_modules=mcfg.num_modules,
            dropout=mcfg.dropout,
        )

    def _embed(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """
        features: [B,T,S,2] with channel0=value, channel1=presence
        presence: [B,T,S] bool
        """
        b, t, s, _ = features.shape
        value = features[..., 0]
        present = features[..., 1:2]

        v_emb, _ = self.quantizer(value)
        p_emb = self.presence_proj(present)

        tidx = torch.arange(t, device=features.device)
        sidx = torch.arange(s, device=features.device)
        t_emb = self.time_embed(tidx)[None, :, None, :]
        s_emb = self.sat_embed(sidx)[None, None, :, :]

        x = v_emb + p_emb + t_emb + s_emb
        x = self.dropout(x)
        x = self.encoder(x, presence)
        return x


class EarlyFusionDetector(_BaseDetector):
    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        self.head = nn.Linear(cfg.model.embedding_dim, 2)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        x = self._embed(features, presence)
        pooled = _masked_mean(x, presence, dim=2)  # [B,T,E]
        logits = self.head(pooled)
        return logits


class LateFusionDetector(_BaseDetector):
    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        e = cfg.model.embedding_dim
        self.sat_head = nn.Linear(e, 2)
        self.weight_head = nn.Linear(e, 1)

    def forward(self, features: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        x = self._embed(features, presence)  # [B,T,S,E]
        sat_logits = self.sat_head(x)  # [B,T,S,2]
        sat_weights = self.weight_head(x).squeeze(-1)  # [B,T,S]

        neg_inf = torch.finfo(sat_weights.dtype).min
        sat_weights = torch.where(presence, sat_weights, torch.full_like(sat_weights, neg_inf))
        alpha = torch.softmax(sat_weights, dim=2).unsqueeze(-1)

        logits = (alpha * sat_logits).sum(dim=2)  # [B,T,2]
        return logits


def build_model(cfg: AppConfig) -> nn.Module:
    if cfg.model.fusion_mode == "early":
        return EarlyFusionDetector(cfg)
    if cfg.model.fusion_mode == "late":
        return LateFusionDetector(cfg)
    raise ValueError(f"Unsupported fusion_mode={cfg.model.fusion_mode}")
