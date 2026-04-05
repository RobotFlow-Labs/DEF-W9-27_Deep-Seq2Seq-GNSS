from __future__ import annotations

import torch
from torch import nn


class SoftQuantizer(nn.Module):
    """Differentiable scalar quantizer -> embedding."""

    def __init__(self, num_bins: int, embedding_dim: int):
        super().__init__()
        self.num_bins = num_bins
        self.embedding_dim = embedding_dim
        self.q = nn.Parameter(torch.linspace(-5.0, 5.0, num_bins))
        self.log_scale = nn.Parameter(torch.zeros(num_bins))
        self.codebook = nn.Parameter(torch.randn(num_bins, embedding_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x shape: [...]
        returns:
          embedding: [..., E]
          probs: [..., num_bins]
        """
        dist = torch.abs(x.unsqueeze(-1) - self.q)
        logits = -torch.exp(self.log_scale) * dist
        probs = torch.softmax(logits, dim=-1)
        emb = probs @ self.codebook
        return emb, probs
