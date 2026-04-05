from __future__ import annotations

import torch


def sign_log_compress(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def second_difference(psr: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """
    Compute second difference along time axis with visibility gating.

    psr: [B, T, S] or [T, S]
    presence: same shape as psr
    returns: same shape, first 2 timesteps are zeros
    """
    added_batch = False
    if psr.ndim == 2:
        psr = psr.unsqueeze(0)
        presence = presence.unsqueeze(0)
        added_batch = True

    psr = psr.float()
    presence_bool = presence.bool()
    out = torch.zeros_like(psr)

    d1 = psr[:, 1:, :] - psr[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]

    valid = presence_bool[:, 2:, :] & presence_bool[:, 1:-1, :] & presence_bool[:, :-2, :]
    out[:, 2:, :] = torch.where(valid, d2, torch.zeros_like(d2))

    if added_batch:
        out = out.squeeze(0)
    return out


def build_features(psr: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    """
    Return input features with channels:
    - channel 0: compressed second-difference pseudo-range
    - channel 1: presence indicator
    shape: [B, T, S, 2] or [T, S, 2]
    """
    added_batch = False
    if psr.ndim == 2:
        psr = psr.unsqueeze(0)
        presence = presence.unsqueeze(0)
        added_batch = True

    sec = second_difference(psr, presence)
    sec = sign_log_compress(sec)
    feat = torch.stack([sec, presence.float()], dim=-1)

    if added_batch:
        feat = feat.squeeze(0)
    return feat
