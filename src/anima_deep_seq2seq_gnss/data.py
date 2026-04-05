from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


AttackType = Literal["clean", "targeted", "regional"]
ATTACK_TO_ID = {"clean": 0, "targeted": 1, "regional": 2}


@dataclass
class SequenceSample:
    psr: np.ndarray  # [T, S]
    presence: np.ndarray  # [T, S], bool
    labels: np.ndarray  # [T], {0,1}
    attack_type: AttackType


class SyntheticGnssGenerator:
    """Synthetic GNSS pseudo-range generator aligned with paper constraints."""

    def __init__(self, cfg: DataConfig, seed: int = 0):
        self.cfg = cfg
        self.base_rng = np.random.default_rng(seed)

    def _rng_for(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(int(self.base_rng.integers(0, 2**32 - 1)) + idx)

    def _generate_nominal(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t = self.cfg.seq_len
        s = self.cfg.max_sats

        base = rng.uniform(2.20e7, 2.30e7, size=(1, s))
        slope = rng.normal(0.0, 35.0, size=(1, s))
        time = np.arange(t, dtype=np.float64).reshape(-1, 1)

        # Correlated atmospheric-like noise + white receiver noise.
        rho = np.zeros((t, s), dtype=np.float64)
        atm = np.zeros((s,), dtype=np.float64)
        for i in range(t):
            atm = 0.97 * atm + rng.normal(0.0, 0.8, size=(s,))
            white = rng.normal(0.0, 0.7, size=(s,))
            rho[i] = base + slope * time[i] + atm + white

        # Presence mask with random outages and independent dropout.
        presence = rng.random((t, s)) > self.cfg.missing_prob
        for sat in range(s):
            if rng.random() < 0.35:
                outage_len = int(rng.integers(5, max(6, t // 8)))
                start = int(rng.integers(0, max(1, t - outage_len)))
                presence[start : start + outage_len, sat] = False

        rho = np.where(presence, rho, 0.0)
        return rho.astype(np.float32), presence.astype(bool)

    def _attack_window(self, rng: np.random.Generator) -> tuple[int, int]:
        t = self.cfg.seq_len
        dur = int(rng.integers(self.cfg.attack_min_duration, self.cfg.attack_max_duration + 1))
        dur = min(dur, t - 4)
        start = int(rng.integers(2, max(3, t - dur - 1)))
        end = start + dur
        return start, end

    def _apply_targeted(self, rho: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t, s = rho.shape
        start, end = self._attack_window(rng)
        shift = float(rng.uniform(self.cfg.shift_min_m, self.cfg.shift_max_m))

        labels = np.zeros((t,), dtype=np.int64)
        labels[start:end] = 1

        curve = np.zeros((t,), dtype=np.float32)
        span = max(1, end - start)
        tau = np.linspace(0.0, 1.0, span, dtype=np.float32)
        curve[start:end] = shift * (1.0 - np.cos(np.pi * tau)) * 0.5

        per_sat_gain = rng.normal(1.0, 0.04, size=(s,)).astype(np.float32)
        attacked = rho + curve[:, None] * per_sat_gain[None, :]
        return attacked.astype(np.float32), labels

    def _apply_regional(self, rho: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t, s = rho.shape
        start, end = self._attack_window(rng)
        shift = float(rng.uniform(self.cfg.shift_min_m, self.cfg.shift_max_m))

        labels = np.zeros((t,), dtype=np.int64)
        labels[start:end] = 1

        step = np.zeros((t,), dtype=np.float32)
        step[start:end] = shift

        sat_rotation = rng.uniform(-0.12, 0.12, size=(s,)).astype(np.float32)
        gain = (1.0 + sat_rotation).astype(np.float32)
        attacked = rho + step[:, None] * gain[None, :]
        return attacked.astype(np.float32), labels

    def generate_sample(self, idx: int, attack_type: AttackType | None = None) -> SequenceSample:
        rng = self._rng_for(idx)
        rho, presence = self._generate_nominal(rng)

        if attack_type is None:
            is_spoof = rng.random() < self.cfg.spoof_ratio
            if not is_spoof:
                attack_type = "clean"
            else:
                attack_type = "targeted" if rng.random() < 0.5 else "regional"

        if attack_type == "targeted":
            rho, labels = self._apply_targeted(rho, rng)
        elif attack_type == "regional":
            rho, labels = self._apply_regional(rho, rng)
        else:
            labels = np.zeros((self.cfg.seq_len,), dtype=np.int64)

        rho = np.where(presence, rho, 0.0).astype(np.float32)
        return SequenceSample(psr=rho, presence=presence, labels=labels, attack_type=attack_type)


class SyntheticGnssDataset(Dataset):
    def __init__(self, count: int, cfg: DataConfig, seed: int = 0):
        self.count = count
        self.cfg = cfg
        self.gen = SyntheticGnssGenerator(cfg, seed=seed)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.gen.generate_sample(idx)
        return {
            "psr": torch.from_numpy(sample.psr),
            "presence": torch.from_numpy(sample.presence.astype(np.float32)),
            "labels": torch.from_numpy(sample.labels),
            "attack_type": torch.tensor(ATTACK_TO_ID[sample.attack_type], dtype=torch.long),
        }
