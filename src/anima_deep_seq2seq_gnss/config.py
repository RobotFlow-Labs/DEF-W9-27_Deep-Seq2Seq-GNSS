from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class RandomConfig:
    seed: int = 42


@dataclass
class DataConfig:
    seq_len: int = 768
    max_sats: int = 12
    missing_prob: float = 0.05
    train_size: int = 4096
    test_size: int = 512
    val_size: int = 512
    spoof_ratio: float = 0.2196
    attack_min_duration: int = 100
    attack_max_duration: int = 568
    shift_min_m: float = 300.0
    shift_max_m: float = 1000.0


@dataclass
class ModelConfig:
    fusion_mode: str = "early"  # early|late
    encoder_type: str = "mha"  # mha|lstm
    embedding_dim: int = 128
    quant_bins: int = 64
    num_heads: int = 8
    ff_hidden_dim: int = 1024
    num_modules: int = 4
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "auto"
    checkpoint_path: str = "artifacts/checkpoints/model.pt"


@dataclass
class EvalConfig:
    threshold: float = 0.5


@dataclass
class AppConfig:
    random: RandomConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig


DEFAULT_CONFIG = AppConfig(
    random=RandomConfig(),
    data=DataConfig(),
    model=ModelConfig(),
    train=TrainConfig(),
    eval=EvalConfig(),
)


def _merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _to_dict(cfg: AppConfig) -> dict:
    return {
        "random": vars(cfg.random),
        "data": vars(cfg.data),
        "model": vars(cfg.model),
        "train": vars(cfg.train),
        "eval": vars(cfg.eval),
    }


def load_config(path: str | Path | None = None) -> AppConfig:
    merged = _to_dict(DEFAULT_CONFIG)
    if path:
        with open(path, "rb") as f:
            user = tomllib.load(f)
        merged = _merge_dict(merged, user)
    return AppConfig(
        random=RandomConfig(**merged["random"]),
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        train=TrainConfig(**merged["train"]),
        eval=EvalConfig(**merged["eval"]),
    )
