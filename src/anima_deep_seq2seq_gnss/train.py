from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import AppConfig, load_config
from .data import SyntheticGnssDataset
from .evaluate import evaluate_model
from .models import build_model
from .preprocess import build_features


def _config_to_plain_dict(cfg: AppConfig) -> dict:
    return {
        "random": vars(cfg.random),
        "data": vars(cfg.data),
        "model": vars(cfg.model),
        "train": vars(cfg.train),
        "eval": vars(cfg.eval),
    }


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    grad_clip: float,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    count = 0

    for batch in loader:
        psr = batch["psr"].to(device)
        presence = batch["presence"].to(device)
        labels = batch["labels"].to(device)

        features = build_features(psr, presence)
        logits = model(features, presence.bool())
        loss = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running += loss.item()
        count += 1

    return running / max(1, count)


def run_train(cfg: AppConfig) -> None:
    torch.manual_seed(cfg.random.seed)
    device = _resolve_device(cfg.train.device)

    train_ds = SyntheticGnssDataset(cfg.data.train_size, cfg.data, seed=cfg.random.seed)
    val_ds = SyntheticGnssDataset(cfg.data.val_size, cfg.data, seed=cfg.random.seed + 111)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    best_error = float("inf")
    ckpt_path = Path(cfg.train.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, cfg.train.grad_clip, device)
        metrics = evaluate_model(model, val_loader, threshold=cfg.eval.threshold, device=device)
        val_error = metrics["total"]["error"]

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_error={val_error:.6f}")

        if val_error < best_error:
            best_error = val_error
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": _config_to_plain_dict(cfg),
                    "epoch": epoch,
                    "best_val_error": best_error,
                },
                ckpt_path,
            )

    print(f"best_val_error={best_error:.6f} checkpoint={ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNSS spoofing detector")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_train(cfg)


if __name__ == "__main__":
    main()
