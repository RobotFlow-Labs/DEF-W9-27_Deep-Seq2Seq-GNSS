from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import AppConfig, load_config
from .data import ATTACK_TO_ID, SyntheticGnssDataset
from .models import build_model
from .preprocess import build_features


def binary_metrics(pred: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = pred.long().reshape(-1)
    labels = labels.long().reshape(-1)

    total = labels.numel()
    error = (pred != labels).float().mean().item()

    negatives = (labels == 0)
    positives = (labels == 1)

    fp = ((pred == 1) & negatives).sum().item()
    fn = ((pred == 0) & positives).sum().item()

    fa = fp / max(1, negatives.sum().item())
    md = fn / max(1, positives.sum().item())
    return {"error": error, "fa": fa, "md": md, "count": float(total)}


def evaluate_model(model: torch.nn.Module, loader: DataLoader, threshold: float, device: torch.device) -> dict[str, Any]:
    model.eval()

    all_pred = []
    all_labels = []
    by_type = {"targeted": [], "regional": []}

    with torch.no_grad():
        for batch in loader:
            psr = batch["psr"].to(device)
            presence = batch["presence"].to(device)
            labels = batch["labels"].to(device)
            attack_type = batch["attack_type"].to(device)

            features = build_features(psr, presence)
            logits = model(features, presence.bool())
            probs = torch.softmax(logits, dim=-1)[..., 1]
            pred = (probs >= threshold).long()

            all_pred.append(pred.cpu())
            all_labels.append(labels.cpu())

            for name, aid in (("targeted", ATTACK_TO_ID["targeted"]), ("regional", ATTACK_TO_ID["regional"])):
                keep = (attack_type == aid)
                if keep.any():
                    by_type[name].append((pred[keep].cpu(), labels[keep].cpu()))

    total_pred = torch.cat(all_pred, dim=0)
    total_labels = torch.cat(all_labels, dim=0)

    out = {"total": binary_metrics(total_pred, total_labels)}
    for key in ("targeted", "regional"):
        if by_type[key]:
            pred_parts = [x[0] for x in by_type[key]]
            label_parts = [x[1] for x in by_type[key]]
            out[key] = binary_metrics(torch.cat(pred_parts, dim=0), torch.cat(label_parts, dim=0))
        else:
            out[key] = {"error": 0.0, "fa": 0.0, "md": 0.0, "count": 0.0}
    return out


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def run_eval(cfg: AppConfig, checkpoint: str | None = None) -> dict[str, Any]:
    device = _resolve_device(cfg.train.device)
    model = build_model(cfg).to(device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])

    ds = SyntheticGnssDataset(count=cfg.data.test_size, cfg=cfg.data, seed=cfg.random.seed + 999)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False)

    return evaluate_model(model, loader, threshold=cfg.eval.threshold, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GNSS spoofing detector")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out", type=str, default="artifacts/reports/eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = run_eval(cfg, checkpoint=args.checkpoint)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
