from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .data import SyntheticGnssGenerator
from .models import build_model
from .preprocess import build_features


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNSS spoofing inference on one sample")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--attack-type", type=str, default=None, choices=[None, "clean", "targeted", "regional"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _resolve_device(cfg.train.device)

    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    gen = SyntheticGnssGenerator(cfg.data, seed=cfg.random.seed + 7)
    sample = gen.generate_sample(idx=args.sample_index, attack_type=args.attack_type)

    psr = torch.from_numpy(sample.psr).unsqueeze(0).to(device)
    presence = torch.from_numpy(sample.presence.astype("float32")).unsqueeze(0).to(device)
    features = build_features(psr, presence)

    with torch.no_grad():
        logits = model(features, presence.bool())
        probs = torch.softmax(logits, dim=-1)[0, :, 1].cpu().tolist()

    out = {
        "attack_type": sample.attack_type,
        "spoof_prob_preview": probs[:20],
        "num_timesteps": len(probs),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
