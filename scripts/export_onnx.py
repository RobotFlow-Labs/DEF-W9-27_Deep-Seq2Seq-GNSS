from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anima_deep_seq2seq_gnss.config import load_config
from anima_deep_seq2seq_gnss.models import build_model
from anima_deep_seq2seq_gnss.preprocess import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GNSS spoofing model to ONNX")
    parser.add_argument("--config", type=str, default="configs/debug.toml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/checkpoints/model.onnx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    b = 1
    t = cfg.data.seq_len
    s = cfg.data.max_sats

    psr = torch.randn(b, t, s)
    presence = torch.ones(b, t, s)
    features = build_features(psr, presence)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (features, presence.bool()),
        str(out_path),
        input_names=["features", "presence"],
        output_names=["logits"],
        dynamic_axes={"features": {1: "time"}, "presence": {1: "time"}, "logits": {1: "time"}},
        opset_version=17,
    )
    print(f"exported: {out_path}")


if __name__ == "__main__":
    main()
