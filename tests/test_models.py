import torch

from anima_deep_seq2seq_gnss.config import load_config
from anima_deep_seq2seq_gnss.models import build_model
from anima_deep_seq2seq_gnss.preprocess import build_features


def _random_batch(b: int = 2, t: int = 64, s: int = 6):
    psr = torch.randn(b, t, s)
    presence = (torch.rand(b, t, s) > 0.15).float()
    feat = build_features(psr, presence)
    return feat, presence.bool()


def test_model_forward_early() -> None:
    cfg = load_config("configs/debug.toml")
    cfg.model.fusion_mode = "early"
    model = build_model(cfg)
    feat, mask = _random_batch()
    logits = model(feat, mask)
    assert logits.shape[:2] == feat.shape[:2]
    assert logits.shape[-1] == 2


def test_model_forward_late() -> None:
    cfg = load_config("configs/debug.toml")
    cfg.model.fusion_mode = "late"
    model = build_model(cfg)
    feat, mask = _random_batch()
    logits = model(feat, mask)
    assert logits.shape[:2] == feat.shape[:2]
    assert logits.shape[-1] == 2
