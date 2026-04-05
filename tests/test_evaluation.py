import torch

from anima_deep_seq2seq_gnss.evaluate import binary_metrics


def test_binary_metrics_basic() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    pred = torch.tensor([0, 1, 1, 0])
    m = binary_metrics(pred, labels)
    assert abs(m["error"] - 0.5) < 1e-6
    assert abs(m["fa"] - 0.5) < 1e-6
    assert abs(m["md"] - 0.5) < 1e-6
