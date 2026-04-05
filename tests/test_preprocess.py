import torch

from anima_deep_seq2seq_gnss.preprocess import build_features, second_difference, sign_log_compress


def test_sign_log_compress_properties() -> None:
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    y = sign_log_compress(x)
    assert y[2].item() == 0.0
    assert y[0].item() < 0.0
    assert y[-1].item() > 0.0


def test_second_difference_simple_case() -> None:
    psr = torch.tensor([[0.0], [1.0], [3.0], [6.0]], dtype=torch.float32)
    presence = torch.ones_like(psr)
    sec = second_difference(psr, presence)
    expected = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
    assert torch.allclose(sec, expected)


def test_build_features_shape() -> None:
    psr = torch.randn(2, 16, 4)
    presence = (torch.rand(2, 16, 4) > 0.1).float()
    feat = build_features(psr, presence)
    assert feat.shape == (2, 16, 4, 2)
