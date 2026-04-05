from anima_deep_seq2seq_gnss.config import DataConfig
from anima_deep_seq2seq_gnss.data import SyntheticGnssGenerator


def _cfg() -> DataConfig:
    return DataConfig(seq_len=128, max_sats=8, train_size=10, test_size=4, val_size=4, attack_min_duration=20, attack_max_duration=60)


def test_generate_targeted_has_positive_labels() -> None:
    gen = SyntheticGnssGenerator(_cfg(), seed=1)
    sample = gen.generate_sample(0, attack_type="targeted")
    assert sample.psr.shape == (128, 8)
    assert sample.presence.shape == (128, 8)
    assert sample.labels.sum() > 0
    assert sample.attack_type == "targeted"


def test_generate_regional_has_positive_labels() -> None:
    gen = SyntheticGnssGenerator(_cfg(), seed=2)
    sample = gen.generate_sample(1, attack_type="regional")
    assert sample.labels.sum() > 0
    assert sample.attack_type == "regional"
