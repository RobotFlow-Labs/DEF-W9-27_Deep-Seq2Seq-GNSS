# Deep-Seq2Seq-GNSS (ANIMA Defense Module)

Local implementation of the paper **"Deep Sequence-to-Sequence Models for GNSS Spoofing Detection" (arXiv:2510.19890)** with:
- synthetic targeted + regional spoofing dataset generation,
- preprocessing pipeline (second difference + sign-log + presence indicator),
- LSTM and attention encoder variants,
- early and late fusion detectors,
- train/eval/infer CLIs.

## Layout
- `papers/` source paper
- `prds/` ANIMA PRDs
- `tasks/` build tasks
- `src/anima_deep_seq2seq_gnss/` implementation
- `tests/` unit tests
- `configs/` runtime configs

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

pytest -q
python -m anima_deep_seq2seq_gnss.train --config configs/debug.toml
python -m anima_deep_seq2seq_gnss.evaluate --config configs/debug.toml --checkpoint artifacts/checkpoints/model.pt
python -m anima_deep_seq2seq_gnss.infer --config configs/debug.toml --checkpoint artifacts/checkpoints/model.pt
```

## Configs
- `configs/debug.toml`: tiny and fast local smoke run.
- `configs/default.toml`: practical local baseline.
- `configs/paper.toml`: paper-like scale knobs (heavy).

## Notes
- The paper does not provide an official repository; this implementation is paper-faithful but pragmatic.
- The code is ready for migration to a CUDA server with the same API and config surface.
