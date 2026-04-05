# PRD-01: Foundation, Config, and Data Pipeline

> Module: deep-seq2seq-gnss | Priority: P0
> Depends on: None
> Status: ⬜ Not started

## Objective
Create a reproducible project foundation with configs, synthetic GNSS spoofing data generation, and preprocessing aligned to the paper.

## Context (from paper)
The paper's core contribution begins with a synthetic data generator supporting targeted and regional spoofing plus missing satellite signals.
Paper reference: Section V (Generator), Section VI (Input signal processing).

## Acceptance Criteria
- [ ] `pyproject.toml` defines an installable package and CLI entry points.
- [ ] Synthetic generator creates targeted and regional attacks with timing/offset ranges from Section V.
- [ ] Preprocessing implements second-difference + signed-log compression from Section VI.
- [ ] Presence indicators are included as inputs.
- [ ] Unit tests pass for config, data, and preprocessing.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `pyproject.toml` | package + dependencies | — | ~60 |
| `configs/default.toml` | default training config | IX | ~70 |
| `configs/paper.toml` | paper-aligned larger run config | IX | ~70 |
| `src/anima_deep_seq2seq_gnss/data.py` | synthetic GNSS generator | V | ~260 |
| `src/anima_deep_seq2seq_gnss/preprocess.py` | feature engineering | VI | ~180 |
| `tests/test_generator.py` | data tests | V | ~80 |
| `tests/test_preprocess.py` | preprocessing tests | VI | ~80 |

## Dependencies
- `torch`, `numpy`, `pydantic` optional, `typer` optional.

## Test Plan
- `pytest tests/test_generator.py -v`
- `pytest tests/test_preprocess.py -v`

## References
- Paper Section V: targeted/regional spoofing scenarios.
- Paper Section VI: second difference, sign-log compression, indicator variable.
