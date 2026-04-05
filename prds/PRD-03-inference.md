# PRD-03: Inference Pipeline and CLI

> Module: deep-seq2seq-gnss | Priority: P0
> Depends on: PRD-02
> Status: ⬜ Not started

## Objective
Provide training/inference entry points to run online spoofing detection per timestep from generated or loaded GNSS sequences.

## Context (from paper)
The models are designed for online detection and output spoofing confidence over time.
Paper reference: Abstract, Section I, Section VIII.

## Acceptance Criteria
- [ ] CLI for train/evaluate/infer exists and is documented.
- [ ] Inference loads model checkpoint and returns per-timestep spoof confidence.
- [ ] Latency and basic throughput are measurable locally.
- [ ] Output supports both early and late fusion models.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_deep_seq2seq_gnss/train.py` | training loop + checkpointing | IX | ~260 |
| `src/anima_deep_seq2seq_gnss/infer.py` | inference utility + CLI | I, VIII | ~180 |
| `README.md` | usage instructions | — | ~200 |

## Test Plan
- Train a tiny debug model for 1 epoch and run inference on one sample.
