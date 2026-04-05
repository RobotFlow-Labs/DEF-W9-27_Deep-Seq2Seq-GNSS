# PRD-04: Evaluation and Benchmark Comparison

> Module: deep-seq2seq-gnss | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Implement reproducible evaluation that reports classification error, false alarm rate, and missed detection rate on total, targeted, and regional subsets.

## Context (from paper)
Paper reports metrics across total and per-attack-type subsets (Table I).
Paper reference: Section IX, Table I.

## Acceptance Criteria
- [ ] Evaluator computes error, FA, and MD.
- [ ] Reports per attack type and aggregate.
- [ ] Produces JSON/markdown summary and confusion counts.
- [ ] Includes threshold-based decision (default 0.5).

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_deep_seq2seq_gnss/evaluate.py` | benchmark evaluator | IX, Table I | ~220 |
| `tests/test_evaluation.py` | metric correctness tests | IX | ~90 |
| `artifacts/reports/.gitkeep` | report output dir | — | 1 |

## Test Plan
- Run `python -m anima_deep_seq2seq_gnss.evaluate --help`
- Validate metric functions on synthetic fixed labels.
