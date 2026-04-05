# PRD-07: Production Hardening and CUDA Migration Readiness

> Module: deep-seq2seq-gnss | Priority: P2
> Depends on: PRD-04
> Status: ⬜ Not started

## Objective
Prepare the module for robust deployment and later CUDA-server optimization without changing functional behavior.

## Context (from paper)
Paper-level metrics are from simulation; production needs reliability controls and reproducibility for retraining.
Paper reference: Section X.

## Acceptance Criteria
- [ ] Deterministic training/inference settings documented.
- [ ] Checkpoint export path supports `.pt` and ONNX.
- [ ] Runtime safeguards for NaNs/missing channels.
- [ ] Monitoring hooks for spoofing rate drift.
- [ ] CUDA migration checklist created.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `docs/CUDA_MIGRATION.md` | server migration plan | X | ~180 |
| `docs/OPERATIONS.md` | runbook + failure modes | X | ~180 |
| `scripts/export_onnx.py` | model export utility | — | ~120 |
