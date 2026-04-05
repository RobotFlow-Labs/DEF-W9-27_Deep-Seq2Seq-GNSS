# PRD-05: API Serving and Dockerization

> Module: deep-seq2seq-gnss | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Expose online spoofing detection via a service interface suitable for ANIMA pipeline integration.

## Context (from paper)
Online detection behavior must be callable in operational systems.
Paper reference: Abstract + Section I (online detectors).

## Acceptance Criteria
- [ ] `/health` and `/predict` endpoints.
- [ ] Request payload supports `psr` and `presence` arrays.
- [ ] Dockerfile + compose profile for serving.
- [ ] CPU fallback mode documented.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_deep_seq2seq_gnss/api.py` | FastAPI serving app | I | ~180 |
| `Dockerfile.serve` | serving image | — | ~80 |
| `docker-compose.serve.yml` | compose service | — | ~50 |
| `.env.serve.example` | runtime vars | — | ~30 |
