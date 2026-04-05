# PRD-02: Core Sequence-to-Sequence Spoofing Model

> Module: deep-seq2seq-gnss | Priority: P0
> Depends on: PRD-01
> Status: ⬜ Not started

## Objective
Implement both LSTM- and attention-based sequence-to-sequence detectors with early and late fusion modes.

## Context (from paper)
The paper compares LSTM vs MHA and early vs late fusion, with a modular encoder repeated N times.
Paper reference: Section VII (fusion), Section VIII (model), Figure 5, Figure 6.

## Acceptance Criteria
- [ ] Soft quantizer embedding implemented using a learnable quantized basis (Eq. 2 inspired).
- [ ] Encoder module supports LSTM and time-satellite attention variants.
- [ ] Early-fusion model outputs 2-class logits per timestep.
- [ ] Late-fusion model aggregates per-satellite outputs via learned weights.
- [ ] Residual + layernorm + FFN stack implemented.
- [ ] Model shape tests pass on dummy tensors.

## Files to Create
| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_deep_seq2seq_gnss/models/quantizer.py` | soft quantization embedding | VI Eq. (2) | ~120 |
| `src/anima_deep_seq2seq_gnss/models/encoder.py` | LSTM/MHA module stack | VIII, Fig. 5 | ~280 |
| `src/anima_deep_seq2seq_gnss/models/detector.py` | early/late fusion detectors | VII, VIII | ~260 |
| `tests/test_models.py` | forward and shape tests | VIII | ~120 |

## Test Plan
- `pytest tests/test_models.py -v`
- smoke forward pass for both LSTM and MHA, early and late fusion.

## References
- Section VII: early vs late fusion.
- Section VIII: two-value output with softmax + CE, module design.
