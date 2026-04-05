# Deep-Seq2Seq-GNSS Task Index — 18 Tasks

## Build Order

| Task | Title | Depends | Status |
|------|-------|---------|--------|
| PRD-0101 | Package + config scaffold | None | ⬜ |
| PRD-0102 | Synthetic GNSS trajectory generator | PRD-0101 | ⬜ |
| PRD-0103 | Spoofing attack injection (targeted/regional) | PRD-0102 | ⬜ |
| PRD-0104 | Preprocessing (2nd diff + sign-log + indicators) | PRD-0103 | ⬜ |
| PRD-0201 | Soft quantizer embedding | PRD-0104 | ⬜ |
| PRD-0202 | LSTM encoder module stack | PRD-0201 | ⬜ |
| PRD-0203 | Time-satellite attention encoder stack | PRD-0201 | ⬜ |
| PRD-0204 | Early-fusion detector head | PRD-0202, PRD-0203 | ⬜ |
| PRD-0205 | Late-fusion detector head + weighting | PRD-0202, PRD-0203 | ⬜ |
| PRD-0206 | Model tests and shape validation | PRD-0204, PRD-0205 | ⬜ |
| PRD-0301 | Training loop and checkpoint IO | PRD-0206 | ⬜ |
| PRD-0302 | Inference CLI and confidence outputs | PRD-0301 | ⬜ |
| PRD-0401 | Metric library (err/fa/md) | PRD-0302 | ⬜ |
| PRD-0402 | Evaluation runner with attack-type splits | PRD-0401 | ⬜ |
| PRD-0501 | FastAPI health + predict service | PRD-0302 | ⬜ |
| PRD-0502 | Docker serving assets | PRD-0501 | ⬜ |
| PRD-0701 | CUDA migration and operations docs | PRD-0402 | ⬜ |
| PRD-0702 | ONNX export utility | PRD-0301 | ⬜ |
