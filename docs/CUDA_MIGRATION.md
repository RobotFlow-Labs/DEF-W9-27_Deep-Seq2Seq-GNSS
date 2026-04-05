# CUDA Migration Plan

## Goal
Move local training/inference workloads to CUDA server without changing interfaces.

## Steps
1. Clone repository on CUDA host and create environment with matching Python and torch versions.
2. Install package with `pip install -e .`.
3. Start from `configs/paper.toml` and tune:
- batch size
- number of workers
- mixed precision policy
4. Enable deterministic seeds for regression checks.
5. Validate parity:
- Run debug config on CPU and CUDA with same seed.
- Compare output probabilities on fixed samples (tolerance based).
6. Scale training and checkpoint outputs to shared storage.

## Risks
- Different GPU kernels can introduce slight nondeterminism.
- Sequence length x satellite count may exceed VRAM with larger batch sizes.
- Attention memory may dominate when increasing sequence length.

## Mitigations
- Use gradient accumulation.
- Use bf16 or fp16 mixed precision when validated.
- Keep checkpoint format stable (`model_state` dictionary).
