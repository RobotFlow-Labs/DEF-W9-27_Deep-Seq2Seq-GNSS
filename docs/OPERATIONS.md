# Operations Runbook

## Health
- API health endpoint: `GET /health`
- Model inference endpoint: `POST /predict`

## Monitoring Signals
- spoof probability distribution drift
- false alarm rate on known-clean windows
- missed detection rate on synthetic canary attacks

## Failure Modes
- Missing or all-zero satellite windows
- abrupt sensor outages
- invalid tensor shapes from upstream pipeline

## Safeguards
- Presence mask required for every input window.
- Masked aggregation avoids exploding weights for missing satellites.
- Preprocessing zeros out invalid second-difference regions.

## Recovery
- Roll back to last known-good checkpoint.
- Re-run quick smoke with `configs/debug.toml`.
- Recompute evaluation report on latest test split.
