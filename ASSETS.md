# Deep-Seq2Seq-GNSS — Asset Manifest

## Paper
- Title: Deep Sequence-to-Sequence Models for GNSS Spoofing Detection
- ArXiv: 2510.19890
- Authors: Jan Zelinka, Oliver Kost, Marek Hruz
- PDF: `papers/2510.19890.pdf`

## Status: ALMOST

## Pretrained Weights
| Model | Size | Source | Path on Server | Status |
|-------|------|--------|---------------|--------|
| MHA Early Fusion detector | TBD | Not released in paper | /mnt/forge-data/models/gnss/deep-seq2seq-mha-early.pt | MISSING |
| LSTM Early Fusion detector | TBD | Not released in paper | /mnt/forge-data/models/gnss/deep-seq2seq-lstm-early.pt | MISSING |

## Datasets
| Dataset | Size | Split | Source | Path | Status |
|---------|------|-------|--------|------|--------|
| Synthetic GNSS spoofing corpus | 67k train + 1k test sequences | train/test | Generated from paper rules | /mnt/forge-data/datasets/gnss-spoofing/synth-v1 | MISSING |
| Local bootstrap synthetic set | configurable | train/val/test | `src/anima_deep_seq2seq_gnss/data.py` | `./artifacts/data/` | READY |

## Hyperparameters (from paper)
| Param | Value | Paper Section |
|-------|-------|---------------|
| embedding_dim | 128 | IX |
| ff_hidden_dim | 1024 | IX |
| attention_heads | 8 | IX |
| encoder_modules | 1..8 sweep | IX + Fig. 6 |
| loss | cross-entropy on 2-class logits | VIII |

## Expected Metrics (from paper)
| Benchmark | Metric | Paper Value | Our Target |
|-----------|--------|-------------|-----------|
| Total (MHA Early Fusion) | classification error | 0.16% | <= 1.00% on local synthetic bootstrap |
| Targeted attacks (MHA Early) | classification error | 0.31% | <= 2.00% on local synthetic bootstrap |
| Regional attacks (MHA Early) | classification error | 0.03% | <= 1.00% on local synthetic bootstrap |

## Notes
- Paper does not publish an official repository URL.
- This module includes a fully local synthetic generator and train/eval stack so CUDA migration can be done later without redesign.
