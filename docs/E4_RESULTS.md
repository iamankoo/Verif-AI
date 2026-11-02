# Experiment E4 - results (completed, verified)

**Status: completed. E4 did NOT pass the production-promotion rule. By owner decision (2026-11-02) it is the default model of the local examiner demonstration only; Model 4 is an alternative, v2 stays the rollback and the model served by the live website. E4 is not deployed live.**
Protocol: [E3_PROTOCOL.md](E3_PROTOCOL.md) + [E4_PROTOCOL_ADDENDUM.md](E4_PROTOCOL_ADDENDUM.md) (frozen before training). Nothing below changed the protocol, a threshold or a dataset.

## What E4 is
The E3A recipe (ImageNet initialisation, quality augmentation 0.5, AdamW 1e-4, batch 32, seed 42, at most 12 epochs, early-stopping patience 4) with every **DeepFakeFace training row repeated four times** (validation and test splits unchanged), to test whether the small share of DeepFakeFace in the E3 training set (about 5% of the fakes) explains the loss of DeepFakeFace detection. Train set: 45,571 rows with repeats (20,565 real, 25,006 fake); validation 4,455 rows.

## Run record (Kaggle notebook `iamankoo/verif-ai-e4`, Tesla T4)
| Item | Value |
|---|---|
| Final status | COMPLETE (notebook exit code 0, no failure message) |
| Epochs run / best epoch | 12 of 12 (the epoch cap) / epoch 12 (training had not plateaued) |
| Validation selection score (mean of clean and degraded validation ROC-AUC) | 0.9797 |
| Validation accuracy / ROC-AUC (epoch 12, **validation - not test**) | 0.954 / 0.990 |
| Training wall time | 134.5 minutes (whole notebook about 4 h 30 m, incl. Celeb-DF/modern crop extraction and evaluation) |
| Checkpoint | `runs/e4a_dffx4_modern_qa0.5/best.pth`, 48,119,236 bytes, SHA-256 `0d553de918c120994e0919386d7afb6378cd489cc9ef4abc2aa4850e5cf9cba0` |
| Results archive | `verifai_e4_results.zip`, 44,934,864 bytes, SHA-256 `f9dd4f6c45ef4e385ca3ed2c0cc4f07a358fc5968e3a53a887d31170502599d4` (checkpoint, metrics and manifest only: no images or crops) |

Machine-readable records: `docs/results/e4_modern/` (`e4_selection.json`, `e4_data.json`, `e4_history.csv`, `e4_summary.json`, `e4_config.json`).

## Held-out test results (clean inputs, fixed threshold 0.5; accuracy / ROC-AUC / FPR / FNR)
| Test set | E4 | Model 4 (default) | v2 (rollback) |
|---|---|---|---|
| DeepFakeFace internal (575 crops: 292 real / 283 fake) | 0.831 / 0.908 / 0.130 / 0.208 | 0.835 / 0.921 / 0.130 / 0.201 | 0.875 / 0.941 / 0.140 / 0.110 |
| DeepFakeFace cross-generator (568 crops: 281 / 287; unseen generator) | 0.688 / 0.799 / 0.110 / 0.509 | 0.627 / 0.795 / 0.107 / 0.634 | 0.746 / 0.849 / 0.146 / 0.359 |
| Celeb-DF test, frames | 0.945 / 0.985 / 0.061 / 0.045 | 0.953 / 0.988 / 0.038 / 0.061 | 0.471 / 0.479 / 0.593 / 0.425 |
| Celeb-DF test, **video** (147 videos: 91 / 56) | 0.980 / 0.999 / 0.022 / 0.018 | 0.980 / 0.998 / 0.011 / 0.036 | 0.476 / 0.479 / 0.604 / 0.393 |
| `modern_test` (4,326 crops) | 0.983 / 0.998 / 0.023 / 0.014 | 0.328 / 0.414 / 0.082 / 0.969 | 0.326 / 0.398 / 0.090 / 0.968 |

* **FNR = missed-fake rate** (fake called REAL); **FPR** = real called FAKE.
* `modern_test` is **not generator-disjoint**: its fakes come from the same generator families as the training fakes and all its real faces are FFHQ, so a source shortcut cannot be excluded. Celeb-DF results are within-dataset evidence for one face-swap method (9 held-out identities).
* E4 is a single run (one seed).

## The frozen promotion rule (reporting only - no promotion was made)
| # | Criterion | E4 value | Result |
|---|---|---|---|
| 1 | Celeb-DF video FPR <= Model 4's + 0.03 and accuracy >= 0.95 | FPR 0.022 vs 0.011; accuracy 0.980 | PASS |
| 2 | DeepFakeFace internal: FNR <= v2's + 0.05 (limit 0.160) and FPR <= 0.20 | FNR **0.208**; FPR 0.130 | **FAIL** |
| 2 | DeepFakeFace cross-generator: FNR <= v2's + 0.05 (limit 0.409) and FPR <= 0.20 | FNR **0.509**; FPR 0.110 | **FAIL** |
| 3 | `modern_test` balanced accuracy >= Model 4's + 0.10, interval above 0 | 0.982 vs 0.475 (difference +0.507, lower bound +0.498) | PASS |

**Decision:** criterion 2 fails (internal 0.208 > 0.160, cross-generator 0.509 > 0.409), so E4 is not eligible. Repeating the DeepFakeFace rows recovered part of the unseen-generator detection (FNR 0.509 against 0.634 for Model 4 and 0.624 for E3A) and slightly improved the internal test (0.208 against 0.230 for E3A), but not enough.

## Independent local reproduction
The E4 and Model 4 checkpoints were re-evaluated on this machine (CPU) with `python -m training.evaluate` on the local DeepFakeFace test crops (threshold 0.5, nothing tuned):

| | E4 internal | E4 cross-gen | Model 4 internal | Model 4 cross-gen |
|---|---|---|---|---|
| Local accuracy / ROC-AUC / FPR / FNR | 0.8313 / 0.9083 / 0.1301 / 0.2085 | 0.6884 / 0.7986 / 0.1103 / 0.5087 | 0.8348 / 0.9207 / 0.1301 / 0.2014 | 0.6268 / 0.7954 / 0.1068 / 0.6341 |
| Kaggle record | 0.831 / 0.908 / 0.130 / 0.208 | 0.688 / 0.799 / 0.110 / 0.509 | 0.835 / 0.921 / 0.130 / 0.201 | 0.627 / 0.795 / 0.107 / 0.634 |

They agree to the third decimal (`docs/results/e4_modern/local_reproduction_*.json`). The Celeb-DF and `modern_test` numbers cannot be reproduced locally because those images are not stored here.

## Caution observed in the live-style smoke test (not an evaluation)
On the public NASA test image E4 and Model 4 both return REAL. On the 7.6 s clip made by zooming into that **real** photograph, Model 4 returns REAL (0.18) while E4 (0.97) and v2 (0.79) return FAKE - consistent with the known sensitivity of these detectors to resampled/re-encoded media. One clip is an anecdote, not a measurement.
