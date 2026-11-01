# Current default model: Model 4 (temporary)

**Status:** since 2026-11-01 the application loads **Model 4** by default. This is an owner decision made while more data is tested; it is **not** a claim that Model 4 is the better detector. The model will be reconsidered after those tests. Production **v2** is kept unchanged as the rollback.

| | Model 4 (current default) | v2 (rollback) |
|---|---|---|
| File | `backend/models/verifai_v3_celebdf_qa.pth` | `backend/models/verifai_v2_best.pth` |
| SHA-256 | `e7e5f3dc28b252d43e2e9134e7f41185c49526cff5d1c03367d14ae3cf07eafb` | `3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7` |
| Training data | DeepFakeFace (DFF) + Celeb-DF v2 crops (about 2.3k DFF, about 16.3k Celeb-DF; 41 training identities), with quality augmentation | DFF only |
| In the Git repository | **No** (copied locally; excluded from Git) | yes |

Both checkpoints are unmodified; nothing in this change retrained, fine-tuned or re-thresholded either of them. The decision threshold is still 0.5 and the INCONCLUSIVE band is still +/-0.15.

## How the default is set
* `backend/.env` (local, git-ignored): `VERIFAI_MODEL_PATH=models/verifai_v3_celebdf_qa.pth`. `backend/.env.example` documents the same value but leaves it commented out, so a fresh clone without the Model 4 file falls back to v2 instead of failing at startup.
* `backend/verifai/config.py`: if `VERIFAI_MODEL_PATH` is unset the default order is Model 4, then v2, then the original prototype. If the Model 4 file is not present, the server falls back to v2 (check `/api/model` to see which file is loaded). A path set explicitly in `.env` that does not exist is an error at startup.
* `Dockerfile` still points at v2 because Model 4 is not in the repository and therefore not in the image.

## Switching back to v2
Set `VERIFAI_MODEL_PATH=models/verifai_v2_best.pth` in `backend/.env` (or in the environment) and restart. Verify with `GET /api/model`: `checkpoint_file` and `checkpoint_sha256` must match the table above.

## Measured trade-offs (same pipeline, same threshold 0.5)
Evaluated with the application's own pipeline (face detection, crop, JPEG round-trip, normalisation, model, maximum over faces) on a fixed, seeded, stratified sample (120 images per group) of the DeepFakeFace **test** splits. Neither model was trained on or selected with these test images. Images where no usable face was detected are excluded (counts shown). FPR/FNR are at the fixed 0.5 threshold with Wilson 95% intervals; they describe this sample only.

| Group (label) | n scored (no face) | v2 FNR / FPR | Model 4 FNR / FPR |
|---|---|---|---|
| Real, internal test (real) | 116 (4) | FPR 0.121 [0.073, 0.192] | FPR 0.138 [0.087, 0.212] |
| Inpainting = AI-edited faces (fake) | 117 (3) | FNR 0.085 [0.047, 0.150] | FNR 0.205 [0.142, 0.287] |
| Insight = face-swap (fake) | 112 (8) | FNR 0.116 [0.069, 0.189] | FNR 0.152 [0.097, 0.230] |
| Real, cross-generator test (real) | 113 (7) | FPR 0.115 [0.068, 0.187] | FPR 0.115 [0.068, 0.187] |
| **Text-to-image faces, unseen generator (fake)** | 119 (1) | FNR 0.361 [0.281, 0.451] | **FNR 0.563 [0.473, 0.649]** |

Confusion matrices (TN / FP / FN / TP, threshold 0.5):

| Split | v2 | Model 4 |
|---|---|---|
| DFF internal test sample (116 real, 229 fake) | 102 / 14 / 23 / 206; accuracy 0.893; ROC-AUC 0.949 | 100 / 16 / 41 / 188; accuracy 0.835; ROC-AUC 0.922 |
| DFF cross-generator sample (113 real, 119 fake) | 100 / 13 / 43 / 76; accuracy 0.759; ROC-AUC 0.860 | 100 / 13 / 67 / 52; accuracy 0.655; ROC-AUC 0.787 |

Earlier, on held-out Celeb-DF video identities (`docs/FOUR_MODEL_COMPARISON.md`): v2 is at chance (video accuracy 0.483), Model 4 reaches 0.980. That figure is within-dataset (same face-swap method and footage type as its training data) and must not be read as general video reliability.

## What this means
* **Model 4 is better on genuine camera-style video and on Celeb-DF-like face swaps; it is worse than v2 at detecting AI-edited and text-to-image faces.** On unseen text-to-image faces it lets 56% of fakes through at threshold 0.5 (v2: 36%).
* **Neither model reliably detects modern AI-generated photographs.** Both are trained on DeepFakeFace (about 2023-era generators) and Celeb-DF. A clean, high-resolution image from a recent generator is outside both training distributions; a REAL verdict on such an image is a model limitation, not an implementation fault (see below).
* Model 4's SRM (noise-residual) stream scores unseen text-to-image fakes very low (mean 0.11 versus 0.82 for v2) and its RGB stream is weaker too (0.43 versus 0.60), so the fused score is lower (stream scores measured by neutralising the other stream with mean DFF-train features, on the same text-to-image sample; they are conditional scores, not independent classifiers). `docs/STAGE0_GATE_VALIDATION.md` documents the related stream-collapse behaviour on Celeb-DF.
* The score is **not a calibrated probability**. A score of 0.000 or 1.000 does not mean certainty; the interface labels it "model score, 0-1" and never shows it as a percentage. A REAL verdict is never proof of authenticity.
* Do not claim a general accuracy figure for the system; only the numbers above, on the stated data, are supported.
