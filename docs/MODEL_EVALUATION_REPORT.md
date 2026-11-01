# Model Evaluation Report

Every number below comes from a file in `docs/results/` produced by a command that was actually run. **Validation** numbers were used to choose the model; **test** numbers were never used for any decision. All metrics are **frame/face-level** on one dataset; see [Limitations](#7-limitations-and-what-is-not-claimed) before drawing conclusions.

## 1. Data
DeepFakeFace sample (openrail licence), identity-disjoint and pair-disjoint splits after duplicate removal (details: [DATASET_DOCUMENTATION.md](DATASET_DOCUMENTATION.md)).

| Split | Role | n (real / fake) | Fake generators |
|---|---|---|---|
| train | fitting | 2,315 (1,170 / 1,145) | inpainting, insight |
| val | model selection | 471 (239 / 232) | inpainting, insight |
| test_internal | unseen identities, seen generators | 575 (292 / 283) | inpainting, insight |
| test_crossgen | unseen identities **and unseen generator** | 568 (281 / 287) | text2img |

Real images are IMDB-WIKI photographs. No video dataset was available, so nothing here is video-level.

## 2. Training (Kaggle GPU)
- Notebook `iamankoo/verif-ai-gpu-training` (private), dataset `iamankoo/verifai-bundle` (private); **Tesla T4**, torch 2.11.0+cu128, Python 3.13.
- Same seed (42) and splits for all runs; AdamW lr 1e-4, batch 32, BCE-with-logits (pos_weight 1.02), ReduceLROnPlateau, grad-clip 5, ≤ 15 epochs, early-stopping patience 4 on validation ROC-AUC. 7–9 minutes per run.
- GPU unit tests on Kaggle: 40 passed (includes the CUDA inference test). Inference benchmark of the selected model: 273.8 images/s at batch 32, 117 ms/batch, 589 MB peak GPU memory (`results/gpu_benchmark.json`).
- Artifacts downloaded with `kaggle kernels output`; the zip was verified (60 entries, 0 hash mismatches against the extracted files).

| Run | Initialisation | Augmentation | Best epoch / epochs run | **Validation ROC-AUC** |
|---|---|---|---|---:|
| A | original prototype checkpoint | on | 11 / 15 | 0.9235 |
| **B (selected)** | **ImageNet** | on | 9 / 13 | **0.9477** |
| C | original prototype checkpoint | off | 15 / 15 | 0.9284 |

**Selection rule: highest validation ROC-AUC.** B wins on that rule alone (the test columns happen to agree, but were not consulted). Selected file: `backend/models/verifai_v2_best.pth`, SHA-256 `3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7`, identical to the Kaggle output.

## 3. Results — all three runs (threshold 0.5; 95 % grouped-bootstrap CIs in parentheses)
Confusion cells are TN / FP / FN / TP with fake = positive.

**Validation (selection set — optimistic by construction)**

| Run | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.851 (0.822–0.882) | 0.849 | 0.849 | 0.849 | 0.924 (0.903–0.945) | 0.920 | 204/35/35/197 | 0.146 | 0.151 |
| **B** | 0.877 (0.847–0.905) | 0.843 | 0.922 | 0.881 | 0.948 (0.928–0.965) | 0.944 | 199/40/18/214 | 0.167 | 0.078 |
| C | 0.860 (0.830–0.887) | 0.852 | 0.866 | 0.859 | 0.928 (0.906–0.948) | 0.930 | 204/35/31/201 | 0.146 | 0.134 |

**Independent test — unseen identities, seen generators**

| Run | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.837 (0.807–0.867) | 0.844 | 0.820 | 0.832 | 0.915 (0.891–0.937) | 0.909 | 249/43/51/232 | 0.147 | 0.180 |
| **B** | **0.875 (0.848–0.902)** | 0.860 | 0.890 | 0.875 | **0.941 (0.921–0.960)** | 0.942 | 251/41/31/252 | 0.140 | 0.110 |
| C | 0.854 (0.825–0.881) | 0.864 | 0.834 | 0.849 | 0.926 (0.904–0.947) | 0.923 | 255/37/47/236 | 0.127 | 0.166 |

**Independent test — cross-generator (unseen generator text2img)**

| Run | Acc | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.715 (0.683–0.748) | 0.857 | 0.523 | 0.649 | 0.844 (0.812–0.874) | 0.826 | 256/25/137/150 | 0.089 | 0.477 |
| **B** | **0.746 (0.716–0.778)** | 0.818 | 0.641 | 0.719 | **0.849 (0.818–0.879)** | 0.841 | 240/41/103/184 | 0.146 | 0.359 |
| C | 0.724 (0.691–0.758) | 0.846 | 0.554 | 0.669 | 0.848 (0.817–0.878) | 0.820 | 252/29/128/159 | 0.103 | 0.446 |

Per generator, selected model B (`metrics.json`): real photographs accepted as REAL 85.9 % (internal) / 85.4 % (cross-gen); inpainting detected 91.5 %, insight 86.6 %, **unseen text2img 64.1 %**.
Abstention (INCONCLUSIVE band ±0.15, which is a *reporting* rule, not a threshold change): internal test answers 96.0 % of faces with 88.2 % accuracy on those; cross-generator answers 92.3 % with 75.6 %.
Plots: [`figures/v2_selected/`](figures/v2_selected) (ROC, precision–recall, confusion matrix for each split). Example montages contain real people's faces and are not published.

## 4. Original prototype checkpoint vs the selected model (same data, same code)

| Split | Original: Acc / ROC-AUC / FNR | **Selected: Acc / ROC-AUC / FNR** |
|---|---|---|
| validation | 0.503 / 0.427 / 1.000 | 0.877 / 0.948 / 0.078 |
| internal test | 0.511 / 0.443 / 0.993 | **0.875 / 0.941 / 0.110** |
| cross-generator test | 0.507 / 0.499 / 0.972 | **0.746 / 0.849 / 0.359** |

The original checkpoint labels almost everything REAL on this data (FPR ≈ 0, FNR 97–100 %), i.e. it carries no usable signal here. Its README claim ("99.2 % on Celeb-DF-v2") **could not be reproduced and is not supported by anything in the repository**: Celeb-DF was not available, the code that produced the training data is not in the repository, and the checkpoint is at chance on independent data. The improvement above is on a *different* dataset/task (DeepFakeFace) and must not be read as "the model improved on Celeb-DF".

## 5. Reproduction on this machine (CPU)
The selected checkpoint was re-evaluated locally on CPU with `python -m training.evaluate` (`results/v2_B_imagenetInit_metrics_local_cpu.json`). Accuracy, ROC-AUC, FPR and FNR matched the Kaggle GPU run to every printed digit on all three splits (val 0.8769/0.9477, internal 0.8748/0.9410, cross-gen 0.7465/0.8492).

## 6. Video and end-to-end checks
- `backend/tools/e2e_smoke.py` against a real server with the final checkpoint: **28 / 28 checks passed** (`results/e2e_smoke_report.json`): image analysis for a known real image (REAL, fake-score 0.197), a seen-generator fake (FAKE, 0.9999) and an unseen-generator fake (FAKE, 0.9998); evidence images, Grad-CAM, PDF reports; 415/422 rejections; no-face → INCONCLUSIVE; two video jobs, each analysing 24 sampled frames (not frame 0 only); delete → 404. **These single-image outcomes are smoke-test examples, not accuracy measurements.**
- The two "videos" are slideshows assembled from dataset photographs, so they exercise decode → sample → track → aggregate → report. The real-photo slideshow was classified **FAKE (0.896)** at video level: its per-frame scores ranged 0.27–0.99. That is a false positive and is consistent with the ~14–17 % per-frame FPR compounding when the video score is the maximum over several tracks. A second, independent observation in the Docker run: a 6-second clip made by slowly zooming into one **real** public-domain photograph (resampled with `warpAffine`, re-encoded as MPEG-4) was scored **FAKE (0.85)** at video level, whereas the original still image scores REAL (fake-score ≈ 0.00002). The model is evidently sensitive to resampling/compression artefacts, which real videos always contain, so false positives on genuine video are a realistic risk. Video-level accuracy is therefore **unknown** and the aggregation rule is **unvalidated**; it must not be tuned on this slideshow or on test data.

## 7. Limitations and what is not claimed
1. One dataset, ~3.9 k crops: the CIs above are sampling intervals on that data, not a guarantee for other data.
2. Cross-generator FNR 36 % (64 % detection of the unseen generator): new generators will often be missed. Only one unseen generator was available.
3. No face-swap/deepfake video benchmark: nothing is claimed about FaceForensics++, Celeb-DF, DFDC or real-world deepfake videos.
4. Not a general AI-image detector: it only classifies *faces* and says nothing about non-face content, scenes, text or audio.
5. Real photographs come from one source (IMDB-WIKI) and fakes from one family of pipelines; the classifier may partly learn dataset-specific cues. A file-size shortcut diagnostic was computed (see DATASET_DOCUMENTATION.md) but cannot exclude every shortcut.
6. Scores are uncalibrated sigmoid outputs; do not read 0.9 as "90 % likely fake".
7. The 0.5 threshold was fixed in advance and never tuned on test data.
8. Validation data guided checkpoint and run selection, so validation numbers are optimistic; use the test rows.
