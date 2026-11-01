# FINAL BASELINE AUDIT (recorded BEFORE any improvement work)

Date: 2026-11-01 · Scope: the repository exactly as handed over after "Phase 1 stabilisation" (CHANGELOG `[Phase 1]`).
Method: every file was read; claims in earlier reports were re-checked against code and files; the Phase-1 tests and a
new independent baseline evaluation were executed **before** any change below this audit. Status tags: **[V]** verified · **[IU]** implemented/unverified · **[P]** partial · **[N]** absent.

## 1. Documents read
`Readme.md`, `CHANGELOG.md`, all 8 files in `docs/project-analysis/` (master context, technical report, architecture, ML/datasets, API, research, viva, reference-media report), `backend/scripts/{preprocessing.ipynb, verifai_inference.py, plot.py, pipeline.py}`, `backend/tests/*` (Phase 1), `backend/requirements*.txt`, all checkpoints. **`Phases.md` does not exist** (it will be created at the end).

## 2. Existing architecture [V]
Two-stream late-fusion CNN: RGB stream = EfficientNet-B0 (ImageNet init), noise stream = fixed 3-kernel zero-sum 3x3 high-pass layer (`SRMLayer`) → EfficientNet-B0; 2×1280 pooled features → Linear 512 → ReLU → Dropout .5 → Linear 1 → Sigmoid (real=0, fake=1; decision `p>0.5`). Face crops: MTCNN (224×224, margin 20). Input normalisation: ImageNet mean/std.
Verified in a clean environment: the new package model and the Phase-1 module produce **bit-identical outputs** on the original weights (max |Δ| = 0.0).

## 3. Existing model and checkpoints [V]
| File | Bytes | sha256 (first 16) |
|---|---:|---|
| `backend/scripts/two_stream_model_epoch_10.pth` (the one used by all demos) | 48,169,403 | `9d5a7da92831c7fd` (full: `9d5a7da92831c7fdf16f9b796f8c91c1065b83f5dcfc937aedd965c2ee31dbd2`) |
| `backend/models/two_stream_model_epoch_{1..9}.pth` | 48,168,672 each | `33e93ac8…`, `7d5232b7…`, `d5fc3e24…`, `8190d449…`, `e08a7013…`, `6cb3dff4…`, `20d3a62c…`, `ca608089…`, `c7f76751…` |
11,972,954 stored floats; strict load OK; the SRM kernel is identical in all ten files; weights are genuine successive snapshots (reported earlier, re-confirmed). Hashes are unchanged since Phase 1 (md5 of epoch 10 `6fbbe0f45607c1235db127909ccec065`).

## 4. Datasets: structure and availability
| Item | Finding |
|---|---|
| Celeb-DF videos used by the prototype | **Not available** (not in the repo, not on this machine; access is by request form) |
| `processed/` training crops, `raw_videos/`, `new_test_data/` | **Absent** (git-ignored paths, never present here) |
| Code that created the training crops | Only the extraction cell in the notebook; `processed/` itself cannot be re-created |
| Other local datasets | None found (searched Desktop/Downloads/Documents) |
| Licensing situation | FaceForensics++, Celeb-DF and DFDC require signed access agreements; many Hugging Face "mirrors" re-upload them without permission and were **not** used |
| Dataset acquired for this work | **DeepFakeFace** (Song et al., arXiv:2309.02218; Hugging Face `OpenRL/DeepFakeFace`; licence tag *openrail*, ungated): sampled subset only (≈4,100 images), real = IMDB-WIKI photographs, fakes = Stable-Diffusion inpainting / InsightFace face-swap / SD-v1.5 text-to-image. See `docs/DATASET_DOCUMENTATION.md` |

## 5. Training pipeline (prototype) [V from notebook]
Cell 3: `random_split` 80/20 **over frames** (leaky), batch 4; cell 4: `BCELoss` on sigmoid outputs, Adam 1e-4, 10 epochs, no augmentation, no validation inside the loop, every epoch saved, last epoch used; no scheduler/early stopping/class weighting; hard-coded `D:\` paths; notebook ran out of order on two machines. Training class ratio unknown.

## 6. Preprocessing pipeline [V]
Training crops were written by `cv2.imwrite` (**JPEG q95**) from MTCNN crops (verified bit-exact in Phase 1). Phase 1 made inference reproduce this. No augmentation existed.

## 7. Inference pipeline [V]
`backend/scripts/verifai_inference.py`: device-aware loading, structured result, **single frame only** (frame 0 of whatever path), no multi-face, no image pipeline, no tracking/aggregation, no API/UI.

## 8. Existing test coverage
Phase-1 tests (`test_checkpoint_loading.py`, `test_inference.py`, `test_preprocessing.py`): checkpoint loading, preprocessing parity, structured result, no-face handling. Executed now, before any change:
| Run | Result |
|---|---|
| without private media | **27 passed, 5 skipped** |
| with the two private reference files (local only) | **32 passed** |
Not covered: any pipeline beyond frame 0, API, reports, data tooling, frontend.

## 9. Baseline evaluation (reproducible) — independent, identity-disjoint, never seen by the original model
Command: `python -m training.evaluate --checkpoint scripts/two_stream_model_epoch_10.pth --manifest manifests/dff_faces_manifest.csv --split val:cross_dataset_reference --split test_internal:cross_dataset --split test_crossgen:cross_dataset_unseen_generator --out runs/baseline_original` (threshold fixed at 0.5; metrics in `docs/results/baseline_original_metrics.json`, plots in `docs/figures/baseline_original/`).

| Split (evaluation type) | n (real/fake) | Accuracy [95% CI, identity bootstrap] | ROC-AUC | PR-AUC | Precision / Recall / F1 (fake) | FPR | FNR | Confusion (tn,fp,fn,tp) |
|---|---|---|---|---|---|---|---|---|
| val (cross-dataset, reference only) | 471 (239/232) | 0.503 [0.491, 0.516] | 0.427 | 0.455 | 0.00 / 0.00 / 0.00 | 0.8 % | **100 %** | 237, 2, 232, 0 |
| test_internal (cross-dataset) | 575 (292/283) | 0.511 [0.502, 0.521] | 0.443 | 0.469 | 1.00 / 0.007 / 0.014 | 0.0 % | **99.3 %** | 292, 0, 281, 2 |
| test_crossgen (cross-dataset, unseen text2img) | 568 (281/287) | 0.507 [0.493, 0.522] | 0.499 | 0.518 | 0.89 / 0.028 / 0.054 | 0.4 % | **97.2 %** | 280, 1, 279, 8 |

**Reading:** the original checkpoint is at **chance level** on this data (mean score real 0.07 vs fake 0.06). It answers "REAL" to almost everything. This is the first independent evidence that its self-reported 98.7 %/99.2 % does **not** carry over beyond its own data; it cannot be used to detect AI-generated faces. It also means the earlier reference-media scores of ≈0.002–0.006 on private media carry no information about authenticity.
Self-reported numbers from the notebook (kept for history, **not** independent): train accuracy 0.9921 (epoch 10), "validation" 98.70 % (frame-level random split), 1,000-frame sample 99.20 % (probable overlap with training videos).

## 9a. Shortcut diagnostics of the new data (computed before training)
AUC of raw file size as a real/fake predictor ≈ 0.25–0.29 (fakes are smaller files); after re-encoding the face crop at q95 it is 0.32–0.35 in seen-generator splits and **0.62 in the unseen text2img split (direction reverses)**. So compression size is a partial, generator-dependent cue; it is reported, monitored and mitigated by q95 re-encoding + compression/blur augmentation, never assumed absent.

## 10. Existing defects (carried over, status at baseline)
| # | Defect | Status at baseline |
|---|---|---|
| 1 | Original model does not generalise (chance on DeepFakeFace) | **new finding** |
| 2 | Frame-level split leakage; training frames not reproducible | open (prototype) |
| 3 | No image pipeline; video = frame 0 only | open |
| 4 | No face tracking, multi-face, aggregation, temporal stats | open |
| 5 | No API, UI, DB-free persistence, reports | open |
| 6 | `BCELoss`+Sigmoid, no augmentation, no per-epoch validation, no early stopping | open |
| 7 | `plot.py` hard-coded values, wrong final loss | open (history) |
| 8 | Notebook paths hard-coded, extraction counter bug | open (history) |
| 9 | README describes another system (B4, 30 kernels, TF, per-video) | open |
| 10 | Empty `pipeline.py`, stub `backend/venv`, checkpoints with unused 1000-class heads (2.56 M dead values) | open |
| 11 | Sensitive file exists at `backend/venv/etc/aau_token` (conda stub). **Not read or printed**; excluded by `.gitignore` (`backend/venv/`, `*aau_token*`) | handled |
| 12 | `image6.3.png` (README) shows a person's face; excluded from the repository | handled |

## 11. Dependencies
Before (Phase 1): torch, torchvision, efficientnet-pytorch==0.7.1, facenet-pytorch, opencv-python, numpy, Pillow, tqdm, matplotlib, scikit-learn, seaborn, grad-cam (unused: none). After: split into runtime / train / dev files (see CHANGELOG). Clean-venv installation of the runtime+dev set succeeded (`pip check`: no broken requirements); resolved versions are recorded in `backend/requirements.lock.txt`.

## 12. Hardware
AMD Ryzen 5 5500U (6C/12T), 7.3 GB RAM, integrated Radeon graphics, **no CUDA** (`torch.cuda.is_available() == False`). Everything in this project was executed on CPU; **no GPU training or GPU inference was performed or tested.** Docker Desktop is installed but its daemon was not running.

## 13. Git status and repository configuration
- The project folder was **not a git repository** when received (no history). A new repository was initialised on branch `main`; remote `origin` = `https://github.com/iamankoo/Verif-AI.git` (public, **empty**, no default branch yet); local author `Aniket Raj` with the e-mail already configured globally for the account; GitHub CLI authenticated as `iamankoo` (no new credentials created).
- All files untracked at audit time; no previous commits exist locally or remotely, so there is no published history to preserve or rewrite.
- Checkpoint sizes (48.2 MB each, ≈460 MB total) are below GitHub's 100 MB hard limit (and 50 MB warning) per file; policy chosen: commit the original epoch-10 file and the new best file as plain blobs, **not** epochs 1–9 (documented; kept locally).

## 14. Documentation inaccuracies found
README: TensorFlow, B4, 30 SRM kernels, softmax, per-video output, four notebooks, face alignment, "99.2 % on CelebDF-v2 … strong generalization", "73–90 % AUC" comparison, Zhou et al. year, project path, author block pointing to another repository. `image6.3.png` reference. `docs/project-analysis/*` is historically accurate except one statement corrected in Phase 1 (JPEG quality). Provenance note: the README credits **Lakshya Singh** (github `lukkydiwan/two-stream-deepfake-analyzer`) as author of the prototype; that acknowledgement is preserved in the rewritten README rather than deleted.

## 15. Baseline conclusion
The inherited project is a research notebook plus a single-frame inference module whose model has **no demonstrated ability outside its original dataset**. Everything required for a usable application (data pipeline, leakage-safe training, image/video pipelines, forensics, API, UI, tests, documentation) had to be built, and a new model had to be trained to obtain any detection capability for AI-generated faces.
