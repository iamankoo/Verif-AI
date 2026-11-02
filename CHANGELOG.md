# Changelog

All notable changes to Verif-AI. Dates are ISO. Phase numbering follows the approved plan.

## [2.1.0] — 2026-11-02 — E4 results and model registry

### Added
- **Experiment E4 completed and recorded** (`docs/E4_RESULTS.md`, `docs/results/e4_modern/`): not promoted - fails frozen criterion 2 (DeepFakeFace missed-fake rate 0.208 / 0.509 against limits 0.160 / 0.409). Model 4 stays the default; v2 the rollback.
- **Model registry** (`backend/verifai/registry.py`): production default / rollback / candidate roles, `GET /api/models`, optional `model` form field on the analysis endpoints, lazy loading of at most one optional model, `VERIFAI_ROLLBACK_MODEL_PATH`, `VERIFAI_CANDIDATE_MODEL_PATH`, `VERIFAI_ENABLE_MODEL_SELECTION`.
- Dashboard: model picker (only where the server offers choices), candidate warning, and a result chip naming the model and its role; the PDF report prints the model role.
- E4 candidate model card `backend/models/verifai_e4_candidate.model.json` (the checkpoint itself is never committed).
- Tests: backend `test_model_registry.py` (12) and `test_store_atomic.py` (1); frontend model-selection tests (3) and a Playwright model-selection test.

### Fixed
- Job/result files are now written atomically and read with a short retry: a browser poll of `/status` could hit a half-written `job.json` while the video worker saved progress (HTTP 500, found during the multi-model smoke test).

### Changed
- `Dockerfile` and `render.yaml` set `VERIFAI_ENABLE_MODEL_SELECTION=false`: hosted instances offer only the model they serve.

## [2.0.0] — 2026-11-01 — Application build, retraining, evaluation

### Added
- **Application** (`backend/verifai/`): training-matched preprocessing, MTCNN face pipeline, scoring with an INCONCLUSIVE band, image pipeline (multi-face, Grad-CAM, annotated evidence), video pipeline (≤ 64 sampled frames, IoU tracking, per-track aggregation, suspicious segments), PDF forensic report (ReportLab), FastAPI service with a filesystem result store, typed errors and upload validation.
- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind + Vite — upload, progress, result view, video timeline, model card, history; wired only to real endpoints.
- **Training/data tooling** (`backend/training/`): remote-zip sampling of DeepFakeFace, face preparation, leakage/duplicate/shortcut validation, training (BCE-with-logits, early stopping on validation ROC-AUC), evaluation (metrics, bootstrap CIs, plots, abstention), video tools.
- **Kaggle workflow** (`kaggle/`): bundle builder + GPU notebook comparing three configurations; instructions.
- **Model `verifai_v2_best.pth`** (ImageNet-initialised, trained on Kaggle Tesla T4, selected by validation ROC-AUC) with model card `backend/models/verifai_v2_best.model.json`.
- Tests: backend 108 passed / 6 skipped; frontend 36 passed; Kaggle bundle 39 passed / 1 skipped; real-server smoke test `backend/tools/e2e_smoke.py` 28/28.
- Docs: README, ARCHITECTURE, MODEL_EVALUATION_REPORT, DATASET_DOCUMENTATION, API_DOCUMENTATION, USER_GUIDE, INSTALLATION_GUIDE, PROJECT_EXECUTION_LOG, PROJECT_COMPLETION_REPORT, Phases, genuine screenshots in `docs/screenshots/`.
- `Dockerfile` (single image: built UI + FastAPI, CPU).

### Changed
- `torch.load(..., weights_only=True)` everywhere; checkpoints store only plain-Python metadata (a `TorchVersion` object in metadata once broke loading; fixed and covered by `test_training_smoke.py`).
- README rewritten: the old claims (99.2 % on Celeb-DF-v2, EfficientNet-B4, 30 SRM kernels, TensorFlow) did not match the code or the evidence and were removed.

### Removed
- Empty `backend/scripts/pipeline.py`; `deepfake_banner.svg` (repeated the unsupported claims); `image6.3.png` is not published. Intermediate epoch 1–9 checkpoints stay local only (git-ignored).

### Measured (see docs/MODEL_EVALUATION_REPORT.md)
Original checkpoint on independent data: chance level. Selected model: internal test accuracy 0.875 / ROC-AUC 0.941; unseen-generator test accuracy 0.746 / ROC-AUC 0.849. Video-level accuracy was **not** measured (no labelled video data).

## [Phase 1 — Stabilization] — 2026-11-01

Scope approved by the owner: (1) device-aware checkpoint loading, (2) correct `requirements.txt`, (3) fix the training/inference preprocessing mismatch, (4) structured inference results, (5) basic tests. **Not changed:** model architecture, checkpoints, thresholds, training; **not added:** image or video pipelines.

### Verification performed before changing preprocessing (item 3)
Exact training preprocessing was established from source and by execution, not assumed:
- Notebook cell 0 passes **numpy RGB frames** to `MTCNN(image_size=224, margin=20, post_process=True)(frame, save_path=...)`.
- facenet-pytorch 2.5.3 `extract_face` crops with `cv2.resize(..., INTER_AREA)` and `save_img` writes numpy crops with **`cv2.imwrite(path, BGR)` → OpenCV default JPEG quality 95**; the saved crop is taken *before* `post_process` standardisation.
- Notebook cell 3 reads files with `PIL.Image.open().convert('RGB')` → `Resize((224,224))` (no-op) → `ToTensor` → `Normalize(ImageNet)`.
- Executed check on a real face: file written by the training procedure **==** `cv2.imencode(".jpg", BGR, q95)` + PIL decode, **bit-identical**; PIL's default (quality 75) does **not** match. Max pixel change introduced by the JPEG step on that crop: 26 intensity levels.
- Old inference (`face/255 → interpolate → Normalize`, raw crop with **no JPEG step**) therefore differed from training.
- *Assumption that cannot be verified:* the code that produced the training folder `processed/` is not in the repository; it is assumed to be the cell-0 function (the only extraction code present).
- Correction to my earlier reference-media report: its "after JPEG" columns used a PIL q75 emulation, which was wrong (see that report's correction note).

### Added
| File | Purpose |
|---|---|
| `backend/scripts/verifai_inference.py` | Single source of truth for model + single-frame inference (moved from notebook cell 8). Contains: `SRMLayer`, `TwoStreamNet` (identical architecture/attribute names), `resolve_device`, `load_checkpoint`, `initialize_engine`, `training_jpeg_roundtrip`, `preprocess_face`, `predict_face`, `analyze_rgb_frame` (internal building block), `run_forensic_analysis`. |
| `backend/tests/conftest.py`, `test_checkpoint_loading.py`, `test_inference.py`, `test_preprocessing.py` | 32 tests (27 run without media; 5 more with `VERIFAI_TEST_IMAGE` / `VERIFAI_TEST_VIDEO`). |
| `backend/requirements-dev.txt` | `-r requirements.txt` + `pytest`. |
| `CHANGELOG.md` | this file. |
| `reference_tests/` | Harness scripts, logs, backups of the original files, isolated `site/` with `grad-cam` (kept outside project config). Not part of the product. |

### Changed
| File | Change |
|---|---|
| `backend/scripts/preprocessing.ipynb` | **Only cell 8 changed** (verified: cells 0–7, 9, 10 and metadata are equal to the original). Cell 8 no longer duplicates the model/engine/pipeline code; it imports `SRMLayer, TwoStreamNet, initialize_engine, run_forensic_analysis` from `verifai_inference` and keeps its original "EXECUTION" block (same `MY_WEIGHTS`, same hard-coded `D:\…\id1_0008.mp4` path); the final call is now `result = run_forensic_analysis(...)` followed by `result`. Cell 8's stored outputs (an old CUDA→CPU error) were cleared because they no longer match the code. The original notebook is preserved at `reference_tests/backups/preprocessing.ipynb.orig`. |
| `backend/requirements.txt` | Replaced `pytorch-gradcam` with `grad-cam` (the package that provides `pytorch_grad_cam`); added missing `seaborn`; added `Pillow`; removed unused `torchaudio`, `albumentations`, `pandas`; pinned `efficientnet-pytorch==0.7.1`; added `>=` floors for the rest; documented the versions actually tested. Original: `reference_tests/backups/requirements.txt.orig`. |
| `backend/.gitignore` | Added `__pycache__`, `.pytest_cache`, `*.pyc` (existing entries untouched). Original in `backups/gitignore.orig`. |
| `docs/project-analysis/ML_MODELS_AND_DATASETS.md` | Corrected JPEG-quality statement (now: `cv2.imwrite` default 95, verified). |
| `docs/project-analysis/REFERENCE_MEDIA_TEST_REPORT.md` | Added correction note about the q75 emulation. |

### Behaviour changes (item by item)
1. **Checkpoint loading (item 1).** `torch.load(path, map_location=<device>, weights_only=True)`; `device=None/"auto"` → CUDA if available else CPU; explicit `"cuda"` without CUDA falls back to CPU with a warning; if the installed torch does not know `weights_only` (TypeError) it falls back to a plain load with a warning; `load_state_dict(strict=True)` so incompatible checkpoints raise; `module.` (DataParallel) prefixes tolerated. Fixes the recorded `RuntimeError: Attempting to deserialize object on a CUDA device…`.
   - `initialize_engine(model_path, device=None)` is signature-compatible with the old `initialize_engine(model_path=None)`.
   - **Safety change:** a `model_path` that does not exist now raises `FileNotFoundError` (the old code silently continued with an untrained head).
   - When a checkpoint is given the two backbones are built with `EfficientNet.from_name` (no ImageNet download). Output equivalence with `from_pretrained` + checkpoint is tested (max difference ≤ 1e-7). `TwoStreamNet()` with no argument still behaves like the notebook (`pretrained=True`).
2. **Preprocessing (item 3).** `preprocess_face(..., match_training_jpeg=True)` now applies the training-time JPEG round-trip (OpenCV, BGR, quality 95, PIL decode) before `/255` + ImageNet normalisation. `match_training_jpeg=False` reproduces the **old** inference path exactly (tested bit-for-bit; and the reference scores reproduce: 0.001649 / 0.002334). Default is the matched path. *Consequence:* probabilities change slightly (below). Thresholds and labels logic are untouched.
3. **Structured output (item 4).** `run_forensic_analysis` returns a dict instead of `None`:
   `status` (`ok` | `no_face` | `read_error`), `label` (`DEEPFAKE`/`REAL`/`None`), `fake_probability`, `real_probability` (= 1 − fake), `face_box` (`[x1,y1,x2,y2]` in frame pixels of the selected largest face), `face_confidence` (MTCNN), `threshold` (0.5), `message`, `frame_index`, `preprocessing`, `device`, and `visualization_error` if the explanation failed. Failures that used to print and return `None` now also set `status`/`message`; the original console messages are still printed (`verbose=True`).
   - Still analyses **only the first frame** (unchanged; no video pipeline added). The same function still accepts a still-image path via OpenCV's fallback (unchanged, still not a supported image pipeline).
   - The box/confidence come from a single `mtcnn.detect → select_boxes → extract` sequence that was verified tensor-identical to the old `mtcnn(frame)` call.
   - New optional arguments: `visualize=True`, `verbose=True`, `match_training_jpeg=True`, `save_figure_path=None`.
4. **Grad-CAM.** Same target layer and target as before; now lazily imported, wrapped so an explanation failure cannot break the prediction; the figure shows the pixels actually fed to the model (after the preprocessing fix); no more global `torch.set_grad_enabled(True)` side effect.
5. **Removed dead code:** the "logit vs probability" branch (unreachable because the model ends in a sigmoid).
6. **Console safety:** prints no longer crash on cp1252 consoles (emoji replaced by plain text / ASCII fallback).

### Unchanged (verified)
Model architecture and attribute names (strict load of all keys; SRM kernel values; 11,972,954 stored floats), checkpoints (md5 `6fbbe0f45607c1235db127909ccec065` for epoch 10; epoch 1 `555760b9…`), decision threshold 0.5, labels, MTCNN settings (`image_size=224, margin=20, post_process=False` at inference), ImageNet normalisation, `plot.py`, `pipeline.py` (still empty), `Readme.md`, training cells 0–7.

### Test results
- `python -m pytest backend/tests` → **27 passed, 5 skipped** (media tests skipped without media).
- With `VERIFAI_TEST_IMAGE`/`VERIFAI_TEST_VIDEO` set → **32 passed**.
- Mutation check: with JPEG quality changed to 75 and `weights_only=False` the suite reported **3 failures** (so the tests detect those regressions).

### Reference-media re-check (CPU, same two files; ground truth still unknown)
| Input | Old behaviour (pre-Phase-1) | Phase 1 default (training-matched) | Label |
|---|---|---|---|
| Image (via video function) | `initialize_engine` crashed on CPU as written; with workaround p(fake)=0.001649 | **0.005767** (real 0.994233) | REAL → REAL |
| Video frame 0 | 0.002334 | **0.004132** (real 0.995868), box [603.7, 240.3, 840.2, 544.6] | REAL → REAL |
| 10 sampled frames (harness) | max 0.0023, 0/10 > 0.5 | max 0.0041, 0/10 > 0.5 | unchanged |
Patched notebook cell 8 executes end-to-end on CPU (`Loaded weights from … onto cpu`), returns the structured dict and produces the Grad-CAM figure. Logs: `reference_tests/logs/phase1_run_log.txt`, `reference_tests/logs/phase1/`.

### Known remaining issues (not addressed — outside Phase 1 scope)
1. No image pipeline; no video pipeline (frame 0 only; no sampling, aggregation, tracking).
2. Notebook cell 9 (`deep_load_and_test`) is unchanged: still `strict=False` and references an undefined `model` when run alone.
3. Notebook cells 0, 3, 5, 6 still contain hard-coded `D:\…` / `../` paths; cell 0 config drift (`LIMIT=50`, `CATEGORIES=['Celeb-real']`); `saved` counter bug (increments without a detected face).
4. Training-side issues untouched: frame-level split leakage, `BCELoss`+Sigmoid, no augmentation, no validation per epoch, `plot.py` hard-coded values with wrong final loss.
5. The training preprocessing equivalence assumes `processed/` was built with the cell-0 code (not provable from the repo).
6. Scores remain uncalibrated and saturated; ground truth of the reference media is unknown; no known-fake test exists, so **detection ability on these media is still unmeasured**.
7. README still describes a different architecture (B4, 30 kernels, TensorFlow, per-video output).
8. `requirements.txt` lower bounds were not individually tested; `facenet-pytorch` is kept at `>=2.5.3`; no lock file.
9. Running a GPU path was not possible on this machine (no CUDA); the CUDA branch of `resolve_device`/`load_checkpoint` is covered only by logic/unit tests with `map_location`, not by a real GPU run.
10. Checkpoints (≈460 MB) remain in the repository tree; `*.pth` is not git-ignored; there is no git repository in this folder.
11. The stub `backend/venv` directory and the empty `pipeline.py` remain.
