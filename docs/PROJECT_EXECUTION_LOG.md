# Project Execution Log

Chronological record of the delivery work (dates are the machine's local date, 2026-11-01). Commands are abbreviated; every number below was produced by a command that was actually run. Phases follow the assignment; a phase is marked **complete** only after the verification listed.

---
## Phase 1 - Audit and baseline - COMPLETE
**Tasks:** read all docs/code/notebook/checkpoints; inventory hardware, tools, git, data; run the existing tests; run an independent baseline evaluation (after acquiring an evaluation dataset, see Phase 2).
**Commands:** `git ls-remote`, `gh repo view` (remote is public and empty), `pytest tests/test_checkpoint_loading.py tests/test_inference.py tests/test_preprocessing.py` (27 passed, 5 skipped; 32 passed with private media), `python -m training.evaluate --checkpoint scripts/two_stream_model_epoch_10.pth …`.
**Result:** original checkpoint at chance level on independent data (accuracy 0.503-0.511, ROC-AUC 0.43-0.50, FNR 97-100 %). Report: `docs/project-analysis/FINAL_BASELINE_AUDIT.md`, metrics `docs/results/baseline_original_metrics.json`.
**Findings recorded:** no Celeb-DF data locally; no CUDA; Docker daemon off; `backend/venv/etc/aau_token` exists (never read; git-ignored); `Phases.md` absent; project folder was not a git repo.

## Phase 2 - Dataset and data pipeline - COMPLETE
**Decision:** restricted datasets (FF++, Celeb-DF, DFDC) are not obtainable without signed agreements; unofficial mirrors were rejected. The openly licensed DeepFakeFace (openrail) was used.
**Files:** `backend/training/{remote_zip.py, fetch_deepfakeface.py, prepare_faces.py, data_validation.py, dataset.py, video_tools.py}`, `backend/manifests/*`.
**Commands:** `python -m training.fetch_deepfakeface --seed 42` (4,100 files, only sampled members downloaded), `python -m training.prepare_faces --name dff`.
**Verification:** identity violations 0, pair violations 0, 8 cross-split duplicate images removed, 3,929 images retained; shortcut diagnostics computed; unit tests in `tests/test_data_pipeline.py`, `tests/test_video_tools.py`.
**Limitation:** no video benchmark available; video tooling untested on real video datasets.

## Phase 3 - Training pipeline and model improvement - COMPLETE (trained on Kaggle GPU)
**Files:** `backend/training/{train.py, evaluate.py, metrics.py}`; `TwoStreamNet.logits()` added (forward unchanged, verified bit-identical to the Phase-1 module).
**Local CPU attempt (stopped by decision of the owner):** `python -m training.train --init scripts/two_stream_model_epoch_10.pth --name v2_finetune …` ran ≈ 15 min (80 of 144 steps of epoch 1, ≈ 11 s/step ⇒ ≈ 28 min/epoch ⇒ several hours total). Loss fell 3.03 → 0.61 within epoch 1 (training loss only). It was stopped before finishing an epoch; **no model came out of it and it is not used anywhere**.
**Bug found by the smoke test and fixed:** `train.py` stored `torch.__version__` (a `TorchVersion`) in checkpoint metadata, which made `torch.load(weights_only=True)` reject the file. Fixed (`str(...)`) and covered by `tests/test_training_smoke.py` (tiny synthetic manifest, full epoch, reload through the strict loader).
**Decision:** train on Kaggle GPU via the Kaggle CLI. Package: `kaggle/build_kaggle_bundle.py` → private dataset (code, manifests, the exact local crops, original checkpoint; 116 MB) + GPU notebook comparing 3 configurations with validation-only selection. Self-containment verified locally from inside the bundle (39 tests passed, 1 CUDA skip; train smoke and evaluation OK). Instructions: `kaggle/README.md`.
**Blocker (resolved):** no Kaggle credentials existed on this machine. The owner signed in to Kaggle in Chrome and created a legacy API key (downloaded `kaggle.json`); it was moved to `~/.kaggle/kaggle.json` with restricted permissions **without being opened or printed**. `kaggle datasets list --mine` then authenticated as `iamankoo`. The account showed phone verification and 30 GPU hours.

### Phase 3 execution on Kaggle (2026-11-01)
1. `python kaggle/build_kaggle_bundle.py --username iamankoo` → 116 MB, 3,973 files.
2. `kaggle datasets create -p kaggle/dist/dataset --dir-mode zip` → private dataset `iamankoo/verifai-bundle` (106 MB upload); `datasets status` → `ready`.
3. `kaggle kernels push -p kaggle/dist/kernel` → **version 1 failed** at the "locate dataset" cell (`AssertionError: Attach the dataset…`): the GPU and pip setup cells had passed, but Kaggle unpacked `backend.zip` at the dataset root so the package was at `verifai/` rather than `backend/verifai/`. Fix: the notebook now globs `**/verifai/__init__.py` and prints the input tree on failure; the kernel id was aligned with the slug Kaggle derived from the title (`iamankoo/verif-ai-gpu-training`).
4. **Version 2** ran on a Tesla T4 (torch 2.11.0+cu128) in ≈ 32 minutes (≈ 13:29–14:02 local) → `KernelWorkerStatus.COMPLETE`. GPU test log: 40 passed. Benchmark: 273.8 img/s, 589 MB peak.
5. `kaggle kernels output` → `verifai_results.zip` (145.6 MB, SHA-256 `0942b2f5…`); 60 zip entries compared with the extracted files: 0 mismatches.
6. Selection by validation ROC-AUC only: A 0.9235, **B 0.9477**, C 0.9284 → **B (ImageNet init)**.
7. Local CPU re-evaluation of B (`training.evaluate`, 2 min 42 s): identical to the GPU numbers. Installed as `backend/models/verifai_v2_best.pth` (SHA-256 `3b947755…`, identical to the Kaggle file) with a model card; the original checkpoint is unchanged.
8. Local verification on the final model: backend `pytest` **108 passed, 6 skipped** (5 need private media, 1 needs CUDA); frontend `tsc` clean, **36 passed**, build OK; Kaggle bundle tests **39 passed, 1 skipped (CUDA)**; `tools/e2e_smoke.py` **28/28**.
9. Screenshots captured from the running app (headless Chrome driving the real UI on the real backend) with the public-domain astronaut photograph → `docs/screenshots/`.

## Phases 4-6 - Image pipeline, video pipeline, forensics - COMPLETE (code + tests)
**Files:** `backend/verifai/{image_pipeline, video_pipeline, scoring, faces, explain, report, preprocess}.py`.
**Verification:** `tests/test_pipelines.py`, `tests/test_report.py`, `tests/test_model_package.py`, `tests/test_scoring_and_tracking.py` (scripted-face logic tests + random-weight models; plus real-weight equivalence with the Phase-1 module).

## Phase 7 - Backend - COMPLETE (code + tests)
**Files:** `backend/verifai/{api/app.py, engine.py, store.py, schemas.py, config.py, errors.py}`; `tests/test_api.py` (health, model, upload validation, 413/415/422/404/409/503, path-traversal, job lifecycle, PDF, delete, CORS, degraded mode).

## Phase 8 - Frontend - COMPLETE (code + tests)
**Files:** `frontend/` (React 18, TypeScript, Tailwind, Vite, Vitest). `npx tsc --noEmit` clean; `npm test` **36 passed**; `npm run build` OK (169 kB JS / 54 kB gzip).

## Phase 9-15 - see the entries appended below as they complete.

---
## Clean-environment installation (Phase 14 evidence)
`python -m venv backend/.venv && pip install -r requirements-dev.txt` → exit 0; `pip check`: *No broken requirements found*; resolved versions in `backend/requirements.lock.txt` (torch 2.14.1, torchvision 0.29.2, facenet-pytorch 2.5.3, grad-cam 1.5.7, fastapi 0.162.0, numpy 2.2.6, opencv-python-headless 5.0.0.93, reportlab 5.1.0).
Backend test suite in that environment: **100 passed, 6 skipped** (5 need private media, 1 needs CUDA) at the first complete run; later runs are listed below.
