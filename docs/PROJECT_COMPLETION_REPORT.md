# Project Completion Report

Repository: <https://github.com/iamankoo/Verif-AI> (public, branch `main`). Everything stated here was executed on 2026-11-01 unless noted; "not measured" means exactly that.

## 1. What was delivered
| Area | Result |
|---|---|
| Image pipeline | Multi-face detection (MTCNN), training-matched preprocessing, per-face scores, INCONCLUSIVE band, annotated evidence, Grad-CAM (RGB stream) |
| Video pipeline | ≤ 64 sampled frames across the clip (~2 fps; not frame 0 only), IoU tracking, per-track aggregation, suspicious segments, evidence frames, background job with progress |
| Forensic analysis | Per-face/per-track evidence + PDF report (verdict, scores, method, model card with checkpoint SHA-256, limitations) |
| Backend | FastAPI (`/api/health`, `/api/model`, `/api/analyze/image|video`, `/api/analyses…`, report, artifacts, delete), upload validation, typed errors, filesystem store with TTL, uploads deleted after analysis |
| Frontend | React + TypeScript + Tailwind + Vite; upload, progress, results, video timeline, model card, history; uses only real endpoints |
| Training/data | Identity-disjoint leakage-safe splits, duplicate removal, shortcut diagnostics, reproducible training and evaluation, Kaggle GPU workflow |
| Packaging | Single Docker image (built UI + API, CPU), verified in GitHub Codespaces |
| Documentation | README, ARCHITECTURE, MODEL_EVALUATION_REPORT, DATASET_DOCUMENTATION, API_DOCUMENTATION, USER_GUIDE, INSTALLATION_GUIDE, CHANGELOG, Phases, PROJECT_EXECUTION_LOG, this report, genuine screenshots |

## 2. Selected model
`backend/models/verifai_v2_best.pth` — two-stream EfficientNet-B0 (RGB) + fixed SRM-style layer → EfficientNet-B0 (noise), late fusion; 11,972,954 parameters; threshold 0.5. **ImageNet-initialised** (run B), trained on the DeepFakeFace sample, best epoch 9, SHA-256 `3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7` (model card: `backend/models/verifai_v2_best.model.json`). Selected among three runs **by validation ROC-AUC only** (A 0.9235, **B 0.9477**, C 0.9284); the test splits played no part. The original checkpoint `backend/scripts/two_stream_model_epoch_10.pth` (SHA-256 `9d5a7da92831c7fdf16f9b796f8c91c1065b83f5dcfc937aedd965c2ee31dbd2`, unchanged) is preserved; the nine earlier epoch checkpoints are kept locally only.

## 3. Kaggle execution
- Auth: owner-created legacy API key moved to `~/.kaggle/kaggle.json` (never opened/printed); CLI `kaggle 1.7.4.5`, user `iamankoo`.
- Dataset `iamankoo/verifai-bundle` (private, 106 MB upload, status `ready`). Notebook `iamankoo/verif-ai-gpu-training` (private, GPU + internet).
- Version 1 **failed** (dataset path assumption — Kaggle unpacked the zip at the dataset root); fixed; version 2 **completed** on a **Tesla T4** (torch 2.11.0+cu128, CUDA 12.8) in ≈ 32 minutes; three runs of 7–9 minutes each (A 15 epochs, B 13 with early stopping, C 15).
- GPU tests on Kaggle: 40 passed. GPU inference benchmark: 273.8 images/s (batch 32), 117 ms/batch, 589 MB peak.
- Artifacts downloaded with `kaggle kernels output`; `verifai_results.zip` (145.6 MB, SHA-256 `0942b2f58b4d80cf1dcfc87da58de84f818af5177dce31ead4bc9b6f5120e348`), 60 entries, 0 mismatches vs the extracted files; selected `best.pth` identical to `models/verifai_v2_best.pth`.
- Local CPU re-evaluation matched the GPU numbers exactly on all three splits.

## 4. Dataset and splits
DeepFakeFace sample (openrail): real = IMDB-WIKI photographs; fakes = inpainting, insight (seen) and text2img (unseen generator). After de-duplication (8 cross-split duplicate images removed): train 2,315 (1,170 real / 1,145 fake), val 471, internal test 575, cross-generator test 568 — identity-disjoint and pair-disjoint (0 violations). Restricted datasets (Celeb-DF, FaceForensics++, DFDC) were not used (signed agreements required); unofficial mirrors were rejected. Details: `docs/DATASET_DOCUMENTATION.md`.

## 5. Held-out metrics (selected model, threshold 0.5, face-level)
| Split | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| validation (used for selection) | 471 | 0.877 | 0.843 | 0.922 | 0.881 | 0.948 | 0.944 | 199/40/18/214 | 0.167 | 0.078 |
| **internal test** (unseen identities) | 575 | **0.875** | 0.860 | 0.890 | 0.875 | **0.941** | 0.942 | 251/41/31/252 | 0.140 | 0.110 |
| **cross-generator test** (unseen generator) | 568 | **0.746** | 0.818 | 0.641 | 0.719 | **0.849** | 0.841 | 240/41/103/184 | 0.146 | 0.359 |

Original checkpoint on the same data: accuracy 0.503–0.511, ROC-AUC 0.43–0.50, FNR 97–100 % (chance level). Confidence intervals, all three runs, per-generator results and plots: `docs/MODEL_EVALUATION_REPORT.md`.

## 6. Image and video results
- **Image**: known-real test image → REAL (0.197); seen-generator fake → FAKE (0.9999); unseen-generator fake → FAKE (0.9998) — single smoke-test examples, not accuracy. Population-level image accuracy is the table above.
- **Video**: **video-level metrics were not measured** (no labelled video dataset available). Functional results: the pipeline analysed 24 of 24 sampled frames with faces in the e2e slideshows (not only frame 0). A slideshow of four real photographs was scored FAKE (0.896) and a zoom clip made from one real photograph was scored FAKE (0.85) in Docker — false positives that the per-frame FPR (14–17 %) and sensitivity to resampling/re-encoding make plausible. The video aggregation rule is unvalidated.

## 7. Tests (executed on the final tree)
| Suite | Result |
|---|---|
| Backend `pytest tests` | **108 passed, 6 skipped** (5 need the owner's private reference media; 1 needs CUDA) |
| Frontend `tsc --noEmit` / `npm test` / `npm run build` | clean / **36 passed** / OK |
| Kaggle bundle (from inside the bundle) | **39 passed, 1 skipped** (CUDA) |
| Kaggle GPU tests | 40 passed |
| Real-server smoke `backend/tools/e2e_smoke.py` (final model) | **28 / 28** |
Earlier counts from previous sessions (100/6, 36, 39/1) were not relied upon.

## 8. Docker
Verified in GitHub Codespaces (Docker 29.9.2; commit `98e6c82`): build succeeded (4 min 21 s, 2.44 GB), container healthy, 0 restarts, non-root; health, model info, UI, image analysis, video analysis, PDFs, 415 rejection and delete all worked; UI screenshots captured through a forwarded port. Details and what was not covered: `docs/INSTALLATION_GUIDE.md` §5 and `docs/PROJECT_EXECUTION_LOG.md`. The Codespace was deleted afterwards.

## 9. Screenshots
`docs/screenshots/`: `desktop-01-home`, `-02-preview`, `-03-result`, `mobile-01…03`, `docker-desktop-01…03` — real captures of the running application (headless Chrome driving the real UI against the real backend) using a public-domain photograph. A FAKE-verdict screenshot is not included because it would require a face image of a real person from the dataset; the FAKE path is covered by tests and the e2e smoke run. No video screenshot is included (no public-domain video with a face was available); the video UI is covered by component tests and the API video runs above.

## 10. GitHub
Pushed to `origin/main` with the owner's `gh` session, no force, author *Aniket Raj* (existing Git identity), no attribution trailers. Commits: `1471e1b` original prototype · `129c556` backend, training, tests, model · `eb86f29` frontend · `eed9056` Kaggle workflow + Dockerfile · `98e6c82` docs/results/screenshots · plus a final documentation-finalisation commit (see `git log`). Verified with `git ls-remote` and the GitHub API. No secrets, `kaggle.json`, `.env`, private media, virtualenvs, datasets or run directories are tracked. Checkpoints are plain blobs (48 MB each; Git LFS not needed).

## 11. Limitations
1. Face-only; trained/evaluated on one openly licensed dataset (~3.9 k crops): generalisation to other generators, face-swap videos, compression, populations or non-face content is **unmeasured** — it is not a general AI-image detector.
2. Cross-generator FNR 36 %; only one unseen generator was available.
3. No video-level accuracy; video aggregation unvalidated; false positives on resampled/re-encoded real media observed.
4. Scores are uncalibrated; REAL ≠ authentic; Grad-CAM covers the RGB stream only.
5. The original prototype's "99.2 % on Celeb-DF-v2" claim could not be reproduced and was removed; the improvement reported is on a different dataset.
6. Validation numbers are optimistic (used for selection); use the test rows.
7. Docker: CPU image only; GPU image, registry push and reverse-proxy deployment untested. Local Docker was unavailable (Codespaces used).
8. The owner's private reference media and the report about them are deliberately not published.

## 12. Reproduction commands
```bash
# backend + tests
cd backend && python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
python -m pytest tests -q
VERIFAI_FRONTEND_DIR=../frontend/dist uvicorn verifai.api.app:app --port 8000
# frontend
cd frontend && npm ci && npm test && npx tsc --noEmit && npm run build
# Docker
docker build -t verifai . && docker run --rm -p 8000:8000 -v verifai-data:/app/storage verifai
# data + training (Kaggle GPU) — see kaggle/README.md
cd backend && python -m training.fetch_deepfakeface --seed 42 && python -m training.prepare_faces --name dff
python kaggle/build_kaggle_bundle.py --username <you> && kaggle datasets create -p kaggle/dist/dataset --dir-mode zip && kaggle kernels push -p kaggle/dist/kernel
# evaluate a checkpoint
cd backend && python -m training.evaluate --checkpoint models/verifai_v2_best.pth --split val --split test_internal --split test_crossgen --out runs/eval
```
(`fetch_deepfakeface`/`prepare_faces` recreate the data locally; the data itself is not in the repository.)
