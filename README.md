# Verif-AI

**Face-based synthetic-media detection with explainable forensic reports** — upload an image or a video, get a per-face REAL / FAKE / INCONCLUSIVE result, a Grad-CAM evidence view and a downloadable PDF report.

Verif-AI is a two-stream EfficientNet-B0 classifier (RGB stream + fixed SRM-style noise-residual stream) wrapped in a FastAPI backend and a React frontend. Everything runs locally; no external AI service or API key is used.

> **What it is, and is not.** It analyses *faces*. Production model v2 was trained and evaluated on a sample of DeepFakeFace; the current default Model 4 adds Celeb-DF video frames (research-use data). It is **not** a general "is this AI-generated?" detector, its scores are **not calibrated probabilities**, and a REAL result is **not proof of authenticity**. Read [Results](#results-and-limitations) before relying on it.

![Result page](docs/screenshots/desktop-03-result.png)

## Features
- **Image analysis** — multi-face detection (MTCNN), per-face scores, annotated evidence image, Grad-CAM heat-map on the RGB stream.
- **Video analysis** — up to 64 frames sampled across the video (~2 fps, not just frame 0), face tracking across frames, per-track aggregation, suspicious-segment timeline, representative evidence frames.
- **Forensic report** — PDF with verdict, scores, method, model card (checkpoint SHA-256, threshold) and limitations.
- **Honest outputs** — an INCONCLUSIVE band (|score − 0.5| < 0.15) instead of forced verdicts; "no face found" is a result, not an error; uploads are deleted right after analysis; derived results expire after 72 h.
- **API** — documented REST endpoints (`/api/health`, `/api/model`, `/api/analyses`, …) with typed errors and upload validation.
- **Reproducible training** — identity-disjoint splits, duplicate removal, leakage checks, Kaggle GPU workflow, validation-only model selection.

## Architecture (as implemented)
```
image / video ─► decode ─► MTCNN face crop (224, margin 20) ─► JPEG q95 round-trip ─► ImageNet normalise
                                                     ├─► EfficientNet-B0 (RGB) ───────────────┐
                                                     └─► fixed 3-kernel SRM layer ─► EfficientNet-B0 (noise) ─┤
                                                                         concat 2560 ─► Linear 512 ─► ReLU ─► Dropout .5 ─► Linear 1 ─► sigmoid (fake=1)
```
11,972,954 parameters; decision threshold fixed at 0.5. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Current default model (temporary)
The server currently loads **Model 4** (`backend/models/verifai_v3_celebdf_qa.pth`: DeepFakeFace + Celeb-DF, with quality augmentation) by default, as an owner decision while more data is tested. It handles genuine camera video far better than v2 but is **weaker at detecting some AI-generated images**; the measured trade-offs and the rollback (`VERIFAI_MODEL_PATH=models/verifai_v2_best.pth`) are in [docs/CURRENT_DEFAULT_MODEL.md](docs/CURRENT_DEFAULT_MODEL.md). The sections below describe the production model v2.

## Live deployment
| Part | Where | Notes |
|---|---|---|
| Frontend | https://verfai.vercel.app (Vercel, static build of `frontend/`) | hash routes, so refresh and deep links work; the API origin is the build-time variable `VITE_API_BASE_URL` |
| Backend API | https://verifai-api-11wg.onrender.com (Render, Docker, **free tier**, branch `release/live`) | health `/api/health`, docs `/docs`; CORS allows only the Vercel origin |
The free Render instance has 512 MB RAM, so it runs in **lite mode**: the repository's production model v2 (Model 4 is not in Git), **Grad-CAM heat-maps switched off** (the UI says so instead of showing nothing), small batches, short uploads, no persistent disk (results disappear on restart) and a cold start of about a minute after idle. Full functionality (Model 4, heat-maps, durable results) needs a plan with at least 2 GB RAM and a disk, or running the app locally; the measured numbers, setup steps and checklist are in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Do not upload personal media to the public demo.

## Research status
Experiments E1-E4 (quality-shortcut diagnosis, stage-0 gate, modern-generator training) are documented in `docs/` (`E1_QUALITY_BIN_DIAGNOSTIC.md`, `STAGE0_GATE_VALIDATION.md`, `FOUR_MODEL_COMPARISON.md`, `E3_PROTOCOL.md`, `E3_RESULTS.md`, `E4_PROTOCOL_ADDENDUM.md`) with the dataset review in [docs/DATASET_LICENCE_REVIEW.md](docs/DATASET_LICENCE_REVIEW.md). Short version: no model has met the pre-registered promotion rule; the E3 models detect held-out modern fakes (not generator-disjoint) but not better than Model 4 on DeepFakeFace; nothing supports a claim of general AI-image or AI-video detection.

## Results and limitations
Selected model: **`verifai_v2_best.pth`** — a two-stream network initialised from ImageNet and trained on the DeepFakeFace sample on a Kaggle Tesla T4. Chosen among three trained configurations **by validation ROC-AUC only**; the test splits were never used for selection. Threshold 0.5. Frame-level metrics on identity-disjoint splits (95 % bootstrap CIs in the full report):

| Split | What it measures | n | Accuracy | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|
| validation | used to select the model | 471 | 0.877 | 0.948 | 0.167 | 0.078 |
| **internal test** | unseen identities, seen generators | 575 | **0.875** | **0.941** | 0.140 | 0.110 |
| **cross-generator test** | unseen identities **and an unseen generator** (text2img) | 568 | **0.746** | **0.849** | 0.146 | 0.359 |

The original prototype checkpoint (`two_stream_model_epoch_10.pth`), measured on the same independent data, was at **chance level** (accuracy ≈ 0.50, ROC-AUC 0.43–0.50), so the original "99.2 % on Celeb-DF" claim could not be reproduced and has been removed. Full tables, per-run comparison, plots and caveats: [docs/MODEL_EVALUATION_REPORT.md](docs/MODEL_EVALUATION_REPORT.md).

Known limitations (not exhaustive):
- Trained on ~3.9 k face crops from one dataset (IMDB-WIKI real photos; inpainting/insight/text2img fakes). Behaviour on face-swap deepfakes, other generators, heavy compression or other populations is **unmeasured**.
- Cross-generator FNR is ~36 %: unseen generators are missed often.
- **No labelled video dataset was available** (Celeb-DF, FF++, DFDC require signed agreements), so **video-level accuracy has not been measured** and the video aggregation rule is unvalidated. In the end-to-end test a slideshow of four *real* photographs was scored FAKE at video level, consistent with a ~14–17 % per-frame false-positive rate compounding across tracks.
- A 6-second zoom clip made from one real photograph (resampled and re-encoded) was also scored FAKE, while the original still is REAL: expect false positives on re-compressed or resampled media.
- The Grad-CAM view explains the RGB stream only and is not proof that pixels were altered.

## Quick start
Requirements: Python 3.10+, Node 20+ (frontend), ~2 GB disk. CPU is sufficient (~0.5 s per image).

```bash
# backend
cd backend
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                  # defaults are fine; no secrets needed
VERIFAI_FRONTEND_DIR=../frontend/dist uvicorn verifai.api.app:app --port 8000

# frontend (build once, served by the backend at http://localhost:8000)
cd ../frontend && npm ci && npm run build
# or, for development with hot reload:  npm run dev   (proxy to :8000)
```
Docker (`docker build -t verifai . && docker run -p 8000:8000 -v verifai-data:/app/storage verifai`; build and runtime verified in GitHub Codespaces): see [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md). Usage: [docs/USER_GUIDE.md](docs/USER_GUIDE.md). API: [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md).

## Tests
```bash
cd backend && python -m pytest tests -q          # model, data pipeline, scoring/tracking, image & video pipelines, API, PDF, training smoke
cd frontend && npm test && npx tsc --noEmit && npm run build
python backend/tools/e2e_smoke.py --manifest backend/manifests/dff_faces_manifest.csv   # real server + real weights
```
Latest recorded counts are in [docs/PROJECT_COMPLETION_REPORT.md](docs/PROJECT_COMPLETION_REPORT.md).

## Reproduce the training (Kaggle GPU)
Training was done on a free Kaggle GPU with the Kaggle CLI; see [kaggle/README.md](kaggle/README.md) and [docs/DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md). Local pipeline: `backend/training/` (`fetch_deepfakeface`, `prepare_faces`, `train`, `evaluate`).

## Repository layout
```
backend/verifai/     application core (model, preprocessing, faces, scoring, image/video pipelines, Grad-CAM, PDF report, FastAPI app)
backend/training/    dataset preparation, validation, training, evaluation, video tooling
backend/tests/       pytest suite            backend/tools/e2e_smoke.py   end-to-end smoke test
backend/models/      verifai_v2_best.pth (+ model card)      backend/scripts/  original prototype (checkpoint, notebook, plots)
frontend/            React + TypeScript + Tailwind + Vite (Vitest unit tests, Playwright E2E, vercel.json)
render.yaml          Render blueprint for the API (Dockerfile at the repository root)
kaggle/              Kaggle bundle builder and instructions
docs/                architecture, evaluation, dataset, API, user/installation guides, results, figures, screenshots, research history
```

## Acknowledgements
The original two-stream prototype (notebook, training checkpoints and plots in `backend/scripts/` and `backend/plots/`) was written by **Lakshya Singh** ([GitHub](https://github.com/lukkydiwan)). This repository turns it into a tested, documented application and re-trains/evaluates the model; the original code and checkpoint are preserved. Dataset: DeepFakeFace (openrail licence). References: Fridrich & Kodovsky (SRM, 2012); Tan & Le (EfficientNet, 2019); Zhou et al. (two-stream tampered-face detection, 2017).

## Author
Aniket Raj — [GitHub](https://github.com/iamankoo)
