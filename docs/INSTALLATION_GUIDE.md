# Installation Guide

Tested on Windows 11, Python 3.10, CPU only. Linux/macOS use the same commands with the usual path/activation differences. **GPU/CUDA was not available for testing**; the code selects CUDA automatically when present but that path is only covered by unit logic, not by a real GPU run.

## 1. Requirements
* Python 3.10 (3.9-3.12 expected to work; 3.10 is what was tested)
* Node.js 20+ and npm (only to build/run the frontend)
* ~3 GB free disk for the Python environment (PyTorch), 1 GB RAM minimum for inference (≈600 MB measured for the model + detector), more for long videos
* Git. (Docker optional.)

## 2. Get the code
```bash
git clone https://github.com/iamankoo/Verif-AI.git
cd Verif-AI
```
The repository contains two checkpoints (`backend/models/verifai_v2_best.pth` - the trained model, and `backend/scripts/two_stream_model_epoch_10.pth` - the original prototype). No dataset is included.

## 3. Backend
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt   # runtime only
# development / tests / training extras:
pip install -r requirements-dev.txt
cp .env.example .env              # optional: adjust settings (no secrets needed)
```
For a CUDA machine install the matching CUDA build of PyTorch first (see pytorch.org), then `pip install -r requirements.txt`.
Exact versions that were verified together are in `backend/requirements.lock.txt`.

Start the API:
```bash
uvicorn verifai.api.app:app --host 127.0.0.1 --port 8000
# http://127.0.0.1:8000/docs   (Swagger)      http://127.0.0.1:8000/api/health
```
First start takes a few seconds to load the model. No internet access is needed at run time (MTCNN weights ship inside `facenet-pytorch`; the EfficientNet backbones are built without downloading ImageNet weights because the checkpoint supplies all weights).

## 4. Frontend (development)
```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api to http://127.0.0.1:8000)
```
Production build served by the API itself:
```bash
cd frontend && npm run build
cd ../backend && set VERIFAI_FRONTEND_DIR=../frontend/dist && uvicorn verifai.api.app:app --port 8000   # Linux: export VERIFAI_FRONTEND_DIR=../frontend/dist
# UI at http://127.0.0.1:8000/
```

## 5. Docker (CPU)
```bash
docker build -t verifai .
docker run --rm -p 8000:8000 -v verifai-data:/app/storage verifai       # http://localhost:8000
```
The image bundles the built UI and the checkpoints from `backend/models` and `backend/scripts`.

**Verified (2026-11-01, GitHub Codespaces, Docker 29.9.2, 4 vCPU / 16 GB, commit `98e6c82`):** `docker build -t verifai .` succeeded in 4 min 21 s (image 2.44 GB); the container became *healthy* with 0 restarts and runs as the non-root user `verifai`; `/api/health` reported `model_loaded: true` (CPU), `/api/model` reported checkpoint SHA-256 `3b947755…`, `/` served the UI (HTTP 200, also screenshotted through a forwarded port: `docs/screenshots/docker-desktop-0*.png`); a real image analysis, a video job (13 sampled frames, 2 tracks) and both PDF reports worked; a `.py` upload returned 415; `DELETE` returned 204 then 404; no errors in the container log. The Docker run is a functional check on a public-domain photograph and a synthetic video made from it, not an accuracy measurement. Not tested: GPU images, other CPU architectures, a registry push, running behind a reverse proxy.

## 6. Verify the installation
```bash
cd backend && python -m pytest tests -q          # expect: all pass; a few skip (private media / CUDA)
cd ../frontend && npm test && npm run build
python tools/e2e_smoke.py --help                  # real-server smoke test (needs the dataset sample)
```

## 7. Reproducing training (optional)
```bash
cd backend
pip install -r requirements-train.txt
python -m training.fetch_deepfakeface --out data/raw/dff --seed 42
python -m training.prepare_faces --manifest manifests/dff_raw_manifest.csv --name dff
python -m training.train --manifest manifests/dff_faces_manifest.csv --init scripts/two_stream_model_epoch_10.pth --name v2_finetune --epochs 8
python -m training.evaluate --checkpoint runs/v2_finetune/best.pth --manifest manifests/dff_faces_manifest.csv --split test_internal:internal_test --split test_crossgen:cross_generator --out runs/v2_finetune/eval
```
On a GPU machine use the same commands (CUDA is detected automatically); consider `--batch-size 32 --workers 4`. CPU training of the delivered model took several hours (see `docs/MODEL_EVALUATION_REPORT.md`).

## 8. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `/api/health` shows `degraded` | checkpoint missing/corrupt: set `VERIFAI_MODEL_PATH` (file must be a real `.pth`, not a Git-LFS pointer) |
| `Grad-CAM unavailable` warning | install `grad-cam` (not `pytorch-gradcam`) |
| `unreadable_video` | the container/codec is not supported by the installed OpenCV/FFmpeg build; re-encode to H.264 MP4 |
| First request slow | model/detector warm-up |
| CORS error in the browser | add the UI origin to `VERIFAI_CORS_ORIGINS` |
