# Running Verif-AI — Complete Guide (Windows PowerShell)

This guide explains how to install and start Verif-AI on a fresh laptop, how to start it again later, and how to fix the problems that can actually occur. Every command here was checked against the repository; things that could not be tested are marked **(not tested)**. For a 1-page version see [QUICK_START.md](QUICK_START.md).

## A. Prerequisites

| Item | Requirement | Source / status |
|---|---|---|
| Python | **3.10** (tested). 3.9–3.12 are expected to work but were not tested | `docs/INSTALLATION_GUIDE.md`; the Dockerfile also uses 3.10 |
| Node.js + npm | Node **20 or newer**, npm. Tested with Node 24.14 / npm 11.9; the Docker build uses Node 22 | `docs/INSTALLATION_GUIDE.md` |
| Git | any recent version | to clone the repository |
| Disk | about **3 GB** for the Python environment (PyTorch) plus the repository (~100 MB, mostly the two 48 MB checkpoints) and `node_modules` | `docs/INSTALLATION_GUIDE.md` |
| RAM | about **600 MB** measured for model + face detector; more for long videos; 8 GB laptop is comfortable | `docs/INSTALLATION_GUIDE.md` |
| GPU | **Not required.** CPU is fully supported and is what was tested. If CUDA is present it is used automatically | `backend/verifai/model.py` (`resolve_device`) |
| Internet | needed once to download packages. **Not needed to run**: the face-detector weights ship inside the `facenet-pytorch` package and the checkpoint contains all model weights | `docs/INSTALLATION_GUIDE.md` |
| System libraries | none beyond Python/Node on Windows (OpenCV is installed as `opencv-python-headless`) | `backend/requirements.txt` |
| Environment variables | **none are required**. Optional settings are listed in §E | `backend/.env.example` |

## B. First-time setup

Run these in **one PowerShell window** (the "setup terminal"). Replace the folder you clone into as you like.

### 1–2. Clone and enter the project
```powershell
git clone https://github.com/iamankoo/Verif-AI.git
cd Verif-AI
```
Expected folders: `backend`, `frontend`, `docs`, `kaggle`, plus `Dockerfile`, `README.md`.

### 3–4. Create and activate the Python virtual environment
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
The prompt now starts with `(.venv)`. If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (affects only this window) and activate again. This is standard Windows behaviour and not specific to the project **(not tested here)**.

### 5. Install backend dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
This installs PyTorch, torchvision, `efficientnet-pytorch`, `facenet-pytorch`, OpenCV (headless), Grad-CAM (`grad-cam`), FastAPI, Uvicorn, ReportLab and the other runtime packages. It takes several minutes and downloads a few GB. The exact versions verified together are in `backend/requirements.lock.txt`. Only if you also want to run the tests: `pip install -r requirements-dev.txt`.

Quick check that the install is complete:
```powershell
pip check
```
(expected: `No broken requirements found.`)

### 6. Install and build the frontend
```powershell
cd ..\frontend
npm ci
npm run build
cd ..
```
`npm ci` installs exactly what `package-lock.json` specifies. `npm run build` type-checks and creates `frontend\dist`, which the backend serves.

### 7. Configure environment variables (optional)
Nothing is required. To change defaults, copy the example file and edit it:
```powershell
Copy-Item backend\.env.example backend\.env
```
`backend\.env` is git-ignored. Do not put secrets in it – the project has none.

### 8. Check that the model checkpoint exists
**Current default: Model 4** (`verifai_v3_celebdf_qa.pth`, temporary - see [CURRENT_DEFAULT_MODEL.md](CURRENT_DEFAULT_MODEL.md)). Production v2 is kept unchanged as the rollback.
```powershell
Test-Path backend\models\verifai_v3_celebdf_qa.pth
(Get-FileHash backend\models\verifai_v3_celebdf_qa.pth -Algorithm SHA256).Hash   # starts E7E5F3DC28B2...
Test-Path backend\models\verifai_v2_best.pth
(Get-FileHash backend\models\verifai_v2_best.pth -Algorithm SHA256).Hash
```
Expected: `True` for both; v2's full SHA-256 is `3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7` (case does not matter). Each file is about 48 MB. Model 4 is not stored in the repository; without it the server falls back to v2. To use v2: `VERIFAI_MODEL_PATH=models/verifai_v2_best.pth` in `backend\.env`. If it is missing the server falls back to the original prototype checkpoint `backend\scripts\two_stream_model_epoch_10.pth` (which is **at chance level** – do not demonstrate with it).

### 9–10. Start the application
There are two ways. **Use mode 1 for the examiner demonstration** (one terminal, one URL).

**Mode 1 – single program (backend serves the built UI), recommended**
```powershell
cd backend
.\.venv\Scripts\Activate.ps1
$env:VERIFAI_FRONTEND_DIR = "..\frontend\dist"
python -m uvicorn verifai.api.app:app --host 127.0.0.1 --port 8000
```
Wait for `Application startup complete.` (a few seconds for the model to load). Open **http://127.0.0.1:8000**.

**Mode 2 – development (two terminals, hot reload)**
- Terminal 1 (backend):
  ```powershell
  cd backend
  .\.venv\Scripts\Activate.ps1
  python -m uvicorn verifai.api.app:app --host 127.0.0.1 --port 8000
  ```
- Terminal 2 (frontend):
  ```powershell
  cd frontend
  npm run dev
  ```
  Open **http://localhost:5173**. The Vite dev server forwards every `/api` request to `http://127.0.0.1:8000` (`frontend/vite.config.ts`), so the backend **must** be on port 8000 in this mode.

Both modes were tested on this repository: `/`, `/api/health`, `/docs`, `/openapi.json` and `/api/analyses` answered HTTP 200 in mode 1, and `http://localhost:5173/` and `http://localhost:5173/api/health` answered 200 in mode 2.

## C. Starting again later (no re-installation)
Only the "every time" commands are needed:
```powershell
cd Verif-AI\backend
.\.venv\Scripts\Activate.ps1
$env:VERIFAI_FRONTEND_DIR = "..\frontend\dist"
python -m uvicorn verifai.api.app:app --host 127.0.0.1 --port 8000
```
then open http://127.0.0.1:8000. Which terminal does what:

| Terminal | Command | Needed in |
|---|---|---|
| T1 – backend (keep open) | the four lines above | mode 1 and mode 2 (in mode 2 omit the `$env:` line) |
| T2 – frontend dev server | `cd frontend; npm run dev` | **mode 2 only** |

Stop with `Ctrl+C`. You must re-run the `$env:VERIFAI_FRONTEND_DIR` line in every new PowerShell window (or put `VERIFAI_FRONTEND_DIR=../frontend/dist` into `backend\.env` and run the server from the `backend` folder).

Useful URLs: `http://127.0.0.1:8000/` (UI), `/api/health`, `/api/model`, `/docs` (interactive API page), `/openapi.json`.

## D. Troubleshooting (only problems the code can actually produce)

**Python dependency errors**
- `pip install` fails or is slow: check internet; upgrade pip (`python -m pip install --upgrade pip`); use Python 3.10. Use `pip install -r requirements.txt` from inside `backend`.
- `ModuleNotFoundError: verifai`/`fastapi`: you are not in the activated venv or not in `backend`. Run `.\.venv\Scripts\Activate.ps1` and start from the `backend` folder.
- Grad-CAM missing: the analysis still works and the result shows the warning "Grad-CAM is unavailable in this environment (pytorch_grad_cam not installed)". Fix: `pip install grad-cam` (not `pytorch-gradcam`).
- PDF fails: ReportLab is missing → `pip install reportlab` (it is in `requirements.txt`).

**Node.js dependency errors**
- `npm ci` fails: check Node is 20+ (`node -v`); delete `frontend\node_modules` and run `npm ci` again; or use `npm install`.
- `npm run build` stops with a TypeScript error: it runs `tsc --noEmit` first; the repository builds cleanly, so this usually means a modified file or an old Node.

**Backend does not start**
- Read the last lines in the terminal. If the model fails to load, the server still starts in *degraded mode*: `/api/health` returns `"status":"degraded","model_loaded":false` with the reason in `detail`, the UI shows the message "Cannot reach…" or the analyse button is disabled, and analysis requests return HTTP 503 `model_unavailable`.
- *Model checkpoint not found / cannot load*: confirm §B step 8; if you moved the file set `VERIFAI_MODEL_PATH` (see §E). Re-clone if the hash does not match (a partial download or Git line-ending conversion breaks it).

**Frontend not connecting to backend**
- The UI shows "Cannot reach the Verif-AI server. Is the backend running?" → start/restart the backend and refresh.
- Mode 2 only: the backend must be on port 8000 (hard-coded in `frontend/vite.config.ts`). If you must use another port, use mode 1 instead and open that port.
- If you open the UI from a different address than `localhost:5173`/`127.0.0.1:5173` in mode 2, set `VERIFAI_CORS_ORIGINS` accordingly (default allows only those two). Mode 1 is same-origin and needs no CORS.
- `{"detail":"Not Found"}` at `/`: `VERIFAI_FRONTEND_DIR` was not set in that terminal, or `frontend\dist` does not exist (run `npm run build`).

**CUDA / GPU errors**
- A GPU is optional. Force CPU: `$env:VERIFAI_DEVICE = "cpu"` before starting. If you ask for CUDA when none exists the code warns "CUDA requested but unavailable; using CPU." and continues. A real GPU run was **not tested on this machine** (GPU results come from the Kaggle Tesla T4 run, where CPU and GPU outputs matched).

**CPU inference seems slow**
- Measured on the development laptop (CPU): about **0.4–0.6 s per image**, and about **7 s for a 24-frame video**; the first request after start can be slower while caches warm up. Videos are capped at 64 sampled frames, so run time does not grow with video length beyond that. Only **one video is processed at a time**; extra uploads wait in a queue (`Engine` uses a single worker). Close other heavy programs; do not run two demos at once. To skip the heat-maps (the most expensive extra step) set `$env:VERIFAI_ENABLE_HEATMAPS = "false"` **(not timed)**.

**Port already in use**
- Error `[Errno 10048] … only one usage of each socket address`: another program (often an earlier Verif-AI window) holds the port. Find and stop it:
  ```powershell
  Get-NetTCPConnection -State Listen -LocalPort 8000 | Select-Object LocalPort,OwningProcess
  Stop-Process -Id <OwningProcess>
  ```
  or start on another port with `--port 8001` (mode 1 only, then open `http://127.0.0.1:8001`). This exact error happened during development when a stale server held port 8000.

**Image upload failures** (exact messages)
- Wrong type → HTTP 415 `Unsupported image type '.txt'. Allowed: .bmp, .jpeg, .jpg, .png, .tif, .tiff, .webp.` (the browser also blocks it first: `Unsupported file type ".xyz"…`).
- Too big → the browser: `The file is X MB; the limit for images is 15 MB.` Server: HTTP 413 `File exceeds the 15 MB limit for images.` (videos: 200 MB).
- Not really an image (e.g., text renamed `.jpg`) → HTTP 422 `File is not a readable image (UnidentifiedImageError).`
- Smaller than 64 px on the shortest side → 422 `Image is too small (WxH); minimum side is 64px.`; more than 60 million pixels → 422.
- Empty file → `The uploaded file is empty.`

**Video processing failures**
- Text renamed `.mp4`, corrupt file or unsupported codec → the upload is accepted (HTTP 202) and the job then ends as *failed* with `The file could not be opened as a video (unsupported or corrupt container/codec).`; the UI shows this message in a red banner. Re-save/convert the video to a common MP4 file with a tool of your choice and retry **(conversion not tested)**.
- A video with no usable face finishes with verdict INCONCLUSIVE and an explanation, not an error.
- Videos longer than 900 s: only the first 900 s are analysed (a warning is shown).

**PDF generation failures**
- `409` "not ready" means the analysis has not finished yet – wait for the result page. The PDF is built on request from the stored result; if the analysis was deleted or has expired (72 h) the request returns 404. Re-run the analysis and download again.

## E. Optional environment variables (all optional; defaults in `backend/.env.example`)
`VERIFAI_MODEL_PATH`, `VERIFAI_DEVICE` (`auto|cpu|cuda`), `VERIFAI_STORAGE_DIR` (default `backend\storage`), `VERIFAI_RESULT_TTL_HOURS` (72), `VERIFAI_MAX_IMAGE_MB` (15), `VERIFAI_MAX_VIDEO_MB` (200), `VERIFAI_MAX_IMAGE_PIXELS`, `VERIFAI_UNCERTAINTY_MARGIN` (0.15), `VERIFAI_VIDEO_SAMPLE_FPS` (2.0), `VERIFAI_VIDEO_MAX_FRAMES` (64), `VERIFAI_VIDEO_MAX_DURATION_S` (900), `VERIFAI_VIDEO_AGGREGATION` (median), `VERIFAI_ENABLE_HEATMAPS`, `VERIFAI_CORS_ORIGINS`, `VERIFAI_FRONTEND_DIR`. The 0.5 decision threshold is **not** configurable on purpose. In PowerShell set one for the current window with `$env:NAME = "value"`.

## F. Where files go
- Uploaded files are copied to `backend\storage\tmp` under a random name and **deleted as soon as analysis ends** (also on failure). Your original file on the laptop is never modified.
- Evidence images and the result JSON are kept in `backend\storage\analyses\<analysis_id>\` (images in its `artifacts` subfolder) for 72 hours or until you click **Delete results**. They contain faces – treat the folder as sensitive. `backend\storage` is git-ignored.
- PDFs are not stored: they are generated when you click download and saved by your browser (usually `Downloads\verifai-report-<first 8 characters of the id>.pdf`).

## G. Docker (optional alternative, internet needed for the build)
```powershell
docker build -t verifai .
docker run --rm -p 8000:8000 -v verifai-data:/app/storage verifai
```
Open http://localhost:8000. Verified in GitHub Codespaces (build 4 min 21 s, image 2.44 GB, healthy container). Docker Desktop was **not** available on the development laptop, so a local Windows Docker run was **not tested**. Use the venv method above for the live demonstration.

## H. Running the automated tests (optional, to show quality)
```powershell
cd backend; .\.venv\Scripts\Activate.ps1; pip install -r requirements-dev.txt
python -m pytest tests -q             # recorded result: 108 passed, 6 skipped
cd ..\frontend; npm test              # recorded result: 36 passed
```
The 6 skipped backend tests need your own private media (`VERIFAI_TEST_IMAGE`/`VERIFAI_TEST_VIDEO`) or a CUDA GPU.
