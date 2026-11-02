# Quick Start

Short version for Windows PowerShell. Full explanations: [USER_RUNNING_GUIDE.md](USER_RUNNING_GUIDE.md). Demonstration steps: [IMAGE_VIDEO_DEMONSTRATION_GUIDE.md](IMAGE_VIDEO_DEMONSTRATION_GUIDE.md).

Verif-AI runs on **CPU** (no GPU needed). Tested with Python 3.10 and Node.js 24 / npm 11. No API keys or environment variables are required.

## Examiner demonstration (E4 default) - one command
```powershell
cd Verif-AI
powershell -ExecutionPolicy Bypass -File scripts\start-demo.ps1     # or double-click scripts\start-demo.bat
```
It builds the frontend once if needed, starts the API on the first free port from 8000 and opens the browser. The console prints `Default model: E4 ... checkpoint verifai_e4_candidate.pth`; the dashboard home page shows the same under *Model Information*. Switch models in the upload card (Model 4 = alternative, v2 = rollback). Requires `backend/models/verifai_e4_candidate.pth` (otherwise Model 4, then v2 is used - see `docs/E4_INTEGRATION.md`). Stop with Ctrl+C.

## A. First time only (about 10–20 minutes, needs internet)
```powershell
git clone https://github.com/iamankoo/Verif-AI.git
cd Verif-AI

# backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# frontend (build once; the backend will serve it)
cd ..\frontend
npm ci
npm run build
cd ..
```
If PowerShell refuses to run `Activate.ps1` (script execution disabled), run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window and try again.

## B. Every time you want to run it (one terminal)
```powershell
cd Verif-AI\backend
.\.venv\Scripts\Activate.ps1
$env:VERIFAI_FRONTEND_DIR = "..\frontend\dist"
python -m uvicorn verifai.api.app:app --host 127.0.0.1 --port 8000
```
Wait for the line `Application startup complete.`, then open **http://127.0.0.1:8000** in Chrome/Edge. Keep this terminal open. Stop with `Ctrl+C`.

Other URLs: health check http://127.0.0.1:8000/api/health · API documentation http://127.0.0.1:8000/docs

> Do **not** run `npm run dev` for the demo unless you want development mode (see the running guide, §C). With the command above the UI and the API are the same program on port 8000.

## C. Test an image (1 minute)
1. Open http://127.0.0.1:8000 → click **Browse files** (or drag a file into the dashed box).
2. Pick a `.jpg`/`.png` containing one face from anywhere on your laptop.
3. Click **Analyse image**.
4. Read the verdict badge (REAL / FAKE / INCONCLUSIVE), the Real/Fake scores, and the face box with the Grad-CAM heat-map.
5. Click **Download forensic report (PDF)**.

## D. Test a video (2–3 minutes)
1. Same page → **Browse files** → pick an `.mp4` (or `.avi .mov .mkv .webm .m4v`) containing a face, up to 200 MB.
2. Click **Analyse video** → a progress panel shows the stage; wait until the result page appears (a few seconds to some minutes on CPU).
3. Read the verdict, frame timeline, suspicious timestamps, face tracks and highest-scoring frames; download the PDF.

## E. Check that the model file is there
**Current default: Model 4** (`backend\models\verifai_v3_celebdf_qa.pth`, a temporary owner decision - see [CURRENT_DEFAULT_MODEL.md](CURRENT_DEFAULT_MODEL.md) for its measured strengths *and* weaknesses). This file is not in the repository; it is copied locally. If it is missing the server falls back to v2.
```powershell
cd Verif-AI
Test-Path backend\models\verifai_v3_celebdf_qa.pth        # should print True
(Get-FileHash backend\models\verifai_v3_celebdf_qa.pth -Algorithm SHA256).Hash   # should start E7E5F3DC28B2...
Test-Path backend\models\verifai_v2_best.pth              # production v2 (rollback), unchanged
(Get-FileHash backend\models\verifai_v2_best.pth -Algorithm SHA256).Hash   # should end ...a370d7 (3B94775581931210...)
```
Full expected hashes: Model 4 `e7e5f3dc28b252d43e2e9134e7f41185c49526cff5d1c03367d14ae3c...` (confirm with `/api/model`), v2 `3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7`.
To switch back to v2: set `VERIFAI_MODEL_PATH=models/verifai_v2_best.pth` in `backend\.env` and restart.

## F. If something fails
| Symptom | Quick fix |
|---|---|
| `Cannot reach the Verif-AI server. Is the backend running?` in the browser | Start the backend (section B) and refresh |
| `Address already in use` / port 8000 busy | Use `--port 8001` and open http://127.0.0.1:8001 |
| Page shows `{"detail":"Not Found"}` at `/` | `VERIFAI_FRONTEND_DIR` was not set in that terminal, or `npm run build` was not run |
| Health shows `degraded` | Model file missing/corrupt – see the running guide, §D |

Results are model-based evidence, not proof. Read the limitations in [EXAMINER_DEMONSTRATION_GUIDE.md](EXAMINER_DEMONSTRATION_GUIDE.md) before presenting.
