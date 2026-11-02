# Deployment: Vercel (frontend) + Render (backend)

```
Browser ──► Vercel (static React build, hash routes) ──HTTPS + CORS──► Render web service (FastAPI + PyTorch, Docker)
                                                                         ├─ model checkpoint on a persistent disk (/data/models)
                                                                         └─ results and evidence images on the same disk (/data/storage, deleted after the TTL)
```
The frontend is static, so it can live on Vercel. The backend cannot: it needs PyTorch, a ~48 MB checkpoint, more than 800 MB of RAM during analysis, background video jobs and files that must survive between requests. Serverless functions provide none of that, so it runs as a long-lived container on Render.

## Environment variables
**Frontend (Vercel, build-time, public - never put secrets here):** `VITE_API_BASE_URL` = origin of the Render service, for example `https://<service>.onrender.com`. Leave it empty for local development (the Vite dev server proxies `/api`).

**Backend (Render):**
| Variable | Value | Purpose |
|---|---|---|
| `VERIFAI_MODEL_PATH` | `/data/models/verifai_v3_celebdf_qa.pth` | Current default model (Model 4). Falls back to nothing if the file is missing: `/api/health` then reports `model_loaded: false`. |
| `VERIFAI_STORAGE_DIR` | `/data/storage` | Results and evidence images; on the persistent disk. |
| `VERIFAI_CORS_ORIGINS` | the Vercel origin, e.g. `https://verfai.vercel.app` | Only these browser origins may call the API. Preview deployments need their own origin added. |
| `VERIFAI_PUBLIC_HISTORY` | `false` | Turns off the global `GET /api/analyses` list so visitors cannot see each other's file names and verdicts. Each browser remembers its own analyses. |
| `VERIFAI_RESULT_TTL_HOURS` | `24` | Evidence and results are deleted automatically after this time. |
| `VERIFAI_MAX_IMAGE_MB` / `VERIFAI_MAX_VIDEO_MB` / `VERIFAI_VIDEO_MAX_DURATION_S` | `15` / `100` / `120` | Upload and processing limits. |
| `VERIFAI_DEVICE` | `cpu` | Render has no GPU. |

No secret is required by the application. Nothing in the repository is a credential; do not commit `.env` files.

## Model selection on hosted instances
`VERIFAI_ENABLE_MODEL_SELECTION=false` (set in the `Dockerfile` and `render.yaml`) makes the hosted API offer only the model it serves. Model 4 and the E4 candidate are not in Git and the free instance has no disk, so the free-tier demonstration serves **v2** and cannot host E4. To offer E4 as a *candidate* on a paid instance: copy `verifai_e4_candidate.pth` to `/data/models/`, set `VERIFAI_CANDIDATE_MODEL_PATH=/data/models/verifai_e4_candidate.pth` and `VERIFAI_ENABLE_MODEL_SELECTION=true` (needs >= 2 GB RAM: two models plus Grad-CAM). Details and the rollback procedure: [E4_INTEGRATION.md](E4_INTEGRATION.md).

## Resource and hosting requirements (measured)
* Memory: about 490 MB after the model loads, about 780 MB peak per image analysis (Grad-CAM included), over 820 MB while a video is analysed. Use a plan with at least 2 GB; 512 MB plans will be killed.
* A persistent disk (paid instances only) is required: the instance filesystem is ephemeral and is wiped on every deploy and restart.
* Image analysis is synchronous (about 1-2 s on CPU); video analysis is a background job the browser polls, so HTTP request time limits are not a problem for videos. A single instance processes video jobs one after another; add instances only after adding shared storage.
* Uploads are deleted right after analysis; evidence images and results stay until the TTL or until the user deletes them.

## Free-tier ("lite") mode on Render
Render's free instance has 512 MB RAM, no persistent disk and spins down when idle (a cold start can take about a minute). Measured on a laptop with the CPU model loaded: idle about 490 MB; with Grad-CAM enabled a single image peaks near 670-780 MB and a video near 1.5 GB; with Grad-CAM disabled, small batches and one thread an image peaks near 510 MB and a short video near 580 MB. A default-configured free instance is killed during the first analysis (HTTP 502). The free tier therefore needs lite mode, and it cannot provide Grad-CAM heat-maps:
`VERIFAI_ENABLE_HEATMAPS=false`, `VERIFAI_INFER_BATCH_SIZE=2`, `VERIFAI_VIDEO_BATCH_SIZE=2`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `MALLOC_ARENA_MAX=2`, `VERIFAI_VIDEO_MAX_FRAMES=12`, small upload limits. With no persistent disk the model must be the checkpoint inside the repository (v2, `/app/models/verifai_v2_best.pth`), and results disappear on every restart. Full functionality (Model 4, heat-maps, video evidence, durable results) needs a plan with at least 2 GB RAM and a disk, as described below.

## One-time Render setup
1. Create the web service from `render.yaml` (Blueprint) on the branch you deploy. Plan with >= 2 GB RAM, a 2 GB disk mounted at `/data`.
2. Put the Model 4 checkpoint on the disk. It is intentionally not in Git (checkpoints are never committed and the repository is public). Copy it once with Render's SSH/SCP support: `scp <path>/verifai_v3_celebdf_qa.pth <service>@ssh.<region>.render.com:/data/models/` (create `/data/models` first). Verify the hash with `sha256sum` and with `GET /api/model` (`checkpoint_sha256` must start `e7e5f3dc28b2`).
3. Set `VERIFAI_CORS_ORIGINS` to the final Vercel origin. Redeploy or restart.
4. Check `GET /api/health` returns `{"status":"ok","model_loaded":true}`.

## Vercel setup
1. Project root directory: `frontend`. Framework preset: Vite (also pinned in `frontend/vercel.json`): install `npm ci`, build `npm run build`, output `dist`.
2. Set `VITE_API_BASE_URL` (Production) to the Render origin, then deploy. Routes are hash-based (`/#/model`), so a refresh on any page loads `index.html`; `vercel.json` also rewrites unknown paths to it.
3. Project name / URL: `verfai` (`https://verfai.vercel.app`), otherwise `verifai-anni`.

## Verification checklist (run against the public URLs)
Homepage and navigation; refresh on a deep link; image upload (heat-map, Original/Heatmap/Overlay); video upload, timeline, frame selection and heat-map; face tracks; PDF download; deletion; no console errors; mobile layout; `/api/health`, `/docs`; the checkpoint hash from `/api/model`.

## Known limitations and hardening still advisable
* There is no authentication or rate limiting: anyone with the URL can use CPU time. Put the API behind a rate limiter or access key before any wider release.
* Results are reachable by their unguessable 128-bit id (a capability URL); anyone you give the link to can open it until it expires or is deleted.
* Do not upload personal or sensitive media to a public demo instance.
