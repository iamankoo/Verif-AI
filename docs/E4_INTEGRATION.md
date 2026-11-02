# E4 integration: model registry, artifacts, tests, hosting limits and rollback

**Summary.** E4 is integrated as an explicitly selectable **candidate**. **Model 4 is the production default, v2 is the rollback.** E4 failed the frozen promotion rule (criterion 2), so it was not promoted; its results are in [E4_RESULTS.md](E4_RESULTS.md). No weights, thresholds, datasets or evaluation protocols were changed.

## 1. Artifacts and verification
| Artifact | Location | Size | SHA-256 |
|---|---|---|---|
| Results archive `verifai_e4_results.zip` (checkpoint, metrics, manifest; no images) | Kaggle notebook output `iamankoo/verif-ai-e4`; kept locally at `backend/runs/e4/verifai_e4_results.zip` (git-ignored) | 44,934,864 B | `f9dd4f6c45ef4e385ca3ed2c0cc4f07a358fc5968e3a53a887d31170502599d4` |
| E4 checkpoint `runs/e4a_dffx4_modern_qa0.5/best.pth` | `backend/models/verifai_e4_candidate.pth` (git-ignored, **never committed**) | 48,119,236 B | `0d553de918c120994e0919386d7afb6378cd489cc9ef4abc2aa4850e5cf9cba0` |

Retrieval: the file list of the finished notebook was read through the Kaggle SDK and only `verifai_e4_results.zip` was downloaded (HTTP 200, byte count above). Checks performed and recorded: archive test passed; both SHA-256 values calculated (not assumed) and equal to the values documented in the E4 run record; the checkpoint contains 725 tensors plus metadata (`name`, `version`, `epoch`, `seed`, `trained_on`, `val_metrics`); it loads with the project's own `load_model()` using **strict** key matching and `weights_only=True`; 11,972,954 parameters, input 224, threshold 0.5 (same architecture as Model 4 and v2). The values are asserted by `backend/tests/test_model_registry.py` whenever the file is present.

## 2. Model registry (`backend/verifai/registry.py`)
| Role | Id | File | Offered when |
|---|---|---|---|
| **Production default** | `model4` (or whatever `VERIFAI_MODEL_PATH` points to; `v2` on the hosted demo) | `verifai_v3_celebdf_qa.pth` | always; used when a request names no model |
| **Rollback** | `v2` | `verifai_v2_best.pth` (in Git) | `VERIFAI_ENABLE_MODEL_SELECTION=true` and the file exists |
| **Candidate - not approved** | `e4` | `verifai_e4_candidate.pth` | same; labelled *Candidate - not approved for production* |

* `GET /api/models` lists them; `POST /api/analyze/image|video` take an optional form field `model`; an unknown/unavailable id is rejected with **422 `unknown_model`** before anything is stored; each result stores `model.registry_id`, `model.role`, `model.role_label` and the checkpoint hash (also in the PDF). See [API_DOCUMENTATION.md](API_DOCUMENTATION.md).
* The default model is loaded at start-up. An optional model is **lazy-loaded on first use and at most one is kept** next to the default (the previous one is dropped; running jobs keep their own reference), so memory never holds more than two copies. Face detector, preprocessing, scoring, tracking, Grad-CAM and report code are shared; only the checkpoint differs.
* The dashboard shows a model picker only when the server offers more than one model, preselects the production default, sends no `model` field for the default, shows a warning for a candidate and labels every result with a chip such as *Model: e4 - Candidate - not approved for production*.
* Environment: `VERIFAI_MODEL_PATH`, `VERIFAI_ROLLBACK_MODEL_PATH`, `VERIFAI_CANDIDATE_MODEL_PATH`, `VERIFAI_ENABLE_MODEL_SELECTION` (default `true` locally; `false` in the `Dockerfile` and `render.yaml`).

## 3. Local test results (actually executed on this machine)
| Suite | Result |
|---|---|
| Backend (pytest, full suite) | **165 passed, 7 skipped** (baseline before the change: 152 passed, 7 skipped; +12 `test_model_registry.py`, +1 `test_store_atomic.py`; the 7 skips are the unchanged environment-gated tests needing private media/GPU) |
| Frontend type-check (`tsc --noEmit`) | clean |
| Frontend unit/component tests (Vitest) | **66 passed** (63 before + 3 model-selection tests) |
| Frontend production build | OK |
| Playwright end-to-end (real Chrome, real server, public test media) | **10 passed, 2 skipped** (the 2 skips are the private-media tests, deliberately not run) - includes the new model-selection test |
| Real checkpoints | Model 4, v2 and E4 each match their documented SHA-256 and load strictly (`test_real_checkpoint_hash_and_load`) |
| Multi-model API smoke test (default, `v2`, `e4`; image and video) | all HTTP 200/202 -> completed; Grad-CAM image served; PDF reports valid (`%PDF`); `DELETE` -> 204 then 404; no temporary upload left in `storage/tmp` |
| Concurrency (8 simultaneous mixed-model image requests) | all 200, every result labelled with the requested model, untagged requests all used `model4`, only one optional model stayed loaded |
| Regression | the existing application tests (inference, pipelines, report, API, tracking, privacy/deletion, training smoke) pass unchanged |
| Local reproduction of the documented E4 numbers | E4 and Model 4 on the DeepFakeFace test splits agree with the Kaggle record to the third decimal ([E4_RESULTS.md](E4_RESULTS.md)) |

**A real bug found and fixed.** The smoke test occasionally got HTTP 500 from `/status`: the video worker rewrote `job.json` non-atomically while the browser polled it. Writes are now atomic (temporary file + `os.replace`, with a short retry for Windows) and reads retry briefly; `test_store_atomic.py` fails without the fix and passes with it.

**Observation (not a regression).** One full-suite run was killed by a native "access violation" inside Grad-CAM's backward pass while a server, the frontend tests and a build were running at the same time on this 7.3 GB laptop (0.3 GB free). Re-running the suite alone passed (165/7), as did the first full run; treat the suite as needing about 1.5 GB of free memory.

## 4. Performance and production readiness (measured, CPU, laptop)
| Measure | Model 4 | v2 | E4 |
|---|---|---|---|
| Checkpoint size | 48.1 MB | 48.1 MB | 48.1 MB |
| Load time | 0.48 s | 0.43 s | 0.47 s |
| Inference, batch of 16 crops | 1.82 s | 1.51 s | 1.70 s |
| Image analysis (1 face, Grad-CAM on) | 1.5 s | 4.8 s (first use, includes the lazy load) | 2.5 s (first use, includes the lazy load) |
| Video analysis (7.6 s public clip, 24 frames, Grad-CAM on) | 8.5 s | 14.4 s | 8.3 s |

Server working set peaked at **about 1.47 GB** with two models loaded and a video analysed with Grad-CAM (the single-model default needs about 0.8 GB, see [DEPLOYMENT.md](DEPLOYMENT.md)). Uploads are deleted after analysis (checked: 0 files left), results expire after the TTL, upload/duration limits are unchanged, and logs contain no secrets or media.

## 5. Hosting limits
* **Render free instance (the live API): 512 MB RAM, no disk, Grad-CAM off, v2 served.** Model 4 and E4 are not in Git (by policy: checkpoints are never committed and the repository is public), so the free instance cannot receive them; two models plus Grad-CAM also exceed 512 MB. Therefore **E4 is available as a candidate only in the supported local environment** (and on a paid >= 2 GB instance with a disk, see `docs/DEPLOYMENT.md`). The hosted API runs with `VERIFAI_ENABLE_MODEL_SELECTION=false` and `/api/models` lists only the model it really serves.
* The live site therefore does **not** run E4 and is **not** running Model 4 either: it runs v2 (as before this change). Making Model 4 live needs the checkpoint on host storage (a paid instance with a persistent disk, or an authorised private artifact store). This is an open item for the owner, not something that was bypassed.

## 6. Rollback procedures
* **Default model, local:** set `VERIFAI_MODEL_PATH=models/verifai_v2_best.pth` (or delete the setting when Model 4 is absent) and restart; verify `GET /api/model` (`checkpoint_sha256` `3b947755...`). Nothing else changes.
* **Hide candidates:** `VERIFAI_ENABLE_MODEL_SELECTION=false`.
* **Code:** this work is on branch `feat/e4-candidate-integration`; `git revert <merge commit>` on `release/live` (or redeploy the previous Render deploy from the Render dashboard -> Deploys -> *Rollback*, and promote the previous Vercel production deployment). The previous live frontend deployment was `verfai-5hh8fkkrq-iamankoos-projects.vercel.app`.
* No checkpoint was modified, moved or deleted, so every rollback is just configuration.

## 7. Deployment status (verified 2026-11-02)
| Item | Value |
|---|---|
| Git | feature branch `feat/e4-candidate-integration` (commits `3568468`, `0768a13`, `375dd54`); merged with a merge commit into the deployable branch `release/live` (`2f87f2d`) and `main` fast-forwarded to it; all pushed to `github.com/iamankoo/Verif-AI` (no force-push) |
| Backend | Render web service `verifai-api` (Docker, free tier, auto-deploy on `release/live`): https://verifai-api-11wg.onrender.com - redeployed from `2f87f2d`, health OK |
| Frontend | Vercel project `verfai` (production, deployed with the CLI): https://verfai.vercel.app |
| **Model actually serving the live site** | **v2** (`verifai_v2_best.pth`, SHA-256 `3b947755...370d7`), reported by `/api/model` and `/api/models` as the production default of that instance. **Not Model 4 and not E4** (see section 5) |
| Live `/api/models` | `{default: v2, selection_enabled: false, models: [v2]}` - candidates are not offered on the hosted instance |
| Live smoke tests (Playwright, real Chrome, public test image and synthetic clip, cross-origin Vercel -> Render) | 9 passed, 3 skipped (model selection: the live server offers one model; 2 private-media tests deliberately not run). Covered: site and API docs reachable, navigation, upload card, rejected files, deep links/refresh, image analysis with the Grad-CAM-unavailable state, PDF download, deletion, video job with timeline/tracks/evidence/PDF/deletion, responsive layouts, no console errors |
| Live API checks | public history list empty; analyses deleted by the tests return 404; the Render log window showed health checks 200 and no `Traceback`, `ERROR`, out-of-memory or 5xx lines |
| Previous live state kept for rollback | Vercel deployment `verfai-5hh8fkkrq-iamankoos-projects.vercel.app` (promote it in the Vercel dashboard); Render: Deploys -> previous deploy -> Rollback; or `git revert 2f87f2d -m 1` on `release/live` |
