# Phases

Status of the delivery plan. "Done" means the listed verification was actually executed; see `docs/PROJECT_EXECUTION_LOG.md` for commands and `docs/PROJECT_COMPLETION_REPORT.md` for the final numbers.

| # | Phase | Status | Evidence |
|---|---|---|---|
| 1 | Audit and baseline (original checkpoint measured on independent data) | Done | `docs/project-analysis/FINAL_BASELINE_AUDIT.md`, `docs/results/baseline_original_metrics.json` |
| 2 | Dataset and leakage-safe data pipeline | Done | `docs/DATASET_DOCUMENTATION.md`, `backend/training/`, tests |
| 3 | Training pipeline and retraining (Kaggle GPU, 3 configurations, validation-only selection) | Done | `docs/MODEL_EVALUATION_REPORT.md`, `docs/results/` |
| 4 | Image pipeline | Done | `backend/verifai/image_pipeline.py`, tests, e2e smoke |
| 5 | Video pipeline (sampled frames, tracking, aggregation) | Done — video-level accuracy **not measured** (no labelled videos) | `backend/verifai/video_pipeline.py`, e2e smoke |
| 6 | Forensic analysis (Grad-CAM, evidence, PDF report) | Done | `backend/verifai/explain.py`, `report.py`, tests |
| 7 | FastAPI backend | Done | `backend/verifai/api/`, `docs/API_DOCUMENTATION.md`, tests |
| 8 | React frontend (real endpoints only) | Done | `frontend/`, 36 tests, screenshots |
| 9 | Test suites and real-server smoke test | Done | see completion report for counts |
| 10 | Model evaluation report | Done | `docs/MODEL_EVALUATION_REPORT.md` |
| 11 | Performance and deployment (Docker) | Done — image built and container verified in GitHub Codespaces | `Dockerfile`, `docs/INSTALLATION_GUIDE.md` |
| 12 | Documentation | Done | `README.md`, `docs/` |
| 13 | Cleanup and secret scan | Done | `.gitignore`, completion report |
| 14 | Final verification | Done | completion report |
| 15 | GitHub delivery | Done — pushed to `main` and verified on GitHub | completion report |
| 16 | Experiments E1-E4 (shortcut diagnosis, stage-0 gate, modern-generator and source-balanced training under a frozen promotion rule) | Done - **no candidate promoted** (E3A and E4 fail criterion 2) | `docs/E1_QUALITY_BIN_DIAGNOSTIC.md`, `docs/E3_RESULTS.md`, `docs/E4_RESULTS.md` |
| 17 | Model registry: E4 default for the local demonstration, Model 4 alternative, v2 rollback (API, dashboard, labelled results); `scripts/start-demo.ps1` | Done - tests: see `docs/E4_INTEGRATION.md` | `backend/verifai/registry.py`, `backend/tests/test_model_registry.py` |
| 18 | Redeploy (Render backend + Vercel frontend) and live verification | Done 2026-11-02 - live API serves **v2** (Model 4 / E4 cannot be hosted on the free tier); live E2E 9 passed / 3 skipped - see `docs/E4_INTEGRATION.md` section 7 | `docs/DEPLOYMENT.md` |

## Not done / out of scope
- Validation on face-swap video benchmarks (Celeb-DF, FaceForensics++, DFDC): restricted datasets, not obtainable without agreements.
- Calibrated probabilities; audio analysis; non-face content.
