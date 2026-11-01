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
| 11 | Performance and deployment (Docker) | See completion report for the Docker verification status | `Dockerfile`, `docs/INSTALLATION_GUIDE.md` |
| 12 | Documentation | Done | `README.md`, `docs/` |
| 13 | Cleanup and secret scan | Done | `.gitignore`, completion report |
| 14 | Final verification | See completion report | completion report |
| 15 | GitHub delivery | See completion report for commit hashes | completion report |

## Not done / out of scope
- Validation on face-swap video benchmarks (Celeb-DF, FaceForensics++, DFDC): restricted datasets, not obtainable without agreements.
- Calibrated probabilities; audio analysis; non-face content.
