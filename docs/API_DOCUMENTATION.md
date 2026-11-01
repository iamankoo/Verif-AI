# API Documentation

Base URL (local): `http://localhost:8000` · prefix `/api` · interactive docs: **`/docs`** (Swagger UI), `/redoc`, `/openapi.json`.
All responses are JSON unless stated. Schemas are defined in `backend/verifai/schemas.py` (Pydantic v2).

## Endpoints

| Method | Path | Purpose | Success |
|---|---|---|---|
| GET | `/api/health` | Liveness + model status | 200 `HealthResponse` |
| GET | `/api/model` | Model card, limits, explainability availability | 200 |
| POST | `/api/analyze/image` | Analyse an image (synchronous) | 200 `AnalysisResult` |
| POST | `/api/analyze/video` | Submit a video (background job) | 202 `JobState` |
| GET | `/api/analyses?limit=50` | Recent analyses | 200 `[AnalysisSummary]` |
| GET | `/api/analyses/{id}/status` | Job status / progress | 200 `JobState` |
| GET | `/api/analyses/{id}` | Full result | 200 `AnalysisResult` (409 if not finished/failed) |
| GET | `/api/analyses/{id}/report.pdf` | Forensic report | 200 `application/pdf` |
| GET | `/api/analyses/{id}/artifacts/{name}` | Evidence image (annotated image, face crop, Grad-CAM, video frame) | 200 `image/jpeg` |
| DELETE | `/api/analyses/{id}` | Delete results and cancel a running job | 204 |

`{id}` is a 32-character lowercase hex string; anything else returns 404.

### Health
```json
GET /api/health -> {"status":"ok","model_loaded":true,"device":"cpu","version":"1.0.0","detail":null}
```
If the checkpoint cannot be loaded the service starts in **degraded** mode: `status:"degraded"`, `model_loaded:false`, `detail` = the error; analysis endpoints return **503**.

### Image analysis
```bash
curl -F "file=@photo.jpg" http://localhost:8000/api/analyze/image
```
Accepted: `.jpg .jpeg .png .bmp .webp .tif .tiff`, default limit 15 MB / 60 megapixels / min side 64 px.
```json
{
  "analysis_id": "3f6c…", "status": "completed", "media_type": "image", "filename": "photo.jpg",
  "file_size_bytes": 183441, "file_sha256": "…", "created_at": "2026-11-01T10:00:00Z", "completed_at": "…",
  "classification": "REAL",            // REAL | FAKE | INCONCLUSIVE
  "real_probability": 0.97, "fake_probability": 0.03,   // uncalibrated model scores, see score_semantics
  "decisiveness": 0.94, "decisiveness_label": "high",   // |score-0.5|*2 (NOT a probability of being correct)
  "faces_detected": 1,
  "faces": [{"index":0,"box":[x1,y1,x2,y2],"detection_confidence":0.99,"fake_probability":0.03,
             "real_probability":0.97,"label":"REAL","crop_artifact":"face_0_crop.jpg","heatmap_artifact":"face_0_heatmap.jpg"}],
  "annotated_artifact": "annotated.jpg", "video": null,
  "explanation": ["…"], "warnings": ["…"], "scope_note": "…", "score_semantics": "…",
  "model": {"name":"…","version":"…","checkpoint_file":"…","checkpoint_sha256":"…","parameters":11972954,"device":"cpu","decision_threshold":0.5,…},
  "processing_time_s": 1.42, "aggregation_note": "Image score = maximum face fake-score."
}
```
* Multiple faces are scored independently; the image score is the **maximum** face score.
* No usable face ⇒ `classification: "INCONCLUSIVE"`, scores `null`, HTTP **200** (this is a valid outcome, not an error).
* A score within ±0.15 of 0.5 is reported `INCONCLUSIVE`; the 0.5 decision threshold itself is fixed.

### Video analysis
```bash
curl -F "file=@clip.mp4" http://localhost:8000/api/analyze/video        # -> 202 {"analysis_id":"…","status":"queued",…}
curl http://localhost:8000/api/analyses/$ID/status                      # {"status":"running","stage":"analysing","progress":0.42}
curl http://localhost:8000/api/analyses/$ID                             # AnalysisResult when completed
curl -o report.pdf http://localhost:8000/api/analyses/$ID/report.pdf
```
Accepted: `.mp4 .avi .mov .mkv .webm .m4v` *where the installed codecs permit* (default ≤ 200 MB). `JobState.status ∈ queued | running | completed | failed`. `AnalysisResult.video` contains: `duration_s, fps, width, height, frames_sampled, frames_with_faces, corrupt_frames_skipped, sampling_fps, aggregation_method, aggregation_validated (false), frames[] (timestamp, per-face scores, track ids), tracks[], suspicious_segments[] (start_s, end_s), evidence_frames[], temporal{score_std, mean_abs_frame_delta, longest_suspicious_run, suspicious_fraction, sign_flips}`.

### Model information
`GET /api/model` returns `model` (name, version, sha256, parameters, device, `trained_on`, `training_notes`, `metrics_note`), `scope_note`, `limits` (max sizes, formats, sampling fps/frames, `uncertainty_margin`, `decision_threshold`, `video_aggregation`, `retention_hours`) and `explainability.grad_cam_available`.

## Errors
Error body: `{"detail": "<human readable>", "code": "<machine code>", "extra": {}}`.
| HTTP | `code` | When |
|---|---|---|
| 404 | `not_found` | unknown analysis id / artifact |
| 409 | `not_ready`, `failed` | result requested before completion, or the job failed (detail contains the reason) |
| 413 | `payload_too_large` | file above the size limit |
| 415 | `unsupported_media_type` | extension not allowed |
| 422 | `empty_file`, `unreadable_image`, `image_too_small`, `image_too_large`, `unreadable_video`; FastAPI validation errors (missing `file` field) | corrupt/unsupported/out-of-limit media |
| 503 | `model_unavailable` | checkpoint not loaded |

## Configuration (`backend/.env`, see `.env.example`)
`VERIFAI_MODEL_PATH`, `VERIFAI_DEVICE` (`auto|cpu|cuda`), `VERIFAI_STORAGE_DIR`, `VERIFAI_RESULT_TTL_HOURS`, `VERIFAI_MAX_IMAGE_MB`, `VERIFAI_MAX_VIDEO_MB`, `VERIFAI_MAX_IMAGE_PIXELS`, `VERIFAI_UNCERTAINTY_MARGIN`, `VERIFAI_VIDEO_SAMPLE_FPS`, `VERIFAI_VIDEO_MAX_FRAMES`, `VERIFAI_VIDEO_MAX_DURATION_S`, `VERIFAI_VIDEO_AGGREGATION`, `VERIFAI_ENABLE_HEATMAPS`, `VERIFAI_CORS_ORIGINS`, `VERIFAI_FRONTEND_DIR`. No API keys exist or are required.

## Retention and privacy
Uploads are deleted right after analysis (also when it fails). Derived artifacts and JSON stay for the TTL or until `DELETE`. Evidence images contain faces: treat the storage directory as sensitive data.

## Python usage (no server)
```python
from pathlib import Path
from verifai.config import Settings
from verifai.model import load_model
from verifai.faces import FaceDetector
from verifai.explain import Explainer
from verifai.image_pipeline import analyze_image

s = Settings.from_env()
bundle = load_model(s.model_path, s.device)
res = analyze_image(Path("photo.jpg"), "photo.jpg", bundle, FaceDetector(bundle.device, s), Explainer(bundle), s, artifacts_dir=Path("out"))
print(res.classification, res.fake_probability)
```
