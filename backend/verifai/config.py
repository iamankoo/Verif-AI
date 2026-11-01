"""Runtime configuration (environment variables, optional backend/.env). No secrets are required."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional

BACKEND_DIR = Path(__file__).resolve().parent.parent
VERSION = "1.0.0"

# Constants that define the model's operating point. The decision threshold is NOT configurable on
# purpose: it must never be tuned to make results look better.
DECISION_THRESHOLD = 0.5
INPUT_SIZE = 224
TRAINING_JPEG_QUALITY = 95
FACE_MARGIN = 20


def _load_dotenv(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def _b(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def default_model_path() -> Path:
    # Temporary default (owner decision, 2026-11-01): Model 4 (DFF + Celeb-DF + quality augmentation), see docs/CURRENT_DEFAULT_MODEL.md.
    # The file is not in the repository; if it is absent the server falls back to production v2, then to the original prototype.
    for cand in (BACKEND_DIR / "models" / "verifai_v3_celebdf_qa.pth",
                 BACKEND_DIR / "models" / "verifai_v2_best.pth",
                 BACKEND_DIR / "scripts" / "two_stream_model_epoch_10.pth"):
        if cand.is_file():
            return cand
    return BACKEND_DIR / "models" / "verifai_v2_best.pth"


@dataclass(frozen=True)
class Settings:
    model_path: Path = field(default_factory=default_model_path)
    device: str = "auto"                      # auto | cpu | cuda | cuda:N
    storage_dir: Path = BACKEND_DIR / "storage"
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    max_image_mb: float = 15.0
    max_video_mb: float = 200.0
    max_image_pixels: int = 60_000_000
    min_image_side: int = 64
    detection_max_side: int = 1280            # images/frames are downscaled for detection only
    face_min_confidence: float = 0.90
    face_min_px: int = 32                     # smaller faces are ignored
    face_warn_px: int = 96                    # smaller faces trigger a low-resolution warning
    max_faces_per_image: int = 10
    uncertainty_margin: float = 0.15          # |score-0.5| below this => INCONCLUSIVE (abstain band, not a threshold)
    video_sample_fps: float = 2.0
    video_max_frames: int = 64
    video_max_duration_s: float = 900.0
    video_min_face_frames: int = 3
    video_batch_size: int = 16
    video_evidence_frames: int = 6
    video_aggregation: str = "median"         # mean | median | weighted_mean | suspicious_fraction
    enable_heatmaps: bool = True
    result_ttl_hours: float = 72.0
    infer_batch_size: int = 16
    api_prefix: str = "/api"
    public_history: bool = True               # False on public deployments: GET /api/analyses returns [] (results stay reachable only by their unguessable id)
    serve_frontend_dir: Optional[Path] = None

    @staticmethod
    def from_env(env: Optional[Mapping[str, str]] = None) -> "Settings":
        e = dict(_load_dotenv(BACKEND_DIR / ".env"))
        e.update(os.environ if env is None else env)
        s = Settings()
        g = lambda k, d: e.get(k, d)     # noqa: E731
        fe = g("VERIFAI_FRONTEND_DIR", "")
        return Settings(
            model_path=Path(g("VERIFAI_MODEL_PATH", str(s.model_path))),
            device=g("VERIFAI_DEVICE", s.device),
            storage_dir=Path(g("VERIFAI_STORAGE_DIR", str(s.storage_dir))),
            cors_origins=[o.strip() for o in g("VERIFAI_CORS_ORIGINS", ",".join(s.cors_origins)).split(",") if o.strip()],
            max_image_mb=float(g("VERIFAI_MAX_IMAGE_MB", s.max_image_mb)),
            max_video_mb=float(g("VERIFAI_MAX_VIDEO_MB", s.max_video_mb)),
            max_image_pixels=int(g("VERIFAI_MAX_IMAGE_PIXELS", s.max_image_pixels)),
            uncertainty_margin=float(g("VERIFAI_UNCERTAINTY_MARGIN", s.uncertainty_margin)),
            video_sample_fps=float(g("VERIFAI_VIDEO_SAMPLE_FPS", s.video_sample_fps)),
            video_max_frames=int(g("VERIFAI_VIDEO_MAX_FRAMES", s.video_max_frames)),
            video_max_duration_s=float(g("VERIFAI_VIDEO_MAX_DURATION_S", s.video_max_duration_s)),
            video_aggregation=g("VERIFAI_VIDEO_AGGREGATION", s.video_aggregation),
            enable_heatmaps=_b(g("VERIFAI_ENABLE_HEATMAPS", s.enable_heatmaps)),
            result_ttl_hours=float(g("VERIFAI_RESULT_TTL_HOURS", s.result_ttl_hours)),
            public_history=_b(g("VERIFAI_PUBLIC_HISTORY", s.public_history)),
            serve_frontend_dir=Path(fe) if fe else None,
        )
