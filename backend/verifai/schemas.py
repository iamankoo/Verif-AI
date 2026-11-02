"""Pydantic schemas shared by the pipelines, the API and the report generator."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Classification(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    INCONCLUSIVE = "INCONCLUSIVE"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


SCORE_NOTE = (
    "Scores are raw sigmoid outputs of a neural network. They are NOT calibrated probabilities and "
    "NOT an estimate of how likely the media is to be manipulated in the real world."
)
SCOPE_NOTE = (
    "This model analyses faces. It was trained to separate real face photographs from fabricated ones "
    "(see the model card for the exact training data). It is not validated on every generator or "
    "manipulation type, and a REAL result is not proof of authenticity."
)


class ModelInfo(BaseModel):
    name: str
    version: str
    architecture: str = "Two-stream EfficientNet-B0 (RGB + SRM-style noise residual), late fusion"
    checkpoint_file: str
    checkpoint_sha256: str
    parameters: int
    device: str
    input_size: int = 224
    decision_threshold: float = 0.5
    trained_on: Optional[str] = None
    training_notes: Optional[str] = None
    metrics_note: Optional[str] = None
    registry_id: Optional[str] = None            # model registry id (model4 / v2 / e4 / ...)
    role: Optional[str] = None                   # production | rollback | candidate
    role_label: Optional[str] = None


class FaceResult(BaseModel):
    index: int
    box: List[float] = Field(description="[x1, y1, x2, y2] in original-image pixels")
    detection_confidence: float
    fake_probability: float
    real_probability: float
    label: Classification
    crop_artifact: Optional[str] = None
    heatmap_artifact: Optional[str] = None
    track_id: Optional[int] = None


class FrameFace(BaseModel):
    track_id: int
    box: List[float]
    detection_confidence: float
    fake_probability: float


class FrameResult(BaseModel):
    frame_index: int
    timestamp_s: float
    faces: List[FrameFace]
    frame_fake_probability: Optional[float] = None
    suspicious: bool = False


class TrackSummary(BaseModel):
    track_id: int
    frames: int
    first_timestamp_s: float
    last_timestamp_s: float
    median_fake_probability: float
    mean_fake_probability: float
    suspicious_fraction: float


class SuspiciousSegment(BaseModel):
    start_s: float
    end_s: float
    frames: int
    max_fake_probability: float


class EvidenceFrame(BaseModel):
    frame_index: int
    timestamp_s: float
    fake_probability: float
    frame_artifact: str
    heatmap_artifact: Optional[str] = None


class TemporalStats(BaseModel):
    score_std: float
    mean_abs_frame_delta: float
    longest_suspicious_run: int
    suspicious_fraction: float
    sign_flips: int


class VideoSummary(BaseModel):
    duration_s: float
    fps: float
    width: int
    height: int
    total_frames_reported: int
    frames_sampled: int
    frames_with_faces: int
    corrupt_frames_skipped: int
    sampling_fps: float
    aggregation_method: str
    aggregation_validated: bool = False
    frames: List[FrameResult]
    tracks: List[TrackSummary]
    suspicious_segments: List[SuspiciousSegment]
    evidence_frames: List[EvidenceFrame]
    temporal: Optional[TemporalStats] = None


class AnalysisResult(BaseModel):
    analysis_id: str
    status: JobStatus = JobStatus.COMPLETED
    media_type: Literal["image", "video"]
    filename: str
    file_size_bytes: int
    file_sha256: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None
    classification: Optional[Classification] = None
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    score_semantics: str = SCORE_NOTE
    decisiveness: Optional[float] = Field(
        default=None, description="|score-0.5|*2 in [0,1]: distance from the decision boundary. Not a probability of correctness.")
    decisiveness_label: Optional[Literal["low", "moderate", "high"]] = None
    faces_detected: int = 0
    faces: List[FaceResult] = []
    annotated_artifact: Optional[str] = None
    video: Optional[VideoSummary] = None
    explanation: List[str] = []
    warnings: List[str] = []
    scope_note: str = SCOPE_NOTE
    model: Optional[ModelInfo] = None
    processing_time_s: Optional[float] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    aggregation_note: Optional[str] = None


class JobState(BaseModel):
    analysis_id: str
    status: JobStatus
    media_type: Literal["image", "video"]
    filename: str
    progress: float = 0.0
    stage: str = "queued"
    created_at: datetime = Field(default_factory=utcnow)
    error: Optional[str] = None


class AnalysisSummary(BaseModel):
    analysis_id: str
    media_type: str
    filename: str
    status: JobStatus
    classification: Optional[Classification] = None
    fake_probability: Optional[float] = None
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: str
    version: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
    extra: Dict[str, str] = {}
