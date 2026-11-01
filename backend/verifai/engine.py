"""Engine = loaded model + detector + explainer + store + background video worker."""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from .config import Settings
from .errors import InvalidMediaError, JobCancelled, VerifAIError
from .explain import Explainer
from .faces import FaceDetector
from .image_pipeline import analyze_image
from .model import ModelBundle
from .schemas import AnalysisResult, JobState, JobStatus, utcnow
from .store import AnalysisStore
from .video_pipeline import analyze_video

log = logging.getLogger("verifai.engine")


class Engine:
    def __init__(self, bundle: ModelBundle, settings: Settings, store: Optional[AnalysisStore] = None):
        self.bundle, self.settings = bundle, settings
        self.detector = FaceDetector(bundle.device, settings)
        self.explainer = Explainer(bundle)
        self.store = store or AnalysisStore(settings.storage_dir)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")   # CPU-bound: one video at a time
        self._cancel: Dict[str, threading.Event] = {}

    # ------------------------------------------------------------------ images (synchronous)
    def run_image(self, path: Path, filename: str) -> AnalysisResult:
        aid = uuid.uuid4().hex
        job = JobState(analysis_id=aid, status=JobStatus.RUNNING, media_type="image", filename=filename, stage="analysing", progress=0.1)
        self.store.save_job(job)
        try:
            res = analyze_image(path, filename, self.bundle, self.detector, self.explainer, self.settings,
                                artifacts_dir=self.store.artifacts_dir(aid), analysis_id=aid)
            self.store.save_result(res)
            job.status, job.progress, job.stage = JobStatus.COMPLETED, 1.0, "done"
            self.store.save_job(job)
            return res
        except Exception as exc:
            job.status, job.error, job.stage = JobStatus.FAILED, (exc.message if isinstance(exc, VerifAIError) else "Analysis failed."), "failed"
            self.store.save_job(job)
            raise

    # ------------------------------------------------------------------ videos (background)
    def submit_video(self, path: Path, filename: str) -> JobState:
        aid = uuid.uuid4().hex
        job = JobState(analysis_id=aid, status=JobStatus.QUEUED, media_type="video", filename=filename)
        self.store.save_job(job)
        ev = self._cancel[aid] = threading.Event()
        self._pool.submit(self._run_video, aid, path, filename, ev)
        return job

    def cancel(self, aid: str) -> None:
        ev = self._cancel.get(aid)
        if ev:
            ev.set()

    def _run_video(self, aid: str, path: Path, filename: str, ev: threading.Event) -> None:
        job = self.store.load_job(aid)
        job.status, job.stage = JobStatus.RUNNING, "starting"
        self.store.save_job(job)
        last = {"t": 0.0}

        def progress(stage: str, frac: float) -> None:
            job.stage, job.progress = stage, round(float(frac), 3)
            if frac - last["t"] >= 0.03 or frac >= 1.0:
                last["t"] = frac
                self.store.save_job(job)

        try:
            res = analyze_video(path, filename, self.bundle, self.detector, self.explainer, self.settings,
                                artifacts_dir=self.store.artifacts_dir(aid), analysis_id=aid, progress=progress, cancel=ev.is_set)
            self.store.save_result(res)
            job.status, job.progress, job.stage = JobStatus.COMPLETED, 1.0, "done"
        except JobCancelled:
            job.status, job.error, job.stage = JobStatus.FAILED, "Analysis cancelled.", "cancelled"
        except VerifAIError as exc:
            job.status, job.error, job.stage = JobStatus.FAILED, exc.message, "failed"
        except Exception:
            log.exception("video analysis %s failed", aid)
            job.status, job.error, job.stage = JobStatus.FAILED, "Analysis failed unexpectedly.", "failed"
        finally:
            self.store.save_job(job)
            self._cancel.pop(aid, None)
            try:
                Path(path).unlink(missing_ok=True)          # uploaded media is never retained
            except Exception:
                pass

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
