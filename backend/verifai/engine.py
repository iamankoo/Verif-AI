"""Engine = loaded model + detector + explainer + store + background video worker."""
from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from .config import Settings
from .errors import InvalidMediaError, JobCancelled, ModelUnavailableError, UnknownModelError, VerifAIError
from .explain import Explainer
from .faces import FaceDetector
from .image_pipeline import analyze_image
from .model import ModelBundle, load_model
from .registry import ModelRegistry
from .schemas import AnalysisResult, JobState, JobStatus, utcnow
from .store import AnalysisStore
from .video_pipeline import analyze_video

log = logging.getLogger("verifai.engine")


class Engine:
    def __init__(self, bundle: ModelBundle, settings: Settings, store: Optional[AnalysisStore] = None):
        self.settings = settings
        self.registry = ModelRegistry(settings)
        self.bundle = self._tag(bundle, self.registry.default_id)
        self.detector = FaceDetector(bundle.device, settings)
        self.explainer = Explainer(bundle)
        self.store = store or AnalysisStore(settings.storage_dir)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")   # CPU-bound: one video at a time
        self._cancel: Dict[str, threading.Event] = {}
        # Optional (non-default) models are loaded on first use, at most ONE at a time, so memory never holds more than two copies.
        self._extra: Dict[str, tuple] = {}
        self._extra_lock = threading.Lock()

    # ------------------------------------------------------------------ model selection
    def _tag(self, bundle: ModelBundle, model_id: str) -> ModelBundle:
        e = self.registry.entries.get(model_id)
        if e is not None:
            bundle.info = bundle.info.model_copy(update={"registry_id": e.id, "role": e.role, "role_label": e.role_label,
                                                       "registry_label": e.label, "registry_note": e.description})
        return bundle

    def loaded_ids(self):
        return [self.registry.default_id] + list(self._extra)

    def runtime(self, model_id: Optional[str] = None):
        """(bundle, explainer) for the requested model; None or the default id -> the already-loaded production model."""
        if not model_id or model_id == self.registry.default_id:
            return self.bundle, self.explainer
        if not self.registry.selectable(model_id):
            raise UnknownModelError(f"Model '{model_id}' is not available on this server.", "unknown_model")
        with self._extra_lock:
            if model_id not in self._extra:
                entry = self.registry.entries[model_id]
                try:
                    b = self._tag(load_model(entry.path, self.settings.device), model_id)
                except Exception as exc:
                    log.error("model %s failed to load: %s", model_id, exc)
                    raise ModelUnavailableError(f"Model '{model_id}' could not be loaded.", "model_unavailable")
                self._extra.clear()                       # drop the previously loaded optional model (running jobs keep their own reference)
                self._extra[model_id] = (b, Explainer(b))
                log.info("optional model loaded: %s (%s)", model_id, entry.path.name)
            return self._extra[model_id]

    # ------------------------------------------------------------------ images (synchronous)
    def run_image(self, path: Path, filename: str, model_id: Optional[str] = None) -> AnalysisResult:
        bundle, explainer = self.runtime(model_id)
        aid = uuid.uuid4().hex
        job = JobState(analysis_id=aid, status=JobStatus.RUNNING, media_type="image", filename=filename, stage="analysing", progress=0.1)
        self.store.save_job(job)
        try:
            res = analyze_image(path, filename, bundle, self.detector, explainer, self.settings,
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
    def submit_video(self, path: Path, filename: str, model_id: Optional[str] = None) -> JobState:
        bundle, explainer = self.runtime(model_id)       # validates the id before a job is created
        aid = uuid.uuid4().hex
        job = JobState(analysis_id=aid, status=JobStatus.QUEUED, media_type="video", filename=filename)
        self.store.save_job(job)
        ev = self._cancel[aid] = threading.Event()
        self._pool.submit(self._run_video, aid, path, filename, ev, bundle, explainer)
        return job

    def cancel(self, aid: str) -> None:
        ev = self._cancel.get(aid)
        if ev:
            ev.set()

    def _run_video(self, aid: str, path: Path, filename: str, ev: threading.Event, bundle: ModelBundle, explainer: Explainer) -> None:
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
            res = analyze_video(path, filename, bundle, self.detector, explainer, self.settings,
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
