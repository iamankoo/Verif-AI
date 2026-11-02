"""Lightweight filesystem persistence (no database).

Layout:  <storage_dir>/analyses/<analysis_id>/{job.json, result.json, artifacts/*.jpg}
Uploaded media is deleted right after analysis; only derived artifacts (annotated image / face crops /
evidence frames / heat-maps) and the JSON result are kept, until deleted via the API or expired (TTL).
"""
from __future__ import annotations

import json
import re
import os
import shutil
import threading
import time
from pathlib import Path
from typing import List, Optional

from .errors import NotFoundError
from .schemas import AnalysisResult, AnalysisSummary, JobState, JobStatus

_ID = re.compile(r"^[0-9a-f]{32}$")
_ART = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class AnalysisStore:
    def __init__(self, root: Path):
        self.root = Path(root) / "analyses"
        self.tmp = Path(root) / "tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths
    def _dir(self, analysis_id: str) -> Path:
        if not _ID.match(analysis_id):
            raise NotFoundError("Unknown analysis id.")
        return self.root / analysis_id

    def artifacts_dir(self, analysis_id: str) -> Path:
        d = self._dir(analysis_id) / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def artifact_path(self, analysis_id: str, name: str) -> Path:
        if not _ART.match(name) or ".." in name:
            raise NotFoundError("Unknown artifact.")
        p = self._dir(analysis_id) / "artifacts" / name
        if not p.is_file():
            raise NotFoundError("Unknown artifact.")
        return p

    # ---- job / result
    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        """Write via a temporary file and rename, so a concurrent reader (the browser polling /status while the worker saves progress)
        never sees an empty or half-written file."""
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(40):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:           # Windows: the target is momentarily open in a reader; retry shortly
                time.sleep(0.005 * (attempt + 1))
        try:
            path.write_text(text, encoding="utf-8")      # last resort: never lose a save
        finally:
            tmp.unlink(missing_ok=True)

    @staticmethod
    def _read_text(path: Path) -> str:
        """Read a file that may be replaced at this very moment (Windows can briefly refuse to open it)."""
        for attempt in range(40):
            try:
                return path.read_text(encoding="utf-8")
            except PermissionError:
                time.sleep(0.005 * (attempt + 1))
        return path.read_text(encoding="utf-8")

    def save_job(self, job: JobState) -> None:
        d = self._dir(job.analysis_id); d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._atomic_write(d / "job.json", job.model_dump_json())

    def load_job(self, analysis_id: str) -> JobState:
        p = self._dir(analysis_id) / "job.json"
        if not p.is_file():
            raise NotFoundError("Unknown analysis id.")
        return JobState.model_validate_json(self._read_text(p))

    def save_result(self, result: AnalysisResult) -> None:
        d = self._dir(result.analysis_id); d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._atomic_write(d / "result.json", result.model_dump_json())

    def load_result(self, analysis_id: str) -> Optional[AnalysisResult]:
        p = self._dir(analysis_id) / "result.json"
        return AnalysisResult.model_validate_json(self._read_text(p)) if p.is_file() else None

    def list_summaries(self, limit: int = 50) -> List[AnalysisSummary]:
        out: List[AnalysisSummary] = []
        for d in self.root.iterdir():
            if not d.is_dir() or not _ID.match(d.name):
                continue
            try:
                job = self.load_job(d.name)
                res = self.load_result(d.name)
                out.append(AnalysisSummary(analysis_id=d.name, media_type=job.media_type, filename=job.filename, status=job.status,
                                           classification=res.classification if res else None,
                                           fake_probability=res.fake_probability if res else None, created_at=job.created_at))
            except Exception:
                continue
        out.sort(key=lambda s: s.created_at, reverse=True)
        return out[:limit]

    def delete(self, analysis_id: str) -> None:
        d = self._dir(analysis_id)
        if not d.is_dir():
            raise NotFoundError("Unknown analysis id.")
        shutil.rmtree(d, ignore_errors=True)

    def cleanup_expired(self, ttl_hours: float) -> int:
        cutoff, n = time.time() - ttl_hours * 3600, 0
        for d in list(self.root.iterdir()):
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True); n += 1
        for f in list(self.tmp.iterdir()):
            if f.stat().st_mtime < time.time() - 3600:
                f.unlink(missing_ok=True)
        return n

    def mark_interrupted(self) -> int:
        """After a restart, jobs that were queued/running can never finish."""
        n = 0
        for d in self.root.iterdir():
            try:
                job = self.load_job(d.name)
            except Exception:
                continue
            if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                job.status, job.error, job.stage = JobStatus.FAILED, "Interrupted by a server restart.", "failed"
                self.save_job(job); n += 1
        return n
