"""FastAPI application factory.

Run:  uvicorn verifai.api.app:app --host 0.0.0.0 --port 8000      (from backend/)
Docs: http://localhost:8000/docs  (Swagger UI)   /redoc   /openapi.json
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import VERSION, Settings
from ..engine import Engine
from ..errors import (InvalidMediaError, ModelUnavailableError, PayloadTooLargeError, UnsupportedMediaTypeError, VerifAIError)
from ..image_pipeline import IMAGE_EXTENSIONS
from ..model import load_model
from ..report import build_pdf
from ..schemas import (AnalysisResult, AnalysisSummary, ErrorResponse, HealthResponse, JobState, JobStatus, ModelInfo, SCOPE_NOTE)
from ..video_pipeline import VIDEO_EXTENSIONS

log = logging.getLogger("verifai.api")

DESCRIPTION = """
**Verif-AI** analyses human images and videos with a two-stream (RGB + SRM noise-residual) EfficientNet-B0
face classifier and returns a structured forensic result.

* Scores are **uncalibrated** model outputs; **INCONCLUSIVE** is a valid outcome.
* Results are model-based evidence, not proof of manipulation or authenticity.
"""


def _save_upload(file: UploadFile, dest_dir: Path, allowed: set, max_bytes: int, kind: str) -> Path:
    name = os.path.basename(file.filename or "")
    ext = Path(name).suffix.lower()
    if ext not in allowed:
        raise UnsupportedMediaTypeError(f"Unsupported {kind} type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(allowed))}.", "unsupported_media_type")
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"       # never use the client-supplied name on disk
    size = 0
    try:
        with open(dest, "wb") as out:
            while True:
                chunk = file.file.read(1 << 20)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise PayloadTooLargeError(f"File exceeds the {max_bytes / 1e6:.0f} MB limit for {kind}s.", "payload_too_large")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise InvalidMediaError("The uploaded file is empty.", "empty_file")
    return dest


def create_app(settings: Optional[Settings] = None, engine: Optional[Engine] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    state: dict = {"engine": engine, "load_error": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["engine"] is None:
            try:
                bundle = load_model(settings.model_path, settings.device)
                state["engine"] = Engine(bundle, settings)
                log.info("model loaded: %s on %s", bundle.info.checkpoint_file, bundle.device)
            except Exception as exc:               # degraded mode: health reports it, analysis returns 503
                state["load_error"] = f"{type(exc).__name__}: {exc}"
                log.error("model failed to load: %s", state["load_error"])
        eng = state["engine"]
        if eng is not None:
            eng.store.mark_interrupted()
            eng.store.cleanup_expired(settings.result_ttl_hours)
        yield
        if state["engine"] is not None and engine is None:
            state["engine"].shutdown()

    app = FastAPI(title="Verif-AI API", version=VERSION, description=DESCRIPTION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.exception_handler(VerifAIError)
    async def domain_error(_: Request, exc: VerifAIError):
        return JSONResponse(status_code=exc.http_status, content=ErrorResponse(detail=exc.message, code=exc.code).model_dump())

    def eng() -> Engine:
        if state["engine"] is None:
            raise ModelUnavailableError(f"Model is not loaded: {state['load_error'] or 'unknown error'}", "model_unavailable")
        return state["engine"]

    api = APIRouter(prefix=settings.api_prefix)

    @api.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        e = state["engine"]
        if e is None:
            return HealthResponse(status="degraded", model_loaded=False, device="n/a", version=VERSION, detail=state["load_error"])
        return HealthResponse(status="ok", model_loaded=True, device=str(e.bundle.device), version=VERSION)

    @api.get("/model", tags=["system"])
    def model_info():
        e = eng()
        return {"model": e.bundle.info.model_dump(), "scope_note": SCOPE_NOTE,
                "limits": {"max_image_mb": settings.max_image_mb, "max_video_mb": settings.max_video_mb,
                           "image_formats": sorted(IMAGE_EXTENSIONS), "video_formats": sorted(VIDEO_EXTENSIONS),
                           "video_sample_fps": settings.video_sample_fps, "video_max_frames": settings.video_max_frames,
                           "video_max_duration_s": settings.video_max_duration_s,
                           "uncertainty_margin": settings.uncertainty_margin, "decision_threshold": 0.5,
                           "video_aggregation": settings.video_aggregation, "retention_hours": settings.result_ttl_hours},
                "explainability": {"grad_cam_available": e.explainer.available and settings.enable_heatmaps}}

    @api.post("/analyze/image", response_model=AnalysisResult, tags=["analysis"], responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
    def analyze_image_endpoint(file: UploadFile = File(...)):
        e = eng()
        display = os.path.basename(file.filename or "image")[:120]
        path = _save_upload(file, e.store.tmp, IMAGE_EXTENSIONS, int(settings.max_image_mb * 1e6), "image")
        try:
            return e.run_image(path, display)
        finally:
            path.unlink(missing_ok=True)

    @api.post("/analyze/video", response_model=JobState, status_code=202, tags=["analysis"], responses={413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}})
    def analyze_video_endpoint(file: UploadFile = File(...)):
        e = eng()
        display = os.path.basename(file.filename or "video")[:120]
        path = _save_upload(file, e.store.tmp, VIDEO_EXTENSIONS, int(settings.max_video_mb * 1e6), "video")
        return e.submit_video(path, display)

    @api.get("/analyses", response_model=List[AnalysisSummary], tags=["analysis"])
    def list_analyses(limit: int = 50):
        return eng().store.list_summaries(max(1, min(limit, 200)))

    @api.get("/analyses/{analysis_id}/status", response_model=JobState, tags=["analysis"])
    def analysis_status(analysis_id: str):
        return eng().store.load_job(analysis_id)

    @api.get("/analyses/{analysis_id}", response_model=AnalysisResult, tags=["analysis"], responses={409: {"model": ErrorResponse}})
    def analysis_result(analysis_id: str):
        st = eng().store
        job = st.load_job(analysis_id)
        res = st.load_result(analysis_id)
        if res is None:
            raise _NotReady(job)
        return res

    @api.get("/analyses/{analysis_id}/report.pdf", tags=["reports"], response_class=Response, responses={200: {"content": {"application/pdf": {}}}})
    def analysis_report(analysis_id: str):
        st = eng().store
        res = st.load_result(analysis_id)
        if res is None:
            raise _NotReady(st.load_job(analysis_id))
        pdf = build_pdf(res, st.artifacts_dir(analysis_id))
        return Response(pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="verifai-report-{analysis_id[:8]}.pdf"'})

    @api.get("/analyses/{analysis_id}/artifacts/{name}", tags=["analysis"], response_class=FileResponse)
    def analysis_artifact(analysis_id: str, name: str):
        return FileResponse(eng().store.artifact_path(analysis_id, name), media_type="image/jpeg")

    @api.delete("/analyses/{analysis_id}", status_code=204, tags=["analysis"])
    def delete_analysis(analysis_id: str):
        e = eng(); e.cancel(analysis_id); e.store.delete(analysis_id)
        return Response(status_code=204)

    app.include_router(api)

    fe = settings.serve_frontend_dir
    if fe and Path(fe).is_dir():
        if (Path(fe) / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=Path(fe) / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            cand = (Path(fe) / full_path).resolve()
            if full_path and Path(fe).resolve() in cand.parents and cand.is_file():
                return FileResponse(cand)
            return FileResponse(Path(fe) / "index.html")
    return app


class _NotReady(VerifAIError):
    http_status = 409
    code = "not_ready"

    def __init__(self, job: JobState):
        msg = f"Analysis is {job.status.value}" + (f": {job.error}" if job.error else "") + "."
        super().__init__(msg, "failed" if job.status == JobStatus.FAILED else "not_ready")


app = create_app()          # cheap: the model is loaded lazily in the lifespan handler
