"""Multi-frame video analysis pipeline (replaces the original frame-0-only demo).

Flow: probe metadata -> plan uniform sampling (<= video_max_frames, ~video_sample_fps) -> sequential decode
of the sampled frames -> detect all faces -> IoU tracking -> batched inference -> per-frame / per-track scores
-> video-level aggregation -> temporal statistics + suspicious segments -> evidence frames (+Grad-CAM).

Video-level decision
  * every track with >= video_min_face_frames observations gets a track score = aggregate(method, its scores);
  * the video score is the MAXIMUM track score (a swapped face is usually consistent on ONE person);
  * if no track is long enough but enough frames contain faces, all observations are pooled;
  * otherwise the result is INCONCLUSIVE (not enough evidence).
The default aggregation method (median) was chosen a priori. It has NOT been validated on labelled videos
(none were available); `training/evaluate_video_aggregation.py` performs validation-only selection when a
labelled video validation manifest exists.
"""
from __future__ import annotations

import hashlib
import heapq
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DECISION_THRESHOLD, Settings
from .errors import InvalidMediaError, JobCancelled
from .explain import CAM_NOTE, Explainer, encode_jpeg
from .faces import DetectedFace, FaceDetector, IoUTracker
from .image_pipeline import COLORS, score_crops
from .model import ModelBundle
from .preprocess import preprocess_faces
from .schemas import (AnalysisResult, Classification, EvidenceFrame, FrameFace, FrameResult, JobStatus,
                      TrackSummary, VideoSummary, utcnow)
from .scoring import aggregate, classify, decisiveness, suspicious_segments, temporal_stats

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ProgressFn = Callable[[str, float], None]


def probe_video(path: Path) -> Dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InvalidMediaError("The file could not be opened as a video (unsupported or corrupt container/codec).", "unreadable_video")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise InvalidMediaError("The video contains no decodable frames.", "unreadable_video")
        if w <= 0 or h <= 0:
            h, w = frame.shape[:2]
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0 or fps > 1000:
        fps = 25.0
        fps_guess = True
    else:
        fps_guess = False
    return dict(fps=fps, frame_count=max(n, 0), width=w, height=h, duration_s=(n / fps if n > 0 else 0.0), fps_guessed=fps_guess)


def plan_sampling(frame_count: int, fps: float, sample_fps: float, max_frames: int, max_duration_s: float) -> Tuple[List[int], List[str]]:
    warns: List[str] = []
    usable = frame_count
    if frame_count / fps > max_duration_s:
        usable = int(max_duration_s * fps)
        warns.append(f"Video is longer than {max_duration_s:.0f}s; only the first {max_duration_s:.0f}s were analysed.")
    n_target = int((usable / fps) * sample_fps) + 1
    n = max(1, min(n_target, max_frames, usable))
    idx = sorted(set(int(round(v)) for v in np.linspace(0, usable - 1, n)))
    if len(idx) > 1:
        gap = (usable / fps) / (len(idx) - 1)
        if gap > 5.0:
            warns.append(f"Sampling is sparse (one frame every {gap:.1f}s); brief manipulated segments may be missed.")
    return idx, warns


def _annotate_frame(rgb: np.ndarray, faces: List[Dict]) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    t = max(2, int(round(max(img.shape[:2]) / 400)))
    for f in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in f["box"]]
        col = COLORS[f["label"]][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
        cv2.putText(img, f"T{f['track']} {f['p']:.2f}", (x1 + 2, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, t / 3), col, max(1, t - 1), cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _shrink(rgb: np.ndarray, max_side: int = 960) -> Tuple[np.ndarray, float]:
    s = min(1.0, max_side / max(rgb.shape[:2]))
    if s >= 1.0:
        return rgb, 1.0
    return cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA), s


def video_score_from_frames(frames: List[FrameResult], method: str, min_track_frames: int) -> Tuple[Optional[float], str, Dict[int, float]]:
    """Video-level score from per-frame face results. Used by the pipeline AND by offline evaluation, so
    evaluated numbers and production behaviour cannot drift apart.
    Returns (score or None, 'tracks'|'pooled'|'none', {track_id: track_score})."""
    by_track: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for fr in frames:
        for f in fr.faces:
            w = f.detection_confidence * min(f.box[2] - f.box[0], f.box[3] - f.box[1])
            by_track[f.track_id].append((f.fake_probability, w))
    scores = {tid: aggregate([p for p, _ in rows], method, [w for _, w in rows])
              for tid, rows in by_track.items() if len(rows) >= min_track_frames}
    if scores:
        return max(scores.values()), "tracks", scores
    rows = [r for rs in by_track.values() for r in rs]
    if rows:
        return aggregate([p for p, _ in rows], method, [w for _, w in rows]), "pooled", {}
    return None, "none", {}


def analyze_video(path: Path, filename: str, bundle: ModelBundle, detector: FaceDetector, explainer: Optional[Explainer],
                  settings: Settings, artifacts_dir: Optional[Path] = None, analysis_id: Optional[str] = None,
                  progress: Optional[ProgressFn] = None, cancel: Optional[Callable[[], bool]] = None,
                  aggregation: Optional[str] = None, match_training_jpeg: bool = True) -> AnalysisResult:
    t0 = time.perf_counter()
    path = Path(path)
    method = aggregation or settings.video_aggregation
    report = progress or (lambda stage, frac: None)
    analysis_id = analysis_id or uuid.uuid4().hex

    meta = probe_video(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < 400_000_000 else None
    warnings: List[str] = []
    if meta["fps_guessed"]:
        warnings.append("Frame rate metadata missing/invalid; assumed 25 fps (timestamps are approximate).")
    if meta["frame_count"] <= 0:                      # unknown length: count by a fast pass
        cap = cv2.VideoCapture(str(path)); n = 0
        while cap.grab():
            n += 1
        cap.release()
        meta["frame_count"], meta["duration_s"] = n, n / meta["fps"]
        warnings.append("Container did not report a frame count; it was counted by decoding.")
    if min(meta["width"], meta["height"]) < 240:
        warnings.append(f"Low-resolution video ({meta['width']}x{meta['height']}); detection reliability is reduced.")

    idx, w2 = plan_sampling(meta["frame_count"], meta["fps"], settings.video_sample_fps, settings.video_max_frames, settings.video_max_duration_s)
    warnings += w2
    wanted = set(idx)
    last = idx[-1]

    tracker = IoUTracker()
    obs_frames: List[Dict] = []          # one dict per sampled frame with >=1 face
    pending: List[Tuple[int, int, DetectedFace]] = []        # (obs_frame_idx, face_idx, face)
    corrupt = 0
    frames_sampled = 0
    sample_pos = 0
    evidence: List[Tuple[float, int, Dict]] = []             # min-heap of (score, tiebreak, payload)
    tie = 0
    ts_prev = -1.0

    def flush():
        nonlocal tie
        if not pending:
            return
        probs = score_crops(bundle, [d.crop for _, _, d in pending], settings.video_batch_size, match_training_jpeg)
        for (fi, k, _), p in zip(pending, probs):
            obs_frames[fi]["faces"][k]["p"] = float(p)
        pending.clear()

    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while i <= last:
            if cancel and cancel():
                raise JobCancelled("Analysis cancelled.")
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts = msec / 1000.0 if (msec and msec > 0 and msec / 1000.0 >= ts_prev) else i / meta["fps"]
                ts_prev = ts
                sample_pos += 1
                if not ok or frame is None or frame.size == 0:
                    corrupt += 1
                else:
                    frames_sampled += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    dets = detector.detect(rgb)
                    if dets:
                        tids = tracker.update(len(obs_frames), [d.box for d in dets])
                        small, sc = _shrink(rgb)
                        rec = dict(idx=i, ts=ts, faces=[dict(track=t, box=d.box, conf=d.confidence, p=None, det=d) for t, d in zip(tids, dets)],
                                   img=small, scale=sc)
                        obs_frames.append(rec)
                        for k, d in enumerate(dets):
                            pending.append((len(obs_frames) - 1, k, d))
                        if len(pending) >= settings.video_batch_size:
                            flush()
                report("analysing", 0.9 * sample_pos / max(1, len(idx)))
            i += 1
    finally:
        cap.release()
    flush()

    # ---------------- per-frame / per-track results
    frames_out: List[FrameResult] = []
    by_track: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)       # tid -> (ts, p, weight)
    series_t, series_p = [], []
    for rec in obs_frames:
        faces = [FrameFace(track_id=f["track"], box=[round(v, 1) for v in f["box"]], detection_confidence=round(f["conf"], 4),
                           fake_probability=f["p"]) for f in rec["faces"]]
        fp = max(f["p"] for f in rec["faces"])
        frames_out.append(FrameResult(frame_index=rec["idx"], timestamp_s=round(rec["ts"], 3), faces=faces,
                                      frame_fake_probability=fp, suspicious=fp > DECISION_THRESHOLD))
        series_t.append(rec["ts"]); series_p.append(fp)
        for f in rec["faces"]:
            by_track[f["track"]].append((rec["ts"], f["p"], f["conf"] * f["det"].min_side_px))

    res = AnalysisResult(analysis_id=analysis_id, media_type="video", filename=filename, file_size_bytes=path.stat().st_size,
                         file_sha256=sha, model=bundle.info, image_width=meta["width"], image_height=meta["height"], warnings=warnings)
    res.faces_detected = int(sum(len(r["faces"]) for r in obs_frames))
    step_s = (idx[-1] / meta["fps"]) / max(1, len(idx) - 1) if len(idx) > 1 else 0.0

    tracks: List[TrackSummary] = []
    for tid, rows in sorted(by_track.items()):
        ps = [p for _, p, _ in rows]
        tracks.append(TrackSummary(track_id=tid, frames=len(rows), first_timestamp_s=round(rows[0][0], 3), last_timestamp_s=round(rows[-1][0], 3),
                                   median_fake_probability=float(np.median(ps)), mean_fake_probability=float(np.mean(ps)),
                                   suspicious_fraction=float(np.mean(np.asarray(ps) > DECISION_THRESHOLD))))
    score, mode, track_scores = video_score_from_frames(frames_out, method, settings.video_min_face_frames)

    video = VideoSummary(duration_s=round(meta["duration_s"], 3), fps=round(meta["fps"], 3), width=meta["width"], height=meta["height"],
                         total_frames_reported=meta["frame_count"], frames_sampled=frames_sampled, frames_with_faces=len(obs_frames),
                         corrupt_frames_skipped=corrupt, sampling_fps=settings.video_sample_fps, aggregation_method=method,
                         aggregation_validated=False, frames=frames_out, tracks=tracks, suspicious_segments=[], evidence_frames=[])
    res.video = video
    if corrupt:
        res.warnings.append(f"{corrupt} sampled frame(s) could not be decoded and were skipped.")
    if frames_sampled == 0:
        raise InvalidMediaError("No frames could be decoded from this video.", "unreadable_video")

    if len(obs_frames) < settings.video_min_face_frames:
        res.classification = Classification.INCONCLUSIVE
        res.warnings.append(f"Only {len(obs_frames)} of {frames_sampled} sampled frames contained a usable face (minimum {settings.video_min_face_frames}).")
        res.explanation = ["There was not enough facial evidence to analyse this video.", "INCONCLUSIVE does not mean real or fake."]
        return _finish(res, t0)

    if mode == "tracks":
        res.aggregation_note = f"Video score = maximum over {len(track_scores)} face track(s) of the {method} of that track's frame scores."
    else:
        res.aggregation_note = f"No face track was long enough; all face observations were pooled ({method})."
        res.warnings.append("Faces could not be associated across frames into long tracks; pooled scoring was used.")
    if method == "suspicious_fraction":
        res.aggregation_note += " For this method the score is the share of suspicious frames, not a probability."
    res.fake_probability, res.real_probability = score, 1.0 - score
    res.classification = classify(score, settings.uncertainty_margin)
    res.decisiveness, res.decisiveness_label = decisiveness(score)
    video.temporal = temporal_stats(series_p)
    video.suspicious_segments = suspicious_segments(series_t, series_p, step_s)
    if len(by_track) > 1:
        res.warnings.append(f"{len(by_track)} face tracks found (multiple people or tracking breaks); the highest-scoring eligible track decides.")

    # ---------------- evidence frames (+ Grad-CAM) -- top-K frames by frame score
    report("rendering evidence", 0.92)
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        order = sorted(range(len(obs_frames)), key=lambda j: -series_p[j])[: settings.video_evidence_frames]
        crops, meta_ev = [], []
        for rank, j in enumerate(order):
            rec = obs_frames[j]
            top = max(rec["faces"], key=lambda f: f["p"])
            crops.append(top["det"].crop)
            faces_draw = [dict(box=[v * rec["scale"] for v in f["box"]], track=f["track"], p=f["p"], label=classify(f["p"], settings.uncertainty_margin)) for f in rec["faces"]]
            meta_ev.append((rank, j, faces_draw))
        maps = explainer.heatmaps(preprocess_faces(crops, match_training_jpeg)) if (explainer and settings.enable_heatmaps and crops) else None
        for n, (rank, j, faces_draw) in enumerate(meta_ev):
            rec = obs_frames[j]
            fname = f"evidence_{rank}.jpg"
            (artifacts_dir / fname).write_bytes(encode_jpeg(_annotate_frame(rec["img"], faces_draw)))
            hname = None
            if maps is not None:
                hname = f"evidence_{rank}_heatmap.jpg"
                (artifacts_dir / hname).write_bytes(encode_jpeg(maps[n]))
            video.evidence_frames.append(EvidenceFrame(frame_index=rec["idx"], timestamp_s=round(rec["ts"], 3), fake_probability=series_p[j],
                                                       frame_artifact=fname, heatmap_artifact=hname))
        if explainer is not None and maps is None and settings.enable_heatmaps:
            res.warnings.append("Grad-CAM is unavailable in this environment (pytorch_grad_cam not installed).")

    res.explanation = _explain(res, settings)
    report("done", 1.0)
    return _finish(res, t0)


def _explain(res: AnalysisResult, s: Settings) -> List[str]:
    v = res.video
    assert v is not None
    lines = [f"{v.frames_sampled} frame(s) sampled at ~{v.sampling_fps} fps; faces found in {v.frames_with_faces}; video score {res.fake_probability:.3f} "
             f"({v.aggregation_method}; method not validated on labelled videos).",
             f"Decision rule: score > {DECISION_THRESHOLD} is FAKE, otherwise REAL; within +/-{s.uncertainty_margin} of {DECISION_THRESHOLD} is INCONCLUSIVE."]
    if v.temporal:
        lines.append(f"{v.temporal.suspicious_fraction * 100:.0f}% of analysed frames scored above {DECISION_THRESHOLD}; longest consecutive run: {v.temporal.longest_suspicious_run} frame(s).")
    if v.suspicious_segments:
        seg = ", ".join(f"{a.start_s:.1f}-{a.end_s:.1f}s" for a in v.suspicious_segments[:6])
        lines.append(f"Suspicious time ranges (model-flagged, not proof): {seg}.")
    if res.classification == Classification.FAKE:
        lines.append("Frame scores indicate patterns the model associates with fabricated faces. This is model-based evidence, not proof of manipulation.")
    elif res.classification == Classification.REAL:
        lines.append("Frame scores did not indicate patterns the model associates with fabricated faces. This does not prove authenticity.")
    else:
        lines.append("Evidence is too close to the decision boundary for a reliable verdict.")
    if v.evidence_frames and v.evidence_frames[0].heatmap_artifact:
        lines.append(CAM_NOTE)
    return lines


def _finish(res: AnalysisResult, t0: float) -> AnalysisResult:
    res.status = JobStatus.COMPLETED
    res.completed_at = utcnow()
    res.processing_time_s = round(time.perf_counter() - t0, 3)
    return res
