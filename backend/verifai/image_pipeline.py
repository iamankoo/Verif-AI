"""Dedicated image analysis pipeline (does NOT use the video pipeline).

Steps: validate -> decode (EXIF-aware) -> detect ALL faces -> crop/preprocess exactly like training ->
batched model inference -> per-face scores -> image-level aggregation (documented below) -> optional
Grad-CAM -> structured result.

Image-level aggregation: the image score is the MAXIMUM face fake-score ("any manipulated face flags the
image"). This is conservative for manipulation detection but raises the false-positive rate on group photos;
the per-face scores are always returned alongside it.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import DECISION_THRESHOLD, Settings
from .errors import InvalidMediaError
from .explain import CAM_NOTE, Explainer, encode_jpeg
from .faces import DetectedFace, FaceDetector
from .model import ModelBundle
from .preprocess import preprocess_faces
from .schemas import AnalysisResult, Classification, FaceResult, JobStatus, utcnow
from .scoring import classify, decisiveness

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COLORS = {Classification.REAL: (46, 160, 90), Classification.FAKE: (214, 61, 61), Classification.INCONCLUSIVE: (224, 160, 32)}


def score_crops(bundle: ModelBundle, crops, batch_size: int = 16, match_training_jpeg: bool = True) -> np.ndarray:
    """List of (3,224,224) 0..255 crops -> (N,) p(fake). Shared by image, video and evaluation code."""
    out: List[np.ndarray] = []
    for i in range(0, len(crops), batch_size):
        out.append(bundle.predict_batch(preprocess_faces(crops[i:i + batch_size], match_training_jpeg)))
    return np.concatenate(out) if out else np.zeros(0)


def load_image_rgb(path: Path, settings: Settings) -> Tuple[np.ndarray, List[str]]:
    warnings: List[str] = []
    try:
        with Image.open(path) as probe:
            fmt = probe.format
            w, h = probe.size
            if min(w, h) < settings.min_image_side:
                raise InvalidMediaError(f"Image is too small ({w}x{h}); minimum side is {settings.min_image_side}px.", "image_too_small")
            if w * h > settings.max_image_pixels:
                raise InvalidMediaError(f"Image has {w * h:,} pixels; limit is {settings.max_image_pixels:,}.", "image_too_large")
            probe.verify()
        with Image.open(path) as im:
            if getattr(im, "n_frames", 1) > 1:
                warnings.append("Multi-frame image: only the first frame was analysed.")
            im = ImageOps.exif_transpose(im)
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            rgb = np.array(im.convert("RGB"))
    except InvalidMediaError:
        raise
    except Exception as exc:
        raise InvalidMediaError(f"File is not a readable image ({type(exc).__name__}).", "unreadable_image") from exc
    if fmt and fmt.upper() not in {"JPEG", "PNG", "BMP", "WEBP", "TIFF", "MPO", "GIF"}:
        warnings.append(f"Uncommon image format {fmt}.")
    return rgb, warnings


def _annotate(rgb: np.ndarray, faces: List[FaceResult]) -> np.ndarray:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    thick = max(2, int(round(max(img.shape[:2]) / 400)))
    for f in faces:
        x1, y1, x2, y2 = [int(round(v)) for v in f.box]
        col = COLORS[f.label][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        txt = f"#{f.index + 1} {f.label.value} {f.fake_probability:.2f}"
        fs = max(0.5, thick / 3.0)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, thick - 1))
        cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, txt, (x1 + 3, max(th, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyze_image(path: Path, filename: str, bundle: ModelBundle, detector: FaceDetector, explainer: Optional[Explainer],
                  settings: Settings, artifacts_dir: Optional[Path] = None, analysis_id: Optional[str] = None,
                  with_heatmaps: Optional[bool] = None, match_training_jpeg: bool = True) -> AnalysisResult:
    t0 = time.perf_counter()
    analysis_id = analysis_id or uuid.uuid4().hex
    path = Path(path)
    rgb, warns = load_image_rgb(path, settings)
    h, w = rgb.shape[:2]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    res = AnalysisResult(analysis_id=analysis_id, media_type="image", filename=filename, file_size_bytes=path.stat().st_size,
                         file_sha256=sha, model=bundle.info, image_width=w, image_height=h, warnings=warns,
                         aggregation_note="Image score = maximum face fake-score.")
    detections: List[DetectedFace] = detector.detect(rgb)
    res.faces_detected = len(detections)

    if not detections:
        res.classification = Classification.INCONCLUSIVE
        res.explanation = ["No usable face was detected, so the face-based model could not analyse this image.",
                           "INCONCLUSIVE does not mean real or fake."]
        res.warnings.append("No face detected (or all faces below the minimum size/confidence).")
        return _finish(res, t0, bundle)

    probs = score_crops(bundle, [d.crop for d in detections], settings.infer_batch_size, match_training_jpeg)
    batch = preprocess_faces([d.crop for d in detections], match_training_jpeg)
    maps = explainer.heatmaps(batch) if (explainer and (settings.enable_heatmaps if with_heatmaps is None else with_heatmaps)) else None
    faces: List[FaceResult] = []
    for i, (d, p) in enumerate(zip(detections, probs)):
        p = float(p)
        fr = FaceResult(index=i, box=[round(v, 1) for v in d.box], detection_confidence=round(d.confidence, 4),
                        fake_probability=p, real_probability=1.0 - p, label=classify(p, settings.uncertainty_margin))
        if artifacts_dir is not None:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            crop_u8 = np.clip(d.crop.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
            (artifacts_dir / f"face_{i}_crop.jpg").write_bytes(encode_jpeg(crop_u8))
            fr.crop_artifact = f"face_{i}_crop.jpg"
            if maps is not None:
                (artifacts_dir / f"face_{i}_heatmap.jpg").write_bytes(encode_jpeg(maps[i]))
                fr.heatmap_artifact = f"face_{i}_heatmap.jpg"
        if d.low_resolution:
            res.warnings.append(f"Face #{i + 1} is small ({d.min_side_px:.0f}px); low-resolution faces are less reliable.")
        faces.append(fr)
    res.faces = faces
    if len(faces) > 1:
        res.warnings.append(f"{len(faces)} faces detected; each was scored independently and the image score is the highest face score.")
    if explainer is not None and maps is None and settings.enable_heatmaps:
        res.warnings.append("Grad-CAM is unavailable in this environment (pytorch_grad_cam not installed).")

    top = max(faces, key=lambda f: f.fake_probability)
    p_img = top.fake_probability
    res.fake_probability, res.real_probability = p_img, 1.0 - p_img
    res.classification = classify(p_img, settings.uncertainty_margin)
    res.decisiveness, res.decisiveness_label = decisiveness(p_img)
    if artifacts_dir is not None:
        (artifacts_dir / "annotated.jpg").write_bytes(encode_jpeg(_annotate(rgb, faces)))
        res.annotated_artifact = "annotated.jpg"
    res.explanation = _explain(res, top, settings)
    return _finish(res, t0, bundle)


def _explain(res: AnalysisResult, top: FaceResult, settings: Settings) -> List[str]:
    lines = [f"{res.faces_detected} face(s) analysed; the highest fake-score is {top.fake_probability:.3f} (face #{top.index + 1}).",
             f"Decision rule: fake-score > {DECISION_THRESHOLD} is FAKE, otherwise REAL; scores within "
             f"+/-{settings.uncertainty_margin} of {DECISION_THRESHOLD} are reported as INCONCLUSIVE."]
    if res.classification == Classification.INCONCLUSIVE:
        lines.append("The score is too close to the decision boundary to give a reliable verdict.")
    elif res.classification == Classification.FAKE:
        lines.append("The model's output indicates patterns it associates with fabricated faces. This is model-based evidence, not proof of manipulation.")
    else:
        lines.append("The model found no patterns it associates with fabricated faces. This does not prove the image is authentic.")
    if res.faces and res.faces[0].heatmap_artifact:
        lines.append(CAM_NOTE)
    return lines


def _finish(res: AnalysisResult, t0: float, bundle: ModelBundle) -> AnalysisResult:
    res.status = JobStatus.COMPLETED
    res.completed_at = utcnow()
    res.processing_time_s = round(time.perf_counter() - t0, 3)
    return res
