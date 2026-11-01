"""Face detection (MTCNN, all faces) and a light IoU tracker for consistent identity association in videos."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from .config import FACE_MARGIN, INPUT_SIZE, Settings


@dataclass
class DetectedFace:
    box: List[float]                  # x1,y1,x2,y2 in ORIGINAL frame pixels
    confidence: float
    crop: torch.Tensor                # (3,224,224) float 0..255, same crop function as training
    min_side_px: float
    low_resolution: bool = False


class FaceDetector:
    """MTCNN with keep_all=True. For one face the crop is tensor-identical to the original single-face
    pipeline (same detector, margin 20, 224x224)."""

    def __init__(self, device: torch.device, settings: Settings):
        self.s = settings
        self._lock = threading.Lock()          # MTCNN is not thread-safe
        self.mtcnn = MTCNN(image_size=INPUT_SIZE, margin=FACE_MARGIN, keep_all=True, post_process=False,
                           select_largest=False, device=device)

    def detect(self, rgb: np.ndarray) -> List[DetectedFace]:
        """rgb: (H,W,3) uint8. Detection runs on a downscaled copy; crops are taken from the original."""
        h, w = rgb.shape[:2]
        scale = min(1.0, self.s.detection_max_side / max(h, w))
        small = cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA) if scale < 1 else rgb
        try:
            with self._lock:
                boxes, probs = self.mtcnn.detect(small)
        except Exception:        # extremely small / degenerate images
            return []
        if boxes is None or len(boxes) == 0:
            return []
        boxes = np.asarray(boxes, dtype=np.float64) / scale
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        keep = []
        for i, (b, p) in enumerate(zip(boxes, probs)):
            side = min(b[2] - b[0], b[3] - b[1])
            if p >= self.s.face_min_confidence and side >= self.s.face_min_px:
                keep.append(i)
        if not keep:
            return []
        keep.sort(key=lambda i: -((boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])))
        keep = keep[: self.s.max_faces_per_image]
        sel = boxes[keep]
        with self._lock:
            crops = self.mtcnn.extract(rgb, sel, None)
        if crops.ndim == 3:
            crops = crops.unsqueeze(0)
        out = []
        for j, i in enumerate(keep):
            b = boxes[i]
            side = float(min(b[2] - b[0], b[3] - b[1]))
            out.append(DetectedFace(box=[float(v) for v in b], confidence=float(probs[i]), crop=crops[j],
                                    min_side_px=side, low_resolution=side < self.s.face_warn_px))
        return out


def iou(a: List[float], b: List[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class _Track:
    tid: int
    box: List[float]
    last_step: int


@dataclass
class IoUTracker:
    """Greedy IoU association across SAMPLED frames (heuristic; sampled frames are ~0.5 s apart so the
    threshold is deliberately low). Not a re-identification model: identities can swap when people cross."""
    iou_threshold: float = 0.2
    max_gap: int = 2
    tracks: List[_Track] = field(default_factory=list)
    _next: int = 0

    def update(self, step: int, boxes: List[List[float]]) -> List[int]:
        ids: List[Optional[int]] = [None] * len(boxes)
        cands: List[Tuple[float, int, int]] = []
        active = [t for t in self.tracks if step - t.last_step <= self.max_gap]
        for bi, b in enumerate(boxes):
            for t in active:
                v = iou(b, t.box)
                if v >= self.iou_threshold:
                    cands.append((v, bi, t.tid))
        used_b, used_t = set(), set()
        for v, bi, tid in sorted(cands, reverse=True):
            if bi in used_b or tid in used_t:
                continue
            ids[bi] = tid; used_b.add(bi); used_t.add(tid)
            tr = next(t for t in self.tracks if t.tid == tid)
            tr.box, tr.last_step = boxes[bi], step
        for bi, b in enumerate(boxes):
            if ids[bi] is None:
                self.tracks.append(_Track(self._next, b, step)); ids[bi] = self._next; self._next += 1
        return [int(i) for i in ids]   # type: ignore[arg-type]
