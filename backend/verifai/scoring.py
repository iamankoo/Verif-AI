"""Decision policy, score aggregation and temporal statistics.

* The 0.5 decision threshold is fixed (config.DECISION_THRESHOLD) and is never tuned.
* INCONCLUSIVE is an *abstention band* around 0.5 (default +/-0.15) plus data-sufficiency rules; it is a
  product-safety rule, not a re-thresholding, and is reported separately in evaluation (coverage vs accuracy).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DECISION_THRESHOLD
from .schemas import Classification, SuspiciousSegment, TemporalStats

AGGREGATION_METHODS = ("mean", "median", "weighted_mean", "suspicious_fraction", "max")


def decisiveness(p_fake: float) -> Tuple[float, str]:
    d = float(abs(p_fake - DECISION_THRESHOLD) * 2.0)
    return d, ("low" if d < 0.3 else "moderate" if d < 0.7 else "high")


def classify(p_fake: float, margin: float) -> Classification:
    if abs(p_fake - DECISION_THRESHOLD) < margin:
        return Classification.INCONCLUSIVE
    return Classification.FAKE if p_fake > DECISION_THRESHOLD else Classification.REAL


def label_face(p_fake: float) -> Classification:
    """Per-face label without the abstention band (raw threshold)."""
    return Classification.FAKE if p_fake > DECISION_THRESHOLD else Classification.REAL


def aggregate(scores: Sequence[float], method: str = "median", weights: Optional[Sequence[float]] = None) -> float:
    """Combine per-observation p(fake) values into one score in [0,1].
    suspicious_fraction returns the share of observations with p>0.5 (a vote share, not a probability)."""
    a = np.asarray(list(scores), dtype=np.float64)
    if a.size == 0:
        raise ValueError("no scores to aggregate")
    if method == "mean":
        return float(a.mean())
    if method == "median":
        return float(np.median(a))
    if method == "max":
        return float(a.max())
    if method == "weighted_mean":
        w = np.ones_like(a) if weights is None else np.asarray(list(weights), dtype=np.float64)
        return float((a * w).sum() / max(w.sum(), 1e-12))
    if method == "suspicious_fraction":
        return float((a > DECISION_THRESHOLD).mean())
    raise ValueError(f"unknown aggregation method: {method}")


def temporal_stats(scores: Sequence[float]) -> TemporalStats:
    a = np.asarray(list(scores), dtype=np.float64)
    sus = a > DECISION_THRESHOLD
    run = best = 0
    for s in sus:
        run = run + 1 if s else 0
        best = max(best, run)
    flips = int(np.sum(sus[1:] != sus[:-1])) if a.size > 1 else 0
    delta = float(np.abs(np.diff(a)).mean()) if a.size > 1 else 0.0
    return TemporalStats(score_std=float(a.std()) if a.size else 0.0, mean_abs_frame_delta=delta,
                         longest_suspicious_run=int(best), suspicious_fraction=float(sus.mean()) if a.size else 0.0,
                         sign_flips=flips)


def suspicious_segments(times: Sequence[float], scores: Sequence[float], step_s: float) -> List[SuspiciousSegment]:
    """Group consecutive suspicious samples (p>0.5) into time segments."""
    segs: List[SuspiciousSegment] = []
    cur: Optional[Dict] = None
    for t, s in zip(times, scores):
        if s > DECISION_THRESHOLD:
            if cur is None:
                cur = dict(start=t, end=t, n=1, mx=s)
            else:
                cur["end"], cur["n"], cur["mx"] = t, cur["n"] + 1, max(cur["mx"], s)
        elif cur is not None:
            segs.append(SuspiciousSegment(start_s=round(cur["start"], 3), end_s=round(cur["end"], 3), frames=cur["n"], max_fake_probability=round(cur["mx"], 6)))
            cur = None
    if cur is not None:
        segs.append(SuspiciousSegment(start_s=round(cur["start"], 3), end_s=round(cur["end"], 3), frames=cur["n"], max_fake_probability=round(cur["mx"], 6)))
    return segs
