"""Video-dataset tooling (manifest building, leakage-safe splitting, video-level evaluation).

NOTE: no labelled video dataset was available when this project was delivered (FaceForensics++, Celeb-DF and
DFDC require signed access agreements). These tools are implemented and unit-tested on synthetic inputs, but
have NOT been run on real video benchmarks. See docs/DATASET_DOCUMENTATION.md.

Input CSV for `build_manifest`:  video_path,label,video_id,identity,dataset
  label    : 0 real / 1 fake        video_id : unique source video       identity : person (optional)
Splitting groups by identity when given (so a person never appears in two splits), otherwise by video_id.
If fakes are generated from real source videos, put source and derived fakes under the SAME identity (or give
both the same `identity` value) so that they cannot straddle splits.

CLI (from backend/):
  python -m training.video_tools manifest --csv videos.csv --out manifests/videos_manifest.csv --seed 42
  python -m training.video_tools evaluate --manifest manifests/videos_manifest.csv --checkpoint models/verifai_v2_best.pth \
         --select-on val --report-on test --out runs/video_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from verifai.scoring import AGGREGATION_METHODS
from verifai.schemas import FrameResult
from verifai.video_pipeline import video_score_from_frames

from .data_validation import assign_group_splits, check_group_disjoint
from .metrics import compute_metrics


def build_manifest(rows: Sequence[Dict], ratios: Optional[Dict[str, float]] = None, seed: int = 42) -> List[Dict]:
    ratios = ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
    required = {"video_path", "label", "video_id"}
    for r in rows:
        if not required <= set(r):
            raise ValueError(f"row missing columns {required - set(r)}: {r}")
        if str(r["label"]) not in ("0", "1"):
            raise ValueError(f"label must be 0/1: {r}")
    ids = [r["video_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate video_id values")
    groups = [(r.get("identity") or r["video_id"]) for r in rows]
    mapping = assign_group_splits(groups, ratios, seed)
    out = []
    for r, g in zip(rows, groups):
        out.append({**r, "group": g, "split": mapping[g]})
    bad = check_group_disjoint(out, "group")
    if bad:
        raise AssertionError(f"leakage: {bad}")
    return out


def evaluate_methods(per_video: Sequence[Dict], methods: Sequence[str] = AGGREGATION_METHODS, min_track_frames: int = 3) -> Dict[str, Dict]:
    """per_video: [{'label': int, 'frames': [FrameResult...]}]. Returns metrics per aggregation method
    (videos without enough face evidence are counted separately, never silently dropped)."""
    out = {}
    for m in methods:
        ys, ps, skipped = [], [], 0
        for v in per_video:
            score, mode, _ = video_score_from_frames(v["frames"], m, min_track_frames)
            if score is None or len(v["frames"]) < min_track_frames:
                skipped += 1
                continue
            ys.append(v["label"]); ps.append(score)
        out[m] = {"n_videos": len(ys), "no_face_evidence": skipped,
                  **(compute_metrics(ys, ps) if len(set(ys)) == 2 else {"roc_auc": None, "accuracy": None})}
    return out


def select_aggregation(val_metrics: Dict[str, Dict]) -> str:
    """Pick the method with the best VALIDATION ROC-AUC (ties: higher accuracy, then the more conservative order
    median > mean > weighted_mean > suspicious_fraction > max). Never looks at test data."""
    pref = ["median", "mean", "weighted_mean", "suspicious_fraction", "max"]
    scored = [(m, v.get("roc_auc") if v.get("roc_auc") is not None else -1, v.get("accuracy") or -1) for m, v in val_metrics.items()]
    scored.sort(key=lambda t: (-t[1], -t[2], pref.index(t[0])))
    return scored[0][0]


def _cmd_manifest(a):
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    out = build_manifest(rows, seed=a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    from collections import Counter
    print(Counter((r["split"], r["label"]) for r in out))


def _cmd_evaluate(a):
    from verifai.config import Settings
    from verifai.engine import Engine
    from verifai.model import load_model
    from verifai.video_pipeline import analyze_video
    s = Settings(device="auto")
    eng = Engine(load_model(a.checkpoint, "auto"), s)
    rows = [r for r in csv.DictReader(open(a.manifest, encoding="utf-8")) if r["split"] in (a.select_on, a.report_on)]
    cache: Dict[str, List[FrameResult]] = {}
    for r in rows:
        res = analyze_video(Path(r["video_path"]), os.path.basename(r["video_path"]), eng.bundle, eng.detector, None, s, artifacts_dir=None)
        cache[r["video_id"]] = res.video.frames if res.video else []
    by = lambda split: [{"label": int(r["label"]), "frames": cache[r["video_id"]]} for r in rows if r["split"] == split]   # noqa: E731
    val = evaluate_methods(by(a.select_on))
    chosen = select_aggregation(val)
    report = {"selected_on": a.select_on, "chosen_method": chosen, "validation_by_method": val,
              "report_split": a.report_on, "report_metrics_chosen_method": evaluate_methods(by(a.report_on), [chosen])[chosen]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(os.path.join(a.out, "video_metrics.json"), "w"), indent=2)
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest"); m.add_argument("--csv", required=True); m.add_argument("--out", required=True); m.add_argument("--seed", type=int, default=42)
    m.set_defaults(fn=_cmd_manifest)
    e = sub.add_parser("evaluate"); e.add_argument("--manifest", required=True); e.add_argument("--checkpoint", required=True)
    e.add_argument("--select-on", default="val"); e.add_argument("--report-on", default="test"); e.add_argument("--out", required=True)
    e.set_defaults(fn=_cmd_evaluate)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
