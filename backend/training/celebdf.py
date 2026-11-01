"""Celeb-DF (v2) frames -> identity-disjoint face-crop manifest, and VIDEO-level evaluation.

Data: the Celeb-DF (v2) videos (Li et al., CVPR 2020; 590 real YouTube videos of 59 celebrities, 5,639 face-swap deepfakes, plus 300 extra real
YouTube videos). The dataset is used from a third-party Kaggle copy (`reubensuju/celeb-df-v2`, licence field "unknown") inside a PRIVATE notebook only:
nothing from it is downloaded to a laptop, committed or redistributed, and its official terms (research use only) apply. See docs.

Leakage controls
  * identities (the `idN` part of the file names) are assigned to train / val / test; a REAL video goes to its identity's split;
  * a FAKE video `idA_idB_NNNN` is kept only if BOTH identities belong to the same split (a face-swap shows A's body/scene with B's face, so
    either identity could leak); cross-split fakes are dropped;
  * the 300 extra `YouTube-real` videos have unknown / unrelated identities: they are only used in train and val (never in test);
  * the test split is evaluated at VIDEO level (median fake-score of the video's frames) and never used for any choice.
Crops come from the production face detector (`verifai.faces.FaceDetector`, MTCNN, margin 20, 224 px) and are stored as JPEG quality 95, exactly like the
DeepFakeFace training crops, so training and inference see identical preprocessing.
"""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SPLITS = ("train", "val", "cdf_test")
_REAL = re.compile(r"^id(\d+)_(\d+)\.mp4$", re.I)
_FAKE = re.compile(r"^id(\d+)_id(\d+)_(\d+)\.mp4$", re.I)


def scan(root: str) -> Dict[str, List[Path]]:
    """Find the three video folders anywhere below root (names as in the public release)."""
    out: Dict[str, List[Path]] = {"real": [], "fake": [], "youtube": []}
    for p in Path(root).rglob("*.mp4"):
        parent = p.parent.name.lower()
        if parent == "celeb-real":
            out["real"].append(p)
        elif parent == "celeb-synthesis":
            out["fake"].append(p)
        elif parent == "youtube-real":
            out["youtube"].append(p)
    for k in out:
        out[k].sort()
    return out


def parse(path: Path) -> Tuple[str, Tuple[int, ...]]:
    """-> (kind, identities). kind in real|fake|youtube|unknown."""
    name = path.name
    m = _FAKE.match(name)
    if m:
        return "fake", (int(m.group(1)), int(m.group(2)))
    m = _REAL.match(name)
    if m:
        return "real", (int(m.group(1)),)
    return "unknown", ()


def assign_identity_splits(ids: Sequence[int], seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> Dict[int, str]:
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test, n_val = max(1, round(len(ids) * test_frac)), max(1, round(len(ids) * val_frac))
    out = {}
    for k, i in enumerate(ids):
        out[i] = "cdf_test" if k < n_test else "val" if k < n_test + n_val else "train"
    return out


def plan_videos(videos: Dict[str, List[Path]], seed: int = 42, frames_real: int = 12, frames_test: int = 16) -> Tuple[List[dict], Dict[int, str]]:
    """Decide split, label and number of frames for every video. Pure function (no I/O) -> unit-tested."""
    ids = set()
    for p in videos["real"] + videos["fake"]:
        ids.update(parse(p)[1])
    split_of = assign_identity_splits(sorted(ids), seed)
    rng = random.Random(seed)
    plan: List[dict] = []
    for p in videos["real"]:
        kind, idt = parse(p)
        if kind == "real":
            plan.append(dict(path=p, split=split_of[idt[0]], label=0, source="celebdf_real", identity=f"id{idt[0]}", n_frames=0))
    for p in videos["fake"]:
        kind, idt = parse(p)
        if kind != "fake":
            continue
        sp = {split_of[i] for i in idt}
        if len(sp) != 1:        # cross-split fake: dropped (leakage control)
            continue
        plan.append(dict(path=p, split=sp.pop(), label=1, source="celebdf_fake", identity="+".join(f"id{i}" for i in idt), n_frames=0))
    yt = list(videos["youtube"])
    rng.shuffle(yt)
    for k, p in enumerate(yt):
        plan.append(dict(path=p, split="val" if k % 6 == 0 else "train", label=0, source="youtube_real", identity=f"yt:{p.stem}", n_frames=0))
    # frames per video: test = frames_test for every video (video-level evaluation); train/val = frames_real per real video and as many
    # per fake video as is needed to balance the classes inside the split.
    for sp in ("train", "val"):
        real = [v for v in plan if v["split"] == sp and v["label"] == 0]
        fake = [v for v in plan if v["split"] == sp and v["label"] == 1]
        per_real = frames_real if sp == "train" else max(4, frames_real // 2)
        for v in real:
            v["n_frames"] = per_real
        total_real = per_real * len(real)
        per_fake = max(1, round(total_real / max(1, len(fake))))
        for v in fake:
            v["n_frames"] = per_fake
    for v in plan:
        if v["split"] == "cdf_test":
            v["n_frames"] = frames_test
    for v in plan:
        v["path"] = str(v["path"])
    return plan, split_of


def frame_indices(frame_count: int, n: int, rng: random.Random) -> List[int]:
    if frame_count <= 0 or n <= 0:
        return []
    n = min(n, frame_count)
    edges = np.linspace(0, frame_count, n + 1)
    return [min(frame_count - 1, int(rng.uniform(edges[i], edges[i + 1]))) for i in range(n)]


def _read_frames(path: str, idxs: Sequence[int]):
    import cv2
    cap = cv2.VideoCapture(path)
    out = {}
    try:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok and f is not None:
                out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    return out


def extract(plan: List[dict], out_root: str, detector, workers: int = 4, seed: int = 42, log_every: int = 200) -> List[dict]:
    """Decode frames (thread pool), detect faces with the production detector (main thread), save q95 crops, return manifest rows."""
    import cv2
    rng = random.Random(seed)
    out = Path(out_root)
    jobs = []
    for v in plan:
        cap = cv2.VideoCapture(v["path"]); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
        jobs.append((v, frame_indices(n, v["n_frames"], rng)))
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [(v, idxs, ex.submit(_read_frames, v["path"], idxs)) for v, idxs in jobs]
        for k, (v, idxs, fut) in enumerate(futs):
            frames = fut.result()
            d = out / v["split"] / ("fake" if v["label"] else "real"); d.mkdir(parents=True, exist_ok=True)
            vid = Path(v["path"]).stem
            for i, rgb in frames.items():
                dets = detector.detect(rgb)
                if not dets:
                    continue
                face = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
                arr = np.clip(face.crop.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
                p = d / f"{vid}_f{i:05d}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(split=v["split"], label=v["label"], generator="real" if v["label"] == 0 else "celebdf_synthesis", identity=v["identity"],
                                 crop_path=str(p).replace("\\", "/"), source=v["source"], video=vid, frame=i))
            if log_every and k % log_every == 0:
                print(f"  extracted {k + 1}/{len(futs)} videos, {len(rows)} crops", flush=True)
    return rows


def write_manifest(rows: List[dict], path: str) -> None:
    fields = ["split", "label", "generator", "identity", "crop_path", "source", "video", "frame"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def merge_manifests(dff_manifest: str, celebdf_manifest: str, out: str) -> None:
    """Union of the DeepFakeFace and Celeb-DF crop manifests with a common column set (train/val shared, plus the DFF and Celeb-DF test splits)."""
    fields = ["split", "label", "generator", "identity", "crop_path", "source", "video", "frame"]
    rows = []
    for r in csv.DictReader(open(dff_manifest, encoding="utf-8")):
        rows.append(dict(split=r["split"], label=r["label"], generator=r["generator"], identity=f"dff:{r['identity']}", crop_path=r["crop_path"], source="dff", video="", frame=""))
    rows += list(csv.DictReader(open(celebdf_manifest, encoding="utf-8")))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def video_level(rows: Sequence[dict], scores: Sequence[float], method: str = "median") -> Dict[str, Tuple[int, float]]:
    """Frame scores -> one score per video (the production per-track rule is the median). Returns {video: (label, score)}."""
    by = defaultdict(list)
    lab = {}
    for r, s in zip(rows, scores):
        by[r["video"]].append(float(s)); lab[r["video"]] = int(r["label"])
    agg = {"median": np.median, "mean": np.mean, "max": np.max}[method]
    return {v: (lab[v], float(agg(s))) for v, s in by.items()}
