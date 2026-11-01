"""Labelled VIDEO-CODEC PROXY benchmark: does the *video pipeline* stay reliable when real and fake faces arrive as ordinary H.264 footage?

What it is: every clip is built from ONE held-out dataset photograph (identity-disjoint split, known label) with a slow handheld-style
zoom/drift, encoded with a real H.264 encoder (ffmpeg `h264_mf`) in one of two conditions, then analysed by the production
`analyze_video` pipeline. The label of a clip is the label of its photograph. Real and fake clips are treated identically.

What it is NOT: a deepfake-video benchmark. The clips contain no temporal manipulation artefacts, only a still face with camera-like
motion, so it measures robustness to softness / resampling / video compression (the failure mode of the investigated false positive)
and says nothing about video-level deepfake accuracy.

  python -m training.video_proxy_benchmark build --splits val test_internal test_crossgen --per-class 40 --out runs/vproxy
  python -m training.video_proxy_benchmark score --checkpoint models/verifai_v2_best.pth --clips runs/vproxy --out runs/vproxy/scores_v2.json
Requires `ffmpeg` with the h264_mf (Windows) or libx264 encoder on PATH.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

SIZE = 384            # every photograph is rendered to SIZE x SIZE so real and fake clips have the same geometry
FPS, SECONDS = 24, 3.5
CONDITIONS = {
    # name: (gaussian sigma applied to every frame before encoding, bitrate)
    "clean": (0.0, "8M"),
    "phone_soft": (0.5, "4M"),
}


def _encoder() -> str:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    for enc in ("libx264", "h264_mf", "libopenh264"):
        if f" {enc} " in out:
            return enc
    raise RuntimeError("no H.264 encoder found in ffmpeg")


def render_clip(image_path: str, out_path: Path, sigma: float, bitrate: str, encoder: str, seed: int) -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    interp = cv2.INTER_AREA if max(img.shape[:2]) > SIZE else cv2.INTER_CUBIC
    base = cv2.resize(img, (SIZE, SIZE), interpolation=interp)
    rng = random.Random(seed)
    dx, dy = rng.uniform(-6, 6), rng.uniform(-6, 6)
    n = int(FPS * SECONDS)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-",
           "-c:v", encoder, "-b:v", bitrate, "-pix_fmt", "yuv420p", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        t = i / max(n - 1, 1)
        M = cv2.getRotationMatrix2D((SIZE / 2, SIZE / 2), 0, 1.0 + 0.06 * t)
        M[:, 2] += (dx * t, dy * t)
        f = cv2.warpAffine(base, M, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT)
        if sigma > 0:
            f = cv2.GaussianBlur(f, (0, 0), sigma)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def build(a):
    root = Path(a.out); (root / "clips").mkdir(parents=True, exist_ok=True)
    enc = _encoder()
    rows = [r for r in csv.DictReader(open(a.raw_manifest, encoding="utf-8")) if r["split"] in set(a.splits)]
    rng = random.Random(a.seed)
    chosen = []
    for split in a.splits:
        for lab in ("0", "1"):
            pool = [r for r in rows if r["split"] == split and r["label"] == lab]
            chosen += rng.sample(pool, min(a.per_class, len(pool)))
    out_rows = []
    for k, r in enumerate(chosen):
        for cond, (sigma, br) in CONDITIONS.items():
            name = f"{r['split']}_{r['label']}_{k:04d}_{cond}.mp4"
            dest = root / "clips" / name
            if not dest.exists():
                render_clip(r["raw_path"], dest, sigma, br, enc, seed=k)
            out_rows.append(dict(clip=f"clips/{name}", condition=cond, split=r["split"], label=r["label"], generator=r["generator"], identity=r["identity"], source_sha256=r["sha256"]))
    with open(root / "clips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    print(f"built {len(out_rows)} clips ({len(chosen)} photographs x {len(CONDITIONS)} conditions) with {enc} in {root}")


def score(a):
    from verifai.config import Settings
    from verifai.faces import FaceDetector
    from verifai.model import load_model
    from verifai.video_pipeline import analyze_video

    from .metrics import compute_metrics

    s = Settings.from_env()
    bundle = load_model(a.checkpoint, "auto")
    det = FaceDetector(bundle.device, s)
    root = Path(a.clips)
    rows = list(csv.DictReader(open(root / "clips.csv", encoding="utf-8")))
    res = []
    for i, r in enumerate(rows):
        try:
            out = analyze_video(root / r["clip"], Path(r["clip"]).name, bundle, det, None, s)
            p, cls, nface = out.fake_probability, out.classification.value, out.video.frames_with_faces
        except Exception as exc:  # a clip that cannot be analysed counts as "no verdict"
            p, cls, nface = None, "ERROR", 0
        res.append(dict(r, fake_probability=p, classification=cls, frames_with_faces=nface))
        if i % 20 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    summary = {}
    for cond in CONDITIONS:
        for split in sorted({r["split"] for r in res}) + ["all"]:
            sel = [r for r in res if r["condition"] == cond and (split == "all" or r["split"] == split) and r["fake_probability"] is not None]
            if not sel:
                continue
            y = np.array([int(r["label"]) for r in sel]); p = np.array([r["fake_probability"] for r in sel])
            m = compute_metrics(y, p)
            m["inconclusive_rate_at_0.15_margin"] = float(np.mean([r["classification"] == "INCONCLUSIVE" for r in sel]))
            m["frames_with_faces_median"] = float(np.median([r["frames_with_faces"] for r in sel]))
            m["n_unscorable"] = sum(1 for r in res if r["condition"] == cond and (split == "all" or r["split"] == split) and r["fake_probability"] is None)
            summary[f"{cond}/{split}"] = m
    json.dump(dict(checkpoint=Path(a.checkpoint).name, checkpoint_sha256=bundle.info.checkpoint_sha256, benchmark="video-codec proxy (still-image clips, NOT a deepfake-video benchmark)",
                   aggregation=s.video_aggregation, threshold=0.5, summary=summary, clips=res), open(a.out, "w"), indent=1, default=str)
    for k, m in summary.items():
        print(f"{k:24s} n={m['n']:3d} acc={m['accuracy']:.3f} auc={m['roc_auc']} fpr={m['false_positive_rate']:.3f} fnr={m['false_negative_rate']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw-manifest", default="manifests/dff_raw_manifest.csv")
    b.add_argument("--splits", nargs="+", default=["val", "test_internal", "test_crossgen"])
    b.add_argument("--per-class", type=int, default=40)
    b.add_argument("--out", required=True)
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--clips", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
