"""Evaluate a checkpoint on manifest splits. Threshold fixed at 0.5. Nothing here selects a model or a threshold.

Outputs (per split) in --out: metrics.json, predictions.csv, confusion_matrix_<split>.png, roc_<split>.png,
pr_<split>.png, examples_<split>.png.  Each result block is labelled with its evaluation type, e.g.
  internal_test  = unseen identities, generators seen in training
  cross_generator= unseen identities AND an unseen generator
  validation     = used for model selection (NOT an unbiased estimate)

Usage: python -m training.evaluate --checkpoint runs/v2/best.pth --manifest manifests/dff_faces_manifest.csv \
         --split test_internal:internal_test --split test_crossgen:cross_generator --out runs/v2/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from verifai.model import load_model

from .celebdf import video_level
from .dataset import FaceCropDataset
from .metrics import compute_metrics, coverage_metrics, grouped_bootstrap_ci


def run_model(bundle, ds, batch_size=32):
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    ps = []
    for x, _ in dl:
        ps.append(bundle.predict_batch(x))
    return np.concatenate(ps)


def plots(y, p, out_prefix, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    m = compute_metrics(y, p)
    cm = m["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    ax.imshow(arr, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] > arr.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Real", "Fake"]); ax.set_yticks([0, 1], ["Real", "Fake"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix\n{title}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_prefix.replace("{k}", "confusion_matrix"), dpi=140); plt.close(fig)
    if m["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.plot(fpr, tpr, lw=2, label=f"AUC = {m['roc_auc']:.3f}"); ax.plot([0, 1], [0, 1], "--", c="grey")
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(f"ROC\n{title}", fontsize=9); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_prefix.replace("{k}", "roc"), dpi=140); plt.close(fig)
        pr, rc, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.plot(rc, pr, lw=2, label=f"AP = {m['pr_auc']:.3f}")
        ax.set_xlabel("Recall (fake)"); ax.set_ylabel("Precision (fake)"); ax.set_ylim(0, 1.02); ax.set_title(f"Precision-recall\n{title}", fontsize=9); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out_prefix.replace("{k}", "pr"), dpi=140); plt.close(fig)


def examples(rows, p, path, k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    y = np.array([int(r["label"]) for r in rows]); pred = (p > 0.5).astype(int)
    groups = {"correct fake": np.where((y == 1) & (pred == 1))[0], "correct real": np.where((y == 0) & (pred == 0))[0],
              "missed fake (FN)": np.where((y == 1) & (pred == 0))[0], "real flagged fake (FP)": np.where((y == 0) & (pred == 1))[0]}
    fig, axes = plt.subplots(len(groups), k, figsize=(k * 1.6, len(groups) * 1.9))
    for gi, (name, idx) in enumerate(groups.items()):
        idx = idx[np.argsort(-np.abs(p[idx] - 0.5))][:k] if "correct" in name else idx[np.argsort(-np.abs(p[idx] - 0.5))][:k]
        for j in range(k):
            ax = axes[gi][j]; ax.axis("off")
            if j < len(idx):
                ax.imshow(Image.open(rows[idx[j]]["crop_path"]).convert("RGB")); ax.set_title(f"{p[idx[j]]:.2f}", fontsize=7)
            if j == 0:
                ax.text(-0.05, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle("Sample predictions (score = fake-score)", fontsize=9); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="manifests/dff_faces_manifest.csv")
    ap.add_argument("--split", action="append", required=True, help="split[:evaluation_type], repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--degrade", default="", help="'' (clean), 'mixed' or one name from training/degrade.py: applied identically to every row of every split (robustness evaluation)")
    ap.add_argument("--no-plots", action="store_true", help="skip plots and example montages (faster; used for the many robustness conditions)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bundle = load_model(a.checkpoint, "auto")
    result = {"checkpoint": bundle.info.checkpoint_file, "checkpoint_sha256": bundle.info.checkpoint_sha256, "model_name": bundle.info.name,
              "device": str(bundle.device), "manifest": a.manifest, "threshold": 0.5, "splits": {}}
    pred_rows = []
    for spec in a.split:
        split, _, etype = spec.partition(":")
        ds = FaceCropDataset(a.manifest, [split], train=False, degrade=a.degrade)
        p = run_model(bundle, ds)
        y = np.array(ds.labels())
        m = compute_metrics(y, p)
        m["evaluation_type"] = etype or split
        m["degradation"] = a.degrade or "none"
        m["bootstrap"] = grouped_bootstrap_ci(y, p, [r["identity"] for r in ds.rows], a.boot)
        m["abstention"] = coverage_metrics(y, p, a.margin)
        by_gen = defaultdict(list)
        for r, pi in zip(ds.rows, p):
            by_gen[r["generator"]].append(pi)
        m["per_generator"] = {g: {"n": len(v), "mean_score": float(np.mean(v)),
                                  ("fake_detection_rate" if g != "real" else "real_acceptance_rate"): float(np.mean(np.asarray(v) > 0.5) if g != "real" else np.mean(np.asarray(v) <= 0.5))}
                              for g, v in by_gen.items()}
        m["score_stats"] = {"mean_score_real": float(p[y == 0].mean()), "mean_score_fake": float(p[y == 1].mean())}
        if any(r.get("video") for r in ds.rows):      # video-level metrics: one score (median of its frames) per video
            vl = video_level(ds.rows, p, "median")
            yv = np.array([l for l, _ in vl.values()]); pv = np.array([s for _, s in vl.values()])
            mv = compute_metrics(yv, pv); mv["aggregation"] = "median of the video's frame scores"
            m["video_level"] = mv
            print(f"[{split}] VIDEO-LEVEL n={mv['n']} acc={mv['accuracy']:.4f} auc={mv['roc_auc']} fpr={mv['false_positive_rate']:.4f} fnr={mv['false_negative_rate']:.4f}")
        result["splits"][split] = m
        if not a.no_plots:
            plots(y, p, os.path.join(a.out, f"{{k}}_{split}.png"), f"{bundle.info.name} | {split} ({etype or split})")
            examples(ds.rows, p, os.path.join(a.out, f"examples_{split}.png"))
        for r, pi in zip(ds.rows, p):
            pred_rows.append(dict(split=split, crop_path=r["crop_path"], label=r["label"], generator=r["generator"], identity=r["identity"], video=r.get("video", ""), fake_score=f"{pi:.6f}"))
        print(f"[{split}] n={m['n']} acc={m['accuracy']:.4f} auc={m['roc_auc']} fpr={m['false_positive_rate']:.4f} fnr={m['false_negative_rate']:.4f}")
    json.dump(result, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    with open(os.path.join(a.out, "predictions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0])); w.writeheader(); w.writerows(pred_rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
