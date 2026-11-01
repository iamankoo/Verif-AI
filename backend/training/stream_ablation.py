"""RGB-only / SRM-only / combined fake-scores of a two-stream checkpoint (diagnostic; never used to select or tune anything).

The classifier head is joint, so a single stream cannot be run "alone". The standard way to ask what one stream contributes is to
NEUTRALISE the other one: replace its 1280-d feature vector by the mean feature vector over a class-balanced set of REFERENCE crops
(taken from the training split, both classes, clean), and run the head on the result.
  combined  = head(f_rgb , f_srm)
  rgb_only  = head(f_rgb , mean f_srm)      -> what the RGB stream says when the noise stream carries no information
  srm_only  = head(mean f_rgb , f_srm)      -> what the noise (SRM) stream says when the RGB stream carries no information
These are *conditional* scores, not independent classifiers: the head was trained on both streams together, so they indicate which stream
pushes the decision, and they are reported as such.

  python -m training.stream_ablation --checkpoint models/verifai_v2_best.pth --manifest manifests/dff_faces_manifest.csv \
      --split test_internal --split test_crossgen --degrade none --degrade mixed --out runs/ablation_v2.json
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from verifai.model import load_model

from .dataset import FaceCropDataset
from .metrics import compute_metrics


@torch.no_grad()
def stream_features(model, x: torch.Tensor):
    f1 = F.adaptive_avg_pool2d(model.spatial_net.extract_features(x), 1).flatten(1)
    f2 = F.adaptive_avg_pool2d(model.noise_net.extract_features(model.srm_layer(x)), 1).flatten(1)
    return f1, f2


@torch.no_grad()
def head(model, f1: torch.Tensor, f2: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(model.fc[:-1](torch.cat([f1, f2], dim=1))).squeeze(1).cpu().numpy()


def collect(model, ds: FaceCropDataset, batch_size: int = 32):
    f1s, f2s = [], []
    device = next(model.parameters()).device                     # batches must be on the model's device (GPU runs)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for x, _ in dl:
        a, b = stream_features(model, x.to(device))
        f1s.append(a); f2s.append(b)
    return torch.cat(f1s), torch.cat(f2s)


def reference_means(model, manifest: str, split: str, n_per_class: int, seed: int):
    ds = FaceCropDataset(manifest, [split], train=False)
    rng = random.Random(seed)
    by = {0: [], 1: []}
    for i, r in enumerate(ds.rows):
        by[int(r["label"])].append(i)
    keep = sorted(rng.sample(by[0], min(n_per_class, len(by[0]))) + rng.sample(by[1], min(n_per_class, len(by[1]))))
    ds.rows = [ds.rows[i] for i in keep]
    f1, f2 = collect(model, ds)
    return f1.mean(0, keepdim=True), f2.mean(0, keepdim=True)


def summarise(y: np.ndarray, p: np.ndarray) -> Dict:
    m = compute_metrics(y, p)
    keep = ("accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc", "false_positive_rate", "false_negative_rate")
    out = {k: m[k] for k in keep}
    out["mean_score_real"], out["mean_score_fake"] = float(p[y == 0].mean()), float(p[y == 1].mean())
    return out


def run(checkpoint: str, manifest: str, splits: Sequence[str], degrades: Sequence[str], ref_split: str = "train", n_ref: int = 150,
        n_examples: int = 6, seed: int = 0, device: str = "auto", ref_manifest: str = "") -> Dict:
    """ref_manifest: manifest the neutral reference crops are taken from (default: `manifest`). Using the SAME reference set for every model keeps the comparison fair."""
    bundle = load_model(checkpoint, device)
    model = bundle.model.eval()
    r1, r2 = reference_means(model, ref_manifest or manifest, ref_split, n_ref, seed)
    r1, r2 = r1.to(bundle.device), r2.to(bundle.device)
    result: Dict = dict(checkpoint=bundle.info.checkpoint_file, checkpoint_sha256=bundle.info.checkpoint_sha256, name=bundle.info.name,
                        reference=f"mean stream features of {2 * n_ref} class-balanced clean '{ref_split}' crops", results={})
    rng = random.Random(seed)
    for split in splits:
        for deg in degrades:
            ds = FaceCropDataset(manifest, [split], train=False, degrade="" if deg == "none" else deg)
            f1, f2 = collect(model, ds)
            f1, f2 = f1.to(bundle.device), f2.to(bundle.device)
            comb, rgb, srm = head(model, f1, f2), head(model, f1, r2.expand_as(f2)), head(model, r1.expand_as(f1), f2)
            y = np.array(ds.labels())
            block = {name: summarise(y, p) for name, p in (("combined", comb), ("rgb_only", rgb), ("srm_only", srm))}
            ex = []
            for lab in (0, 1):
                idx = [i for i in range(len(y)) if y[i] == lab]
                for i in rng.sample(idx, min(n_examples, len(idx))):
                    r = ds.rows[i]
                    ex.append(dict(label="fake" if lab else "real", generator=r.get("generator", ""), video=r.get("video", ""), combined=round(float(comb[i]), 3),
                                   rgb_only=round(float(rgb[i]), 3), srm_only=round(float(srm[i]), 3)))
            result["results"][f"{split}/{deg}"] = dict(n=int(len(y)), variants=block, examples=ex)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="manifests/dff_faces_manifest.csv")
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--degrade", action="append", default=None, help="'none', 'mixed' or a name from training/degrade.py (repeatable)")
    ap.add_argument("--ref-split", default="train")
    ap.add_argument("--ref-manifest", default="", help="manifest for the neutral reference crops (default: --manifest)")
    ap.add_argument("--n-ref", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run(a.checkpoint, a.manifest, a.split, a.degrade or ["none"], a.ref_split, a.n_ref, ref_manifest=a.ref_manifest)
    json.dump(res, open(a.out, "w"), indent=1)
    for k, v in res["results"].items():
        print(k, "n=", v["n"])
        for name, m in v["variants"].items():
            print(f"   {name:9s} auc={m['roc_auc']:.3f} fpr={m['false_positive_rate']:.3f} fnr={m['false_negative_rate']:.3f} mean(real)={m['mean_score_real']:.3f} mean(fake)={m['mean_score_fake']:.3f}")


if __name__ == "__main__":
    main()
