"""E1: zero-training quality-bin diagnostic. Is image quality acting as a LABEL PROXY, and how do the streams behave across quality?

No training, no threshold tuning, no checkpoint modification: checkpoints are only run forward; the 0.5 threshold is fixed.

Quality measures, computed on the stored 224x224 face crop (the exact crop the model sees):
  sharpness = variance of the Laplacian of the grey-scale crop (higher = more fine detail)
  hf_noise  = mean absolute high-pass residual (grey minus Gaussian-blurred grey, sigma 1)
Sharpness bins are FIXED in advance on a factor-of-two scale (not derived from any result): <40, 40-80, 80-160, 160-320, >=320.

Label-proxy tests (per split):
  * per-bin real/fake counts and fake fraction; chi-square test of independence between bin and label, with Cramer's V;
  * ROC-AUC of "softness alone" (score = -sharpness, positive class = FAKE): 0.5 means quality carries no label information,
    <0.5 means fakes are SOFTER than reals, >0.5 means fakes are SHARPER; 95% bootstrap CI resampling groups (identity / video).
Model behaviour per bin (per checkpoint, for `combined`, `rgb_only` and `srm_only` scores, see stream_ablation.py):
  FPR (real in the bin called FAKE) and FNR (fake in the bin called REAL) with Wilson 95% CIs and the counts behind them, mean scores,
  within-bin ROC-AUC (when both classes have >= MIN_PER_CLASS crops), and Spearman(sharpness, score) among reals and among fakes.

  python -m training.quality_bins --manifest manifests/dff_faces_manifest.csv --dep-split train --dep-split val --dep-split test_internal \
      --model-split test_internal --model-split test_crossgen --model v2=models/verifai_v2_best.pth --out runs/e1_dff.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import roc_auc_score

BIN_EDGES = (0.0, 40.0, 80.0, 160.0, 320.0, float("inf"))
BIN_NAMES = ("<40", "40-80", "80-160", "160-320", ">=320")
MIN_PER_CLASS = 20


def laplacian_variance(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def hf_noise(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float32)
    return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0)).mean())


def crop_quality(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return laplacian_variance(g), hf_noise(g)


def bin_index(sharp: float) -> int:
    for i in range(len(BIN_NAMES)):
        if BIN_EDGES[i] <= sharp < BIN_EDGES[i + 1]:
            return i
    return len(BIN_NAMES) - 1


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _auc(y, s) -> Optional[float]:
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def grouped_bootstrap_auc(y, s, groups, n_boot: int = 1000, seed: int = 0):
    """95% CI of the ROC-AUC resampling whole groups (identities / videos), so correlated crops do not shrink the interval."""
    y, s = np.asarray(y), np.asarray(s)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by[u] for u in pick])
        a = _auc(y[ii], s[ii])
        if a is not None:
            vals.append(a)
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)) if vals else (None, None)


def label_dependence(sharp, hf, labels, groups, n_boot: int = 1000, seed: int = 0) -> Dict:
    sharp, hf, labels = np.asarray(sharp, float), np.asarray(hf, float), np.asarray(labels, int)
    bins = np.array([bin_index(x) for x in sharp])
    per_bin = []
    for i, name in enumerate(BIN_NAMES):
        m = bins == i
        nr, nf = int((m & (labels == 0)).sum()), int((m & (labels == 1)).sum())
        per_bin.append(dict(bin=name, n_real=nr, n_fake=nf, fake_fraction=round(nf / (nr + nf), 4) if nr + nf else None))
    table = np.array([[p["n_real"], p["n_fake"]] for p in per_bin])
    table = table[table.sum(1) > 0]
    chi2, p, dof, _ = chi2_contingency(table) if table.shape[0] > 1 and (table.sum(0) > 0).all() else (float("nan"), float("nan"), 0, None)
    n = int(table.sum())
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1))) if n and min(table.shape) > 1 and not math.isnan(chi2) else None
    out = dict(n=int(len(labels)), n_real=int((labels == 0).sum()), n_fake=int((labels == 1).sum()), per_bin=per_bin,
               chi2=None if math.isnan(chi2) else round(float(chi2), 2), dof=int(dof), p_value=None if math.isnan(p) else float(p), cramers_v=None if v is None else round(v, 4),
               median_sharpness=dict(real=round(float(np.median(sharp[labels == 0])), 1), fake=round(float(np.median(sharp[labels == 1])), 1)),
               median_hf_noise=dict(real=round(float(np.median(hf[labels == 0])), 3), fake=round(float(np.median(hf[labels == 1])), 3)))
    for name, q in (("softness_alone", -sharp), ("low_hf_noise_alone", -hf)):
        a = _auc(labels, q)
        lo, hi = grouped_bootstrap_auc(labels, q, groups, n_boot, seed)
        out[f"auc_fake_from_{name}"] = dict(auc=None if a is None else round(a, 4), ci95=[lo, hi])
    return out


def model_by_bin(sharp, labels, scores: np.ndarray) -> Dict:
    sharp, labels, scores = np.asarray(sharp, float), np.asarray(labels, int), np.asarray(scores, float)
    bins = np.array([bin_index(x) for x in sharp])
    pred = scores > 0.5
    rows = []
    for i, name in enumerate(BIN_NAMES):
        m = bins == i
        r, f = m & (labels == 0), m & (labels == 1)
        nr, nf = int(r.sum()), int(f.sum())
        fp, fn = int((pred & r).sum()), int((~pred & f).sum())
        within = None
        if nr >= MIN_PER_CLASS and nf >= MIN_PER_CLASS:
            within = round(float(roc_auc_score(labels[m], scores[m])), 4)
        rows.append(dict(bin=name, n_real=nr, n_fake=nf, fp=fp, fn=fn,
                         fpr=None if nr == 0 else round(fp / nr, 4), fpr_ci95=list(wilson(fp, nr)),
                         fnr=None if nf == 0 else round(fn / nf, 4), fnr_ci95=list(wilson(fn, nf)),
                         mean_score_real=None if nr == 0 else round(float(scores[r].mean()), 4), mean_score_fake=None if nf == 0 else round(float(scores[f].mean()), 4),
                         within_bin_auc=within))
    rho = {}
    for lab, nm in ((0, "real"), (1, "fake")):
        m = labels == lab
        rho[nm] = round(float(spearmanr(sharp[m], scores[m])[0]), 4) if m.sum() > 2 else None
    return dict(per_bin=rows, spearman_sharpness_vs_score=rho)


def run(models: Dict[str, str], manifest: str, dep_splits: Sequence[str], model_splits: Sequence[str], ref_manifest: str = "", ref_split: str = "train",
        n_ref: int = 150, n_boot: int = 1000, device: str = "auto", group_key: str = "") -> Dict:
    import torch
    from verifai.model import load_model
    from .dataset import FaceCropDataset, read_manifest
    from .stream_ablation import collect, head, reference_means

    def groups_of(rows):
        k = group_key or ("video" if any(r.get("video") for r in rows) else "identity")
        return [r.get(k) or r.get("identity") for r in rows]

    result: Dict = dict(bins=dict(edges=[e if e != float("inf") else None for e in BIN_EDGES], names=list(BIN_NAMES)), dependence={}, models={})
    cache: Dict[str, tuple] = {}
    for sp in sorted(set(dep_splits) | set(model_splits)):
        rows = read_manifest(manifest, [sp])
        q = np.array([crop_quality(r["crop_path"]) for r in rows])
        cache[sp] = (rows, q[:, 0], q[:, 1], np.array([int(r["label"]) for r in rows]))
    for sp in dep_splits:
        rows, sh, hf, y = cache[sp]
        result["dependence"][sp] = label_dependence(sh, hf, y, groups_of(rows), n_boot)
    for name, ck in models.items():
        b = load_model(ck, device)
        model = b.model.eval()
        r1, r2 = reference_means(model, ref_manifest or manifest, ref_split, n_ref, 0)
        r1, r2 = r1.to(b.device), r2.to(b.device)
        entry = dict(checkpoint=b.info.checkpoint_file, sha256=b.info.checkpoint_sha256, splits={})
        for sp in model_splits:
            rows, sh, hf, y = cache[sp]
            ds = FaceCropDataset(manifest, [sp], train=False)
            f1, f2 = collect(model, ds)
            f1, f2 = f1.to(b.device), f2.to(b.device)
            sc = dict(combined=head(model, f1, f2), rgb_only=head(model, f1, r2.expand_as(f2)), srm_only=head(model, r1.expand_as(f1), f2))
            entry["splits"][sp] = {v: model_by_bin(sh, y, s) for v, s in sc.items()}
        result["models"][name] = entry
        del b, model
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref-manifest", default="")
    ap.add_argument("--dep-split", action="append", default=[])
    ap.add_argument("--model-split", action="append", default=[])
    ap.add_argument("--model", action="append", default=[], help="name=path/to/checkpoint.pth (repeatable)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    models = dict(m.split("=", 1) for m in a.model)
    res = run(models, a.manifest, a.dep_split, a.model_split, a.ref_manifest, n_boot=a.boot)
    json.dump(res, open(a.out, "w"), indent=1)
    for sp, d in res["dependence"].items():
        s = d["auc_fake_from_softness_alone"]
        print(f"[{sp}] n={d['n']} softness-alone AUC={s['auc']} CI={s['ci95']} chi2 p={d['p_value']} V={d['cramers_v']}")


if __name__ == "__main__":
    main()
