"""Evaluation metrics. The decision threshold is a fixed argument (default 0.5) — never fitted on test data."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true: Sequence[int], p_fake: Sequence[float], threshold: float = 0.5) -> Dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_fake, dtype=np.float64)
    pred = (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    return {
        "n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "threshold": threshold,
        "accuracy": float((pred == y).mean()),
        "precision_fake": float(precision_score(y, pred, zero_division=0)),
        "recall_fake": float(recall_score(y, pred, zero_division=0)),
        "f1_fake": float(f1_score(y, pred, zero_division=0)),
        "precision_real": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_real": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,     # real flagged as fake
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else None,     # fake passed as real
    }


def grouped_bootstrap_ci(y: Sequence[int], p: Sequence[float], groups: Sequence[str], n_boot: int = 1000, seed: int = 0,
                         threshold: float = 0.5) -> Dict:
    """95% bootstrap CI for accuracy and ROC-AUC, resampling whole groups (identities / videos) so that
    correlated samples are not treated as independent."""
    y, p, g = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64), np.asarray(groups)
    ug = np.unique(g)
    idx = {k: np.nonzero(g == k)[0] for k in ug}
    rng = np.random.default_rng(seed)
    accs, aucs = [], []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        ii = np.concatenate([idx[k] for k in pick])
        yy, pp = y[ii], p[ii]
        accs.append(((pp > threshold).astype(int) == yy).mean())
        if len(set(yy.tolist())) == 2:
            aucs.append(roc_auc_score(yy, pp))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a else None   # noqa: E731
    return {"accuracy_95ci": q(accs), "roc_auc_95ci": q(aucs), "n_groups": int(len(ug)), "n_boot": n_boot}


def coverage_metrics(y: Sequence[int], p: Sequence[float], margin: float, threshold: float = 0.5) -> Dict:
    """Effect of the INCONCLUSIVE abstention band: coverage and accuracy on the answered subset."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=np.float64)
    answered = np.abs(p - threshold) >= margin
    n = int(answered.sum())
    return {"margin": margin, "coverage": float(answered.mean()), "answered": n,
            "accuracy_on_answered": float(((p[answered] > threshold).astype(int) == y[answered]).mean()) if n else None}
