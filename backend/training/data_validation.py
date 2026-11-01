"""Dataset integrity checks: group-disjoint splits, exact/near duplicates, class balance, shortcut diagnostics.

Every check returns plain data so that it can be written to the dataset report and unit-tested.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


# ------------------------------------------------------------------------------------ group splits
def assign_group_splits(groups: Sequence[str], ratios: Dict[str, float], seed: int = 42) -> Dict[str, str]:
    """Map every distinct group (source video / identity) to exactly one split.
    All items of a group therefore land in the same split => no frame/identity can leak across splits."""
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1")
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    out, start = {}, 0
    names = list(ratios)
    for k, name in enumerate(names):
        n = len(uniq) - start if k == len(names) - 1 else int(round(len(uniq) * ratios[name]))
        for g in uniq[start:start + n]:
            out[g] = name
        start += n
    return out


def check_group_disjoint(rows: Sequence[dict], group_key: str = "identity", split_key: str = "split") -> List[str]:
    """Return a list of violations (empty = OK): groups that appear in more than one split."""
    seen: Dict[str, set] = defaultdict(set)
    for r in rows:
        seen[r[group_key]].add(r[split_key])
    return [f"{group_key}={g} appears in {sorted(s)}" for g, s in seen.items() if len(s) > 1]


# ------------------------------------------------------------------------------------ duplicates
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash (robust to re-encoding/resizing)."""
    g = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_duplicates(items: Sequence[dict], max_hamming: int = 4) -> Dict[str, List[Tuple[int, int, int]]]:
    """items need 'sha256', 'dhash' (int), 'split', 'label'.
    Returns {'exact': [(i, j, 0)], 'near': [(i, j, d)]} for pairs of items in DIFFERENT splits or the same
    class with identical content. Near duplicates are searched within the same label only (a real photo and
    its generated counterpart are intentionally similar)."""
    exact, near = [], []
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_sha[it["sha256"]].append(i)
    for idxs in by_sha.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                exact.append((idxs[a], idxs[b], 0))
    by_label: Dict[int, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_label[it["label"]].append(i)
    for idxs in by_label.values():
        hs = np.array([items[i]["dhash"] for i in idxs], dtype=np.uint64)
        for a in range(len(idxs)):
            x = hs[a] ^ hs[a + 1:]
            d = np.array([bin(int(v)).count("1") for v in x])
            for off in np.nonzero(d <= max_hamming)[0]:
                b = a + 1 + int(off)
                if items[idxs[a]]["sha256"] != items[idxs[b]]["sha256"]:
                    near.append((idxs[a], idxs[b], int(d[off])))
    return {"exact": exact, "near": near}


def cross_split_duplicate_pairs(items: Sequence[dict], dups: Dict[str, list]) -> List[Tuple[int, int, int]]:
    return [(i, j, d) for kind in dups.values() for (i, j, d) in kind if items[i]["split"] != items[j]["split"]]


# ------------------------------------------------------------------------------------ distributions
def distribution(rows: Iterable[dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        out[r["split"]][f"{'fake' if int(r['label']) else 'real'}"] += 1
        if int(r["label"]):
            out[r["split"]][f"gen:{r.get('generator', '?')}"] += 1
    return {k: dict(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------ shortcuts
def auc_of_feature(values: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC of a single scalar feature (e.g. file size) as a label predictor, via rank statistics.
    ~0.5 => no shortcut; far from 0.5 => the feature alone separates the classes (a leakage/shortcut warning)."""
    v, y = np.asarray(values, dtype=np.float64), np.asarray(labels, dtype=int)
    pos, neg = v[y == 1], v[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    avg = (sums / counts)[inv]
    return float((avg[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
