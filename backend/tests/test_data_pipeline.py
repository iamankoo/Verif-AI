import numpy as np
import pytest
from PIL import Image

from training.data_validation import (assign_group_splits, auc_of_feature, check_group_disjoint, cross_split_duplicate_pairs,
                                      dhash, find_duplicates, hamming, sha256_bytes)
from training.metrics import compute_metrics, coverage_metrics, grouped_bootstrap_ci


def test_group_splits_never_cross():
    groups = [f"vid{i // 10}" for i in range(500)]               # 50 source videos, 10 frames each
    mapping = assign_group_splits(groups, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)
    rows = [dict(identity=g, split=mapping[g]) for g in groups]
    assert check_group_disjoint(rows) == []
    assert set(mapping.values()) == {"train", "val", "test"}
    assert sum(v == "train" for v in mapping.values()) == 30


def test_group_split_is_reproducible_and_seed_dependent():
    g = [f"g{i}" for i in range(40)]
    r = {"train": 0.5, "val": 0.25, "test": 0.25}
    assert assign_group_splits(g, r, 7) == assign_group_splits(g, r, 7)
    assert assign_group_splits(g, r, 7) != assign_group_splits(g, r, 8)
    with pytest.raises(ValueError):
        assign_group_splits(g, {"a": 0.5, "b": 0.4})


def test_leakage_detector_flags_shared_identity():
    rows = [dict(identity="a", split="train"), dict(identity="a", split="test"), dict(identity="b", split="val")]
    assert len(check_group_disjoint(rows)) == 1


def _gradient(seed, size=64):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, size)
    base = np.outer(np.sin(x / (20 + seed)) * 60 + 128, np.ones(size)) + rng.normal(0, 5, (size, size))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")


def test_dhash_detects_near_duplicates_but_not_unrelated():
    a = _gradient(1)
    b = a.resize((48, 48)).resize((64, 64))                       # resized copy
    c = Image.fromarray(np.random.default_rng(9).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    near, far = hamming(dhash(a), dhash(b)), hamming(dhash(a), dhash(c))
    assert near <= 10 and far >= 20 and far > 2 * near          # near-duplicates are much closer than unrelated images


def test_duplicate_search_and_cross_split_filter():
    a, b = _gradient(1), _gradient(1).resize((60, 60)).resize((64, 64))
    c = Image.fromarray(np.random.default_rng(3).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    items = [dict(sha256=sha256_bytes(x.tobytes()), dhash=dhash(x), split=s, label=0) for x, s in ((a, "train"), (b, "test"), (c, "test"))]
    d = find_duplicates(items, max_hamming=6)
    cross = cross_split_duplicate_pairs(items, d)
    assert any({i, j} == {0, 1} for i, j, _ in cross)
    assert not any(2 in (i, j) for i, j, _ in cross)


def test_exact_duplicates_found():
    items = [dict(sha256="x", dhash=1, split="train", label=1), dict(sha256="x", dhash=1, split="val", label=1)]
    assert len(find_duplicates(items)["exact"]) == 1


def test_shortcut_auc_feature():
    y = [0] * 50 + [1] * 50
    assert auc_of_feature(list(range(100)), y) == pytest.approx(1.0)
    assert auc_of_feature(list(range(100, 0, -1)), y) == pytest.approx(0.0)
    assert auc_of_feature([1.0] * 100, y) == pytest.approx(0.5)


def test_metrics_values():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    p = [0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7]
    m = compute_metrics(y, p)
    assert m["confusion_matrix"] == dict(tn=3, fp=1, fn=1, tp=3)
    assert m["accuracy"] == 0.75 and m["false_positive_rate"] == 0.25 and m["false_negative_rate"] == 0.25
    assert m["precision_fake"] == 0.75 and m["recall_fake"] == 0.75
    assert 0.8 < m["roc_auc"] <= 1.0 and m["threshold"] == 0.5


def test_metrics_threshold_is_fixed_argument_not_fitted():
    y, p = [0, 1], [0.5, 0.51]
    assert compute_metrics(y, p)["accuracy"] == 1.0
    assert compute_metrics(y, p, threshold=0.6)["accuracy"] == 0.5


def test_bootstrap_and_coverage():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 200), 0, 1)
    ci = grouped_bootstrap_ci(y, p, [f"g{i // 4}" for i in range(200)], n_boot=200)
    assert ci["accuracy_95ci"][0] <= compute_metrics(y, p)["accuracy"] <= ci["accuracy_95ci"][1] + 0.05
    cov = coverage_metrics(y, p, 0.15)
    assert 0 < cov["coverage"] <= 1 and cov["accuracy_on_answered"] >= compute_metrics(y, p)["accuracy"] - 0.05
