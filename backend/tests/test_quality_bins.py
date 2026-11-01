"""E1 quality-bin diagnostic: bins, label-proxy detection, per-bin model metrics, end-to-end run on a tiny synthetic manifest (CPU)."""
import numpy as np
import pytest

from test_training_smoke import make_manifest
from training.quality_bins import (BIN_NAMES, bin_index, crop_quality, grouped_bootstrap_auc, label_dependence, model_by_bin, run, wilson)


def test_bin_edges_are_fixed_and_cover_everything():
    assert BIN_NAMES == ("<40", "40-80", "80-160", "160-320", ">=320")
    assert [bin_index(x) for x in (0, 39.9, 40, 79.9, 80, 159.9, 160, 319.9, 320, 1e9)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_wilson_interval_basic():
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert wilson(0, 0) == (None, None)
    assert wilson(0, 20)[0] == 0.0


def test_label_dependence_detects_a_quality_proxy():
    rng = np.random.default_rng(0)
    n = 400
    y = np.array([0] * n + [1] * n)
    sharp = np.concatenate([rng.uniform(150, 600, n), rng.uniform(10, 160, n)])        # fakes are much softer: quality IS a label proxy
    hf = np.concatenate([rng.uniform(2, 5, n), rng.uniform(0.5, 2, n)])
    groups = np.arange(2 * n) // 4
    d = label_dependence(sharp, hf, y, groups, n_boot=200)
    a = d["auc_fake_from_softness_alone"]
    assert a["auc"] > 0.9 and a["ci95"][0] > 0.85
    assert d["p_value"] < 1e-6 and d["cramers_v"] > 0.5
    assert sum(b["n_real"] + b["n_fake"] for b in d["per_bin"]) == 2 * n


def test_label_dependence_reports_no_proxy_for_independent_quality():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, n)
    sharp = rng.uniform(10, 600, n)
    d = label_dependence(sharp, rng.uniform(0.5, 5, n), y, np.arange(n), n_boot=200)
    a = d["auc_fake_from_softness_alone"]
    assert 0.42 < a["auc"] < 0.58 and a["ci95"][0] < 0.5 < a["ci95"][1]


def test_grouped_bootstrap_interval_is_wider_than_naive_when_groups_are_large():
    rng = np.random.default_rng(2)
    y = np.repeat(rng.integers(0, 2, 40), 20)            # 40 groups of 20 identical-label crops
    offset = np.repeat(rng.normal(0, 1.0, 40), 20)       # crops of one identity share an offset (correlated within group)
    s = y + offset + rng.normal(0, 0.3, len(y))
    lo_g, hi_g = grouped_bootstrap_auc(y, s, np.repeat(np.arange(40), 20), 300)
    lo_i, hi_i = grouped_bootstrap_auc(y, s, np.arange(len(y)), 300)
    assert (hi_g - lo_g) > (hi_i - lo_i)


def test_model_by_bin_counts_and_rates():
    sharp = np.array([10, 10, 10, 10, 500, 500, 500, 500], float)
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    s = np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9])
    r = model_by_bin(sharp, y, s)
    b0, b4 = r["per_bin"][0], r["per_bin"][4]
    assert (b0["n_real"], b0["n_fake"], b0["fp"], b0["fn"]) == (2, 2, 1, 1) and b0["fpr"] == 0.5 and b0["fnr"] == 0.5
    assert (b4["fp"], b4["fn"]) == (0, 0) and b4["fpr"] == 0.0
    assert b0["within_bin_auc"] is None                  # fewer than MIN_PER_CLASS per class
    assert sum(b["n_real"] + b["n_fake"] for b in r["per_bin"]) == len(y)


def test_crop_quality_orders_sharp_above_blurred(tmp_path):
    import cv2
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    p1, p2 = tmp_path / "sharp.png", tmp_path / "soft.png"
    cv2.imwrite(str(p1), img); cv2.imwrite(str(p2), cv2.GaussianBlur(img, (0, 0), 3))
    (s1, h1), (s2, h2) = crop_quality(str(p1)), crop_quality(str(p2))
    assert s1 > s2 and h1 > h2


def test_end_to_end_on_tiny_manifest(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    res = run({"rand": str(random_checkpoint)}, str(m), ["train", "val"], ["val"], ref_split="train", n_ref=4, n_boot=50, device="cpu")
    assert set(res["dependence"]) == {"train", "val"} and set(res["models"]["rand"]["splits"]) == {"val"}
    v = res["models"]["rand"]["splits"]["val"]
    assert set(v) == {"combined", "rgb_only", "srm_only"}
    assert sum(b["n_real"] + b["n_fake"] for b in v["combined"]["per_bin"]) == 8
