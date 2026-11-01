"""Celeb-DF planning logic: identity-disjoint splits, cross-split fake dropping, balanced frame budgets, video-level aggregation (no data needed)."""
import random
from pathlib import Path

import numpy as np

from training.celebdf import assign_identity_splits, frame_indices, parse, plan_videos, scan, video_level


def _videos():
    real = [Path(f"Celeb-real/id{i}_{k:04d}.mp4") for i in range(20) for k in range(3)]
    fake = [Path(f"Celeb-synthesis/id{a}_id{b}_{k:04d}.mp4") for a in range(20) for b in range(20) if a != b for k in range(2)]
    yt = [Path(f"YouTube-real/{k:05d}.mp4") for k in range(30)]
    return dict(real=real, fake=fake, youtube=yt)


def test_parse_names():
    assert parse(Path("id12_0003.mp4")) == ("real", (12,))
    assert parse(Path("id3_id7_0001.mp4")) == ("fake", (3, 7))
    assert parse(Path("00012.mp4"))[0] == "unknown"


def test_identity_splits_are_disjoint_and_cover_everyone():
    s = assign_identity_splits(range(62), seed=1)
    assert set(s) == set(range(62)) and set(s.values()) == {"train", "val", "cdf_test"}
    assert sum(v == "cdf_test" for v in s.values()) >= 1 and sum(v == "val" for v in s.values()) >= 1
    assert assign_identity_splits(range(62), seed=1) == s                       # deterministic


def test_no_identity_appears_in_two_splits_and_cross_split_fakes_are_dropped():
    plan, split_of = plan_videos(_videos(), seed=3)
    owner = {}
    for v in plan:
        if v["source"].startswith("youtube"):
            continue
        for ident in v["identity"].split("+"):
            owner.setdefault(ident, set()).add(v["split"])
    assert all(len(sp) == 1 for sp in owner.values())                          # leakage control
    for v in plan:
        if v["label"] == 1:
            assert len({split_of[int(i[2:])] for i in v["identity"].split("+")}) == 1
    n_fake_total = len(_videos()["fake"])
    assert sum(v["label"] == 1 for v in plan) < n_fake_total                    # some cross-split fakes were dropped


def test_youtube_real_is_never_in_test():
    plan, _ = plan_videos(_videos(), seed=3)
    assert all(v["split"] != "cdf_test" for v in plan if v["source"] == "youtube_real")


def test_frame_budget_balances_classes_and_test_uses_fixed_frames():
    plan, _ = plan_videos(_videos(), seed=3, frames_real=12, frames_test=16)
    for sp in ("train", "val"):
        r = sum(v["n_frames"] for v in plan if v["split"] == sp and v["label"] == 0)
        f = sum(v["n_frames"] for v in plan if v["split"] == sp and v["label"] == 1)
        assert 0.5 < f / r < 2.0, (sp, r, f)
    assert all(v["n_frames"] == 16 for v in plan if v["split"] == "cdf_test")


def test_frame_indices_are_spread_and_in_range():
    idx = frame_indices(300, 12, random.Random(0))
    assert len(idx) == 12 and idx == sorted(idx) and 0 <= idx[0] and idx[-1] < 300
    assert frame_indices(5, 12, random.Random(0)).__len__() == 5 and frame_indices(0, 12, random.Random(0)) == []


def test_video_level_median_and_labels():
    rows = [dict(video="a", label=0)] * 3 + [dict(video="b", label=1)] * 3
    out = video_level(rows, [0.1, 0.9, 0.2, 0.95, 0.9, 0.1])
    assert out["a"] == (0, 0.2) and out["b"] == (1, 0.9)
    assert video_level(rows, [0.1, 0.9, 0.2, 0.95, 0.9, 0.1], "max")["a"][1] == 0.9


def test_scan_finds_the_three_folders(tmp_path):
    for sub, name in [("Celeb-real", "id0_0000.mp4"), ("Celeb-synthesis", "id0_id1_0000.mp4"), ("YouTube-real", "00000.mp4")]:
        (tmp_path / "x" / sub).mkdir(parents=True, exist_ok=True); (tmp_path / "x" / sub / name).write_bytes(b"0")
    got = scan(str(tmp_path))
    assert [len(got[k]) for k in ("real", "fake", "youtube")] == [1, 1, 1]
