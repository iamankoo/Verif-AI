import numpy as np
import pytest

from training.data_validation import check_group_disjoint
from training.video_tools import build_manifest, evaluate_methods, select_aggregation
from verifai.schemas import FrameFace, FrameResult
from verifai.video_pipeline import video_score_from_frames


def frames(scores, track=0, box=(0, 0, 100, 100)):
    return [FrameResult(frame_index=i, timestamp_s=float(i), faces=[FrameFace(track_id=track, box=list(box), detection_confidence=0.99, fake_probability=p)],
                        frame_fake_probability=p, suspicious=bool(p > 0.5)) for i, p in enumerate(scores)]


def test_manifest_groups_by_identity_and_never_leaks():
    rows = []
    for ident in range(20):                                        # 20 people, each with 1 real + 2 derived fakes
        rows.append(dict(video_path=f"r{ident}.mp4", label="0", video_id=f"r{ident}", identity=f"p{ident}", dataset="x"))
        for k in range(2):
            rows.append(dict(video_path=f"f{ident}_{k}.mp4", label="1", video_id=f"f{ident}_{k}", identity=f"p{ident}", dataset="x"))
    out = build_manifest(rows, seed=3)
    assert check_group_disjoint(out, "group") == []
    for ident in range(20):                                        # a real video and its fakes share a split
        assert len({r["split"] for r in out if r["identity"] == f"p{ident}"}) == 1


def test_manifest_without_identity_groups_by_video():
    rows = [dict(video_path=f"{i}.mp4", label=str(i % 2), video_id=f"v{i}") for i in range(30)]
    out = build_manifest(rows)
    assert {r["group"] for r in out} == {f"v{i}" for i in range(30)}


def test_manifest_validation():
    with pytest.raises(ValueError):
        build_manifest([dict(video_path="a", label="2", video_id="a")])
    with pytest.raises(ValueError):
        build_manifest([dict(video_path="a", label="1", video_id="a"), dict(video_path="b", label="0", video_id="a")])
    with pytest.raises(ValueError):
        build_manifest([dict(video_path="a", label="1")])


def test_video_score_from_frames_tracks_and_pooling():
    two_people = frames([0.05] * 6, track=0) + frames([0.95] * 6, track=1)
    s, mode, per = video_score_from_frames(two_people, "median", 3)
    assert mode == "tracks" and s == pytest.approx(0.95) and per[0] == pytest.approx(0.05)
    s, mode, _ = video_score_from_frames(frames([0.9, 0.8]), "mean", 3)            # too short for a track -> pooled
    assert mode == "pooled" and s == pytest.approx(0.85)
    assert video_score_from_frames([], "mean", 3) == (None, "none", {})


def test_aggregation_selection_uses_only_given_data_and_reports_all_methods():
    rng = np.random.default_rng(0)
    vids = []
    for i in range(40):
        fake = i % 2
        base = 0.8 if fake else 0.15
        vids.append(dict(label=fake, frames=frames(list(np.clip(rng.normal(base, 0.1, 8), 0, 1)))))
    res = evaluate_methods(vids)
    assert set(res) == {"mean", "median", "weighted_mean", "suspicious_fraction", "max"}
    assert all(v["n_videos"] == 40 for v in res.values())
    assert select_aggregation(res) in res
    assert res["median"]["roc_auc"] > 0.95


def test_selection_prefers_best_validation_auc_then_conservative_method():
    val = {"max": dict(roc_auc=0.90, accuracy=0.8), "median": dict(roc_auc=0.95, accuracy=0.8), "mean": dict(roc_auc=0.95, accuracy=0.8)}
    assert select_aggregation(val) == "median"
    val = {"max": dict(roc_auc=0.99, accuracy=0.9), "median": dict(roc_auc=0.95, accuracy=0.9)}
    assert select_aggregation(val) == "max"


def test_videos_without_face_evidence_are_counted_not_dropped_silently():
    vids = [dict(label=1, frames=frames([0.9] * 5)), dict(label=0, frames=frames([0.1] * 5)), dict(label=1, frames=[])]
    r = evaluate_methods(vids, ["median"])["median"]
    assert r["n_videos"] == 2 and r["no_face_evidence"] == 1
