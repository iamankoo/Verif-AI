import numpy as np
import pytest

from verifai.config import DECISION_THRESHOLD
from verifai.faces import IoUTracker, iou
from verifai.schemas import Classification
from verifai.scoring import aggregate, classify, decisiveness, label_face, suspicious_segments, temporal_stats


def test_threshold_constant_is_half():
    assert DECISION_THRESHOLD == 0.5


@pytest.mark.parametrize("p,margin,expected", [
    (0.99, 0.15, Classification.FAKE), (0.01, 0.15, Classification.REAL),
    (0.55, 0.15, Classification.INCONCLUSIVE), (0.45, 0.15, Classification.INCONCLUSIVE),
    (0.651, 0.15, Classification.FAKE), (0.349, 0.15, Classification.REAL),
    (0.51, 0.0, Classification.FAKE), (0.49, 0.0, Classification.REAL),
])
def test_classify(p, margin, expected):
    assert classify(p, margin) == expected


def test_label_face_is_raw_threshold():
    assert label_face(0.5001) == Classification.FAKE and label_face(0.5) == Classification.REAL


def test_decisiveness():
    assert decisiveness(0.5)[0] == 0 and decisiveness(0.5)[1] == "low"
    assert decisiveness(1.0) == (1.0, "high")
    assert decisiveness(0.8)[1] == "moderate"


def test_aggregations():
    s = [0.1, 0.2, 0.9, 0.95, 0.15]
    assert aggregate(s, "mean") == pytest.approx(np.mean(s))
    assert aggregate(s, "median") == pytest.approx(0.2)
    assert aggregate(s, "max") == 0.95
    assert aggregate(s, "suspicious_fraction") == pytest.approx(0.4)
    assert aggregate(s, "weighted_mean", [1, 1, 10, 10, 1]) > aggregate(s, "mean")
    with pytest.raises(ValueError):
        aggregate([], "mean")
    with pytest.raises(ValueError):
        aggregate(s, "bogus")


def test_median_is_robust_to_outlier_frames():
    s = [0.02] * 9 + [0.99]
    assert aggregate(s, "median") < 0.1 < aggregate(s, "max")


def test_temporal_stats():
    t = temporal_stats([0.1, 0.9, 0.8, 0.2, 0.7])
    assert t.longest_suspicious_run == 2 and t.sign_flips == 3
    assert t.suspicious_fraction == pytest.approx(0.6)
    assert temporal_stats([0.3]).sign_flips == 0


def test_suspicious_segments_group_consecutive_samples():
    segs = suspicious_segments([0, 1, 2, 3, 4, 5], [0.1, 0.8, 0.9, 0.2, 0.7, 0.1], 1.0)
    assert [(s.start_s, s.end_s, s.frames) for s in segs] == [(1, 2, 2), (4, 4, 1)]
    assert segs[0].max_fake_probability == 0.9


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_tracker_keeps_ids_and_separates_people():
    tr = IoUTracker()
    a = tr.update(0, [[0, 0, 100, 100], [300, 0, 400, 100]])
    b = tr.update(1, [[10, 5, 110, 105], [305, 0, 405, 100]])     # both moved slightly
    assert a == b and a[0] != a[1]
    c = tr.update(2, [[600, 400, 700, 500]])                        # a new person
    assert c[0] not in a


def test_tracker_ends_track_after_gap():
    tr = IoUTracker(max_gap=1)
    first = tr.update(0, [[0, 0, 100, 100]])[0]
    later = tr.update(5, [[0, 0, 100, 100]])[0]                     # too late -> new identity
    assert later != first
