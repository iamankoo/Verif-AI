"""Image + video pipeline logic, tested with scripted faces/scores (deterministic) and random-weight models."""
import numpy as np
import pytest

from conftest import StubDetector, write_png, write_video
from verifai import image_pipeline, video_pipeline
from verifai.errors import InvalidMediaError, JobCancelled
from verifai.explain import Explainer
from verifai.schemas import Classification

BOX = [60, 40, 200, 200]


def script_scores(monkeypatch, module, values):
    """Make score_crops return the next scripted values (consumed in order)."""
    it = iter(values)

    def fake(bundle, crops, batch_size=16, match_training_jpeg=True):
        return np.array([next(it) for _ in crops])
    monkeypatch.setattr(module, "score_crops", fake)


# ------------------------------------------------------------------------------ images
def test_image_validation_errors(tmp_path, settings, random_bundle):
    bad = tmp_path / "x.png"; bad.write_bytes(b"not an image")
    with pytest.raises(InvalidMediaError) as e:
        image_pipeline.load_image_rgb(bad, settings)
    assert e.value.code == "unreadable_image"
    tiny = write_png(tmp_path / "tiny.png", 20, 20)
    with pytest.raises(InvalidMediaError) as e:
        image_pipeline.load_image_rgb(tiny, settings)
    assert e.value.code == "image_too_small"


def test_image_no_face_is_inconclusive(tmp_path, settings, random_bundle):
    img = write_png(tmp_path / "a.png")
    r = image_pipeline.analyze_image(img, "a.png", random_bundle, StubDetector([[]]), None, settings, artifacts_dir=tmp_path / "art")
    assert r.classification == Classification.INCONCLUSIVE and r.faces_detected == 0
    assert r.fake_probability is None and any("No face" in w for w in r.warnings)


def test_image_single_face_fake_and_real(tmp_path, settings, random_bundle, monkeypatch):
    img = write_png(tmp_path / "a.png")
    script_scores(monkeypatch, image_pipeline, [0.97])
    r = image_pipeline.analyze_image(img, "a.png", random_bundle, StubDetector([[BOX]]), None, settings, artifacts_dir=tmp_path / "art")
    assert r.classification == Classification.FAKE and r.faces[0].fake_probability == 0.97
    assert r.real_probability == pytest.approx(0.03) and r.decisiveness_label == "high"
    assert (tmp_path / "art" / "annotated.jpg").is_file() and r.faces[0].crop_artifact
    script_scores(monkeypatch, image_pipeline, [0.02])
    r = image_pipeline.analyze_image(img, "a.png", random_bundle, StubDetector([[BOX]]), None, settings)
    assert r.classification == Classification.REAL


def test_image_borderline_score_is_inconclusive(tmp_path, settings, random_bundle, monkeypatch):
    script_scores(monkeypatch, image_pipeline, [0.58])
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a.png", random_bundle, StubDetector([[BOX]]), None, settings)
    assert r.classification == Classification.INCONCLUSIVE and r.faces[0].fake_probability == 0.58


def test_image_multiple_faces_scored_independently_max_aggregation(tmp_path, settings, random_bundle, monkeypatch):
    script_scores(monkeypatch, image_pipeline, [0.05, 0.95, 0.4])
    boxes = [BOX, [220, 40, 300, 150], [10, 150, 70, 230]]
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a.png", random_bundle, StubDetector([boxes]), None, settings)
    assert r.faces_detected == 3 and [round(f.fake_probability, 2) for f in r.faces] == [0.05, 0.95, 0.4]
    assert r.fake_probability == 0.95 and r.classification == Classification.FAKE
    assert any("3 faces" in w for w in r.warnings)


def test_image_low_resolution_face_warning(tmp_path, settings, random_bundle, monkeypatch):
    script_scores(monkeypatch, image_pipeline, [0.1])
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a.png", random_bundle, StubDetector([[[10, 10, 60, 60]]]), None, settings)
    assert any("small" in w for w in r.warnings)


def test_image_heatmaps_generated_when_available(tmp_path, settings, random_bundle):
    ex = Explainer(random_bundle)
    if not ex.available:
        pytest.skip("pytorch_grad_cam not installed")
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a.png", random_bundle, StubDetector([[BOX]]), ex, settings, artifacts_dir=tmp_path / "art")
    assert r.faces[0].heatmap_artifact and (tmp_path / "art" / r.faces[0].heatmap_artifact).stat().st_size > 1000


def test_image_pipeline_does_not_import_video_pipeline():
    import inspect
    assert "video_pipeline" not in inspect.getsource(image_pipeline)


# ------------------------------------------------------------------------------ videos
def test_probe_and_sampling_plan(tmp_path):
    v = write_video(tmp_path / "v.avi", n=24, fps=12)
    m = video_pipeline.probe_video(v)
    assert m["width"] == 320 and m["frame_count"] == 24 and m["fps"] == pytest.approx(12, rel=0.05)
    idx, warns = video_pipeline.plan_sampling(24, 12.0, 4.0, 12, 900)
    assert idx[0] == 0 and idx[-1] == 23 and len(idx) <= 12 and idx == sorted(set(idx))
    idx, warns = video_pipeline.plan_sampling(10_000, 25.0, 2.0, 16, 900)
    assert len(idx) == 16 and any("sparse" in w for w in warns)
    idx, warns = video_pipeline.plan_sampling(100_000, 25.0, 2.0, 16, 60)
    assert idx[-1] < 60 * 25 and any("only the first" in w for w in warns)


def test_corrupt_video_rejected(tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"\x00\x01garbage" * 100)
    with pytest.raises(InvalidMediaError) as e:
        video_pipeline.probe_video(bad)
    assert e.value.code == "unreadable_video"


def run_video(tmp_path, settings, bundle, detector, scores, monkeypatch, **kw):
    script_scores(monkeypatch, video_pipeline, scores)
    v = write_video(tmp_path / "v.avi", n=48, fps=12)          # 4 s -> 17 candidate samples, capped at video_max_frames=12
    return video_pipeline.analyze_video(v, "v.avi", bundle, detector, None, settings, artifacts_dir=tmp_path / "art", **kw)


def test_video_samples_many_frames_not_only_frame_zero(tmp_path, settings, random_bundle, monkeypatch):
    det = StubDetector([[BOX]])
    r = run_video(tmp_path, settings, random_bundle, det, [0.1] * 12, monkeypatch)
    assert r.video.frames_sampled == 12 and det.calls == 12 and len(r.video.frames) == 12
    assert r.video.frames[0].frame_index == 0 and r.video.frames[-1].frame_index == 47


def test_video_fake_when_most_frames_suspicious(tmp_path, settings, random_bundle, monkeypatch):
    r = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), [0.9, 0.95, 0.85, 0.92, 0.2, 0.93, 0.96, 0.9, 0.88, 0.91, 0.94, 0.89], monkeypatch)
    assert r.classification == Classification.FAKE and r.fake_probability > 0.85
    assert r.video.temporal.longest_suspicious_run >= 4 and r.video.suspicious_segments
    assert r.video.evidence_frames and (tmp_path / "art" / r.video.evidence_frames[0].frame_artifact).is_file()
    assert r.video.aggregation_validated is False


def test_video_real_and_median_ignores_single_spike(tmp_path, settings, random_bundle, monkeypatch):
    sc = [0.02] * 11 + [0.99]
    r = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), sc, monkeypatch)
    assert r.classification == Classification.REAL and r.video.suspicious_segments       # spike is still reported as a segment
    r2 = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), sc, monkeypatch, aggregation="max")
    assert r2.classification == Classification.FAKE                                       # aggregation matters and is selectable


def test_video_no_faces_is_inconclusive(tmp_path, settings, random_bundle, monkeypatch):
    r = run_video(tmp_path, settings, random_bundle, StubDetector([[]]), [], monkeypatch)
    assert r.classification == Classification.INCONCLUSIVE and r.video.frames_with_faces == 0
    assert any("usable face" in w for w in r.warnings)


def test_video_too_few_face_frames_is_inconclusive(tmp_path, settings, random_bundle, monkeypatch):
    det = StubDetector([[BOX]] + [[]] * 11)                      # face only in the first sampled frame
    r = run_video(tmp_path, settings, random_bundle, det, [0.99], monkeypatch)
    assert r.classification == Classification.INCONCLUSIVE


def test_video_multiple_people_tracked_separately(tmp_path, settings, random_bundle, monkeypatch):
    other = [220, 40, 300, 150]
    scores = []
    for _ in range(12):
        scores += [0.03, 0.97]                                    # person A real-like, person B fake-like
    r = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX, other]]), scores, monkeypatch)
    assert len(r.video.tracks) == 2
    meds = sorted(t.median_fake_probability for t in r.video.tracks)
    assert meds == [pytest.approx(0.03), pytest.approx(0.97)]
    assert r.classification == Classification.FAKE                # max over tracks
    assert any("2 face tracks" in w for w in r.warnings)


def test_video_progress_and_cancel(tmp_path, settings, random_bundle, monkeypatch):
    seen = []
    run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), [0.1] * 12, monkeypatch, progress=lambda s, f: seen.append(f))
    assert seen and seen == sorted(seen) and seen[-1] == 1.0
    with pytest.raises(JobCancelled):
        run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), [0.1] * 12, monkeypatch, cancel=lambda: True)


def test_video_variable_missing_fps_assumed(tmp_path, settings, random_bundle, monkeypatch):
    monkeypatch.setattr(video_pipeline, "probe_video", lambda p: dict(fps=25.0, frame_count=48, width=320, height=240, duration_s=1.92, fps_guessed=True))
    r = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), [0.1] * 12, monkeypatch)
    assert any("assumed 25 fps" in w for w in r.warnings)


def test_aggregation_methods_all_run(tmp_path, settings, random_bundle, monkeypatch):
    for m in ("mean", "median", "weighted_mean", "suspicious_fraction", "max"):
        r = run_video(tmp_path, settings, random_bundle, StubDetector([[BOX]]), [0.8] * 12, monkeypatch, aggregation=m)
        assert r.video.aggregation_method == m and r.classification == Classification.FAKE
