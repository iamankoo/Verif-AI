import math

import cv2
import numpy as np
import pytest
import torch

import verifai_inference as vi

REQUIRED_KEYS = {"status", "label", "fake_probability", "real_probability", "face_box",
                 "face_confidence", "threshold", "message"}


def _face(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(3, 224, 224, generator=g) * 255).round()


def test_threshold_is_unchanged():
    assert vi.DECISION_THRESHOLD == 0.5
    assert vi.LABEL_FAKE == "DEEPFAKE" and vi.LABEL_REAL == "REAL"


def test_predict_face_returns_structured_result(model):
    r = vi.predict_face(model, _face())
    assert REQUIRED_KEYS <= set(r)
    assert r["status"] == "ok"
    assert 0.0 <= r["fake_probability"] <= 1.0
    assert math.isclose(r["fake_probability"] + r["real_probability"], 1.0, abs_tol=1e-9)
    assert r["label"] == (vi.LABEL_FAKE if r["fake_probability"] > 0.5 else vi.LABEL_REAL)
    assert r["threshold"] == 0.5
    assert r["face_box"] is None                      # a bare crop has no frame coordinates
    assert not any(isinstance(v, torch.Tensor) for v in r.values())


def test_predict_face_is_deterministic(model):
    f = _face(5)
    assert vi.predict_face(model, f)["fake_probability"] == vi.predict_face(model, f)["fake_probability"]


def test_legacy_mode_matches_original_inference_math(model):
    """With match_training_jpeg=False the probability must equal the original notebook computation."""
    from test_preprocessing import _legacy_inference_tensor
    f = _face(7)
    with torch.no_grad():
        legacy = float(model(_legacy_inference_tensor(f)).item())
    assert vi.predict_face(model, f, match_training_jpeg=False)["fake_probability"] == legacy


def test_label_follows_threshold_without_altering_it(model):
    f = _face(9)
    p = vi.predict_face(model, f)["fake_probability"]
    assert vi.predict_face(model, f, threshold=p + 1e-6)["label"] == vi.LABEL_REAL
    assert vi.predict_face(model, f, threshold=p - 1e-6)["label"] == vi.LABEL_FAKE


def test_no_face_returns_structured_status(model, mtcnn):
    blank = np.full((300, 300, 3), 128, np.uint8)
    r = vi.analyze_rgb_frame(model, mtcnn, blank)
    assert REQUIRED_KEYS <= set(r)
    assert r["status"] == "no_face" and r["label"] is None
    assert r["fake_probability"] is None and r["real_probability"] is None and r["face_box"] is None


def test_unreadable_video_returns_read_error(model, mtcnn, tmp_path):
    r = vi.run_forensic_analysis(str(tmp_path / "missing.mp4"), model, mtcnn, "cpu", visualize=False, verbose=False)
    assert r["status"] == "read_error" and r["label"] is None and "Could not read" in r["message"]


def test_blank_video_returns_no_face(model, mtcnn, tmp_path):
    path = str(tmp_path / "blank.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for _ in range(3):
        w.write(np.full((120, 160, 3), 100, np.uint8))
    w.release()
    r = vi.run_forensic_analysis(path, model, mtcnn, "cpu", visualize=False, verbose=False)
    assert r["status"] == "no_face" and r["frame_index"] == 0


def test_run_forensic_analysis_still_analyses_only_first_frame_signature():
    import inspect
    params = inspect.signature(vi.run_forensic_analysis).parameters
    assert list(params)[:4] == ["video_path", "model", "mtcnn", "device"]     # original positional API kept
    assert "n_frames" not in params                                               # no video pipeline in Phase 1


# ------------------------------------------------------------- real-face tests (need local media)
@pytest.mark.needs_media
def test_reference_image_structured_result(model, mtcnn, media_image):
    r = vi.run_forensic_analysis(media_image, model, mtcnn, "cpu", visualize=False, verbose=False)
    assert r["status"] == "ok" and r["frame_index"] == 0
    assert len(r["face_box"]) == 4 and r["face_box"][2] > r["face_box"][0] and r["face_box"][3] > r["face_box"][1]
    assert 0.0 <= r["face_confidence"] <= 1.0
    assert math.isclose(r["fake_probability"] + r["real_probability"], 1.0, abs_tol=1e-9)


@pytest.mark.needs_media
def test_reference_video_first_frame_structured_result(model, mtcnn, media_video):
    r = vi.run_forensic_analysis(media_video, model, mtcnn, "cpu", visualize=False, verbose=False)
    assert r["status"] == "ok" and r["frame_index"] == 0 and len(r["face_box"]) == 4


@pytest.mark.needs_media
def test_reference_box_matches_plain_mtcnn_crop(mtcnn, media_image):
    rgb = cv2.cvtColor(cv2.imread(media_image), cv2.COLOR_BGR2RGB)
    face, box, conf = vi._detect_largest_face(mtcnn, rgb)
    assert torch.equal(face, mtcnn(rgb))              # identical to the original single call


@pytest.mark.needs_media
def test_visualization_failure_does_not_break_prediction(model, mtcnn, media_image, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("explain failed")
    monkeypatch.setattr(vi, "_visualize", boom)
    r = vi.run_forensic_analysis(media_image, model, mtcnn, "cpu", visualize=True, verbose=False)
    assert r["status"] == "ok" and "explain failed" in r["visualization_error"]
