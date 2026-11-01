import re

import numpy as np
import pytest

from conftest import StubDetector, write_png, write_video
from verifai import image_pipeline, video_pipeline
from verifai.explain import Explainer
from verifai.report import LIMITATIONS, build_pdf

BOX = [60, 40, 200, 200]


def pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def test_image_report(tmp_path, settings, random_bundle, monkeypatch):
    monkeypatch.setattr(image_pipeline, "score_crops", lambda b, c, bs=16, m=True: np.array([0.8] * len(c)))
    art = tmp_path / "art"
    ex = Explainer(random_bundle)
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a<b>&.png", random_bundle, StubDetector([[BOX, [220, 40, 300, 150]]]), ex, settings, artifacts_dir=art)
    pdf = build_pdf(r, art)
    assert pdf[:5] == b"%PDF-" and pages(pdf) >= 1 and len(pdf) > 5000


def test_video_report_with_timeline_and_evidence(tmp_path, settings, random_bundle, monkeypatch):
    monkeypatch.setattr(video_pipeline, "score_crops", lambda b, c, bs=16, m=True: np.array([0.2, 0.9, 0.95, 0.1] * 10)[:len(c)])
    art = tmp_path / "art"
    r = video_pipeline.analyze_video(write_video(tmp_path / "v.avi"), "v.avi", random_bundle, StubDetector([[BOX]]), Explainer(random_bundle), settings, artifacts_dir=art)
    pdf = build_pdf(r, art)
    assert pdf[:5] == b"%PDF-" and len(pdf) > 8000


def test_report_for_inconclusive_result_without_artifacts(tmp_path, settings, random_bundle):
    r = image_pipeline.analyze_image(write_png(tmp_path / "a.png"), "a.png", random_bundle, StubDetector([[]]), None, settings)
    assert build_pdf(r, None)[:5] == b"%PDF-"


def test_limitations_statements_present():
    joined = " ".join(LIMITATIONS).lower()
    for phrase in ("not definitive proof", "uncalibrated", "grad-cam", "does not prove authenticity"):
        assert phrase in joined
