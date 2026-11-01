"""Regression guard on PRIVATE genuine videos (never committed, never hardcoded).

Set  VERIFAI_PRIVATE_GENUINE_VIDEOS  to one or more paths (separated by ';' on Windows or ':' elsewhere) of videos that you KNOW are
genuine (e.g. recorded by you, with consent). Without the variable the tests are skipped, so CI and fresh clones are unaffected.

PowerShell:  $env:VERIFAI_PRIVATE_GENUINE_VIDEOS = "C:\\path\\a.mp4;C:\\path\\b.mp4"; python -m pytest tests/test_private_genuine_video_regression.py -q -s

What is checked (all behavioural, none about a file name):
  * the video pipeline is deterministic (two runs give the same score);
  * a face is found in every sampled frame of these ordinary talking-head clips (detector/tracking sanity);
  * the video is not given a CONFIDENT fake verdict (FAKE with score >= 0.90). This is the failure that was investigated
    (docs/VIDEO_FALSE_POSITIVE_INVESTIGATION.md). It is a smoke guard on a handful of clips, NOT evidence of accuracy.
Scores are printed (-s) so you can see them; the clips and any frames stay on your machine.
"""
import os
from pathlib import Path

import pytest

from verifai.config import Settings
from verifai.faces import FaceDetector
from verifai.model import load_model
from verifai.schemas import Classification
from verifai.video_pipeline import analyze_video

RAW = os.environ.get("VERIFAI_PRIVATE_GENUINE_VIDEOS", "")
VIDEOS = [Path(p) for p in RAW.split(os.pathsep) if p.strip()]

pytestmark = pytest.mark.skipif(not VIDEOS, reason="VERIFAI_PRIVATE_GENUINE_VIDEOS not set (private genuine videos are never part of the repository)")


@pytest.fixture(scope="module")
def stack():
    s = Settings.from_env()
    if not Path(s.model_path).is_file():
        pytest.skip(f"model checkpoint not found: {s.model_path}")
    bundle = load_model(s.model_path, "cpu")
    return s, bundle, FaceDetector(bundle.device, s)


@pytest.mark.parametrize("video", VIDEOS, ids=[p.name for p in VIDEOS])
def test_genuine_video_is_processed_consistently_and_not_confidently_flagged(stack, video):
    if not video.is_file():
        pytest.skip(f"not found: {video}")
    s, bundle, det = stack
    a = analyze_video(video, video.name, bundle, det, None, s)
    b = analyze_video(video, video.name, bundle, det, None, s)
    print(f"\n[private] {video.name}: {a.classification.value} score={a.fake_probability:.4f} sampled={a.video.frames_sampled} "
          f"with_face={a.video.frames_with_faces} model={bundle.info.name}")
    assert a.fake_probability == pytest.approx(b.fake_probability, abs=1e-6)                 # deterministic
    assert a.video.frames_with_faces == a.video.frames_sampled > 0                           # faces found in every sampled frame
    assert not (a.classification == Classification.FAKE and a.fake_probability >= 0.90), (
        f"{video.name} is declared genuine but received a confident FAKE verdict ({a.fake_probability:.3f})")
