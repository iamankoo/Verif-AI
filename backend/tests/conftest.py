"""Shared fixtures for the Verif-AI tests.

Run from backend/:  python -m pytest tests -q
Optional real-face tests: set VERIFAI_TEST_IMAGE / VERIFAI_TEST_VIDEO to local media (skipped otherwise;
no personal media is stored in the repository). Tests that need trained weights are skipped when the
checkpoint files are absent (e.g. a fresh clone without Git LFS content).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
SCRIPTS = BACKEND / "scripts"
for p in (str(BACKEND), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

CHECKPOINT = SCRIPTS / "two_stream_model_epoch_10.pth"           # original prototype checkpoint (legacy tests)
IMAGENET_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth")


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_media: requires VERIFAI_TEST_IMAGE / VERIFAI_TEST_VIDEO")
    config.addinivalue_line("markers", "needs_weights: requires a trained checkpoint file")


# ------------------------------------------------------------------ legacy (Phase 1) fixtures
@pytest.fixture(scope="session")
def checkpoint_path():
    if not CHECKPOINT.is_file():
        pytest.skip(f"checkpoint not present: {CHECKPOINT}")
    return str(CHECKPOINT)


@pytest.fixture(scope="session")
def cpu():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def model(checkpoint_path, cpu):
    import verifai_inference as vi
    m = vi.TwoStreamNet(pretrained=False).to(cpu)
    vi.load_checkpoint(m, checkpoint_path, cpu)
    m.eval()
    return m


@pytest.fixture(scope="session")
def mtcnn(cpu):
    import verifai_inference as vi
    return vi.MTCNN(image_size=vi.INPUT_SIZE, margin=20, device=cpu, post_process=False)


@pytest.fixture(scope="session")
def media_image():
    p = os.environ.get("VERIFAI_TEST_IMAGE")
    if not p or not os.path.isfile(p):
        pytest.skip("VERIFAI_TEST_IMAGE not set")
    return p


@pytest.fixture(scope="session")
def media_video():
    p = os.environ.get("VERIFAI_TEST_VIDEO")
    if not p or not os.path.isfile(p):
        pytest.skip("VERIFAI_TEST_VIDEO not set")
    return p


# ------------------------------------------------------------------ new-package fixtures
@pytest.fixture(scope="session")
def random_checkpoint(tmp_path_factory):
    """A randomly initialised checkpoint (new {state_dict, meta} format) - exercises code paths, not accuracy."""
    from verifai.model import TwoStreamNet
    torch.manual_seed(0)
    p = tmp_path_factory.mktemp("ckpt") / "random.pth"
    torch.save({"state_dict": TwoStreamNet(pretrained=False).state_dict(),
                "meta": {"name": "random-init-test", "version": "0.0", "trained_on": "nothing (unit test)"}}, p)
    return p


@pytest.fixture(scope="session")
def random_bundle(random_checkpoint):
    from verifai.model import load_model
    return load_model(random_checkpoint, "cpu")


@pytest.fixture()
def settings(tmp_path):
    from verifai.config import Settings
    return Settings(storage_dir=tmp_path / "storage", device="cpu", video_sample_fps=4.0, video_max_frames=12, enable_heatmaps=True)


class StubDetector:
    """Returns scripted faces so pipeline logic can be tested deterministically without real faces."""

    def __init__(self, per_call):
        self.per_call = per_call          # list (cycled) of lists of boxes; [] = no face
        self.calls = 0

    def detect(self, rgb):
        from verifai.faces import DetectedFace
        boxes = self.per_call[self.calls % len(self.per_call)]
        self.calls += 1
        g = torch.Generator().manual_seed(self.calls)
        return [DetectedFace(box=list(b), confidence=0.99, crop=(torch.rand(3, 224, 224, generator=g) * 255).round(),
                             min_side_px=min(b[2] - b[0], b[3] - b[1]), low_resolution=min(b[2] - b[0], b[3] - b[1]) < 96) for b in boxes]


@pytest.fixture()
def stub_detector():
    return StubDetector


def write_png(path: Path, w=320, h=240, seed=0) -> Path:
    import cv2
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return path


def write_video(path: Path, n=24, fps=12, w=320, h=240) -> Path:
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    rng = np.random.default_rng(1)
    for i in range(n):
        vw.write(np.clip(rng.integers(0, 255, (h, w, 3)) * 0.3 + i, 0, 255).astype(np.uint8))
    vw.release()
    return path
