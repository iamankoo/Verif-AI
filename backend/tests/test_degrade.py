"""Quality degradations: deterministic evaluation conditions, class-blind training augmentation, dataset hook, training smoke."""
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import BACKEND
from test_training_smoke import make_manifest
from training.dataset import AugConfig, FaceCropDataset, make_transform
from training.degrade import FIXED_NAMES, RandomQuality, fixed_degrade, random_degrade


def _img(seed=0, size=224):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _sharp(im):
    import cv2
    g = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@pytest.mark.parametrize("name", FIXED_NAMES)
def test_fixed_degradation_is_deterministic_and_keeps_shape(name):
    a, b = fixed_degrade(_img(), name, seed=3), fixed_degrade(_img(), name, seed=3)
    assert a.size == (224, 224) and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_degradations_reduce_detail_on_a_detailed_image():
    base = _sharp(_img())
    for name in ("blur_s1.5", "blur_s2.5", "downup_0.5", "downup_0.3", "phone_soft"):
        assert _sharp(fixed_degrade(_img(), name)) < base, name


def test_unknown_degradation_rejected():
    with pytest.raises(ValueError):
        fixed_degrade(_img(), "nope")


def test_random_degradation_is_class_blind():
    """The augmentation takes only the image: there is no label argument and the same call is used for every class."""
    import inspect
    assert list(inspect.signature(random_degrade).parameters) == ["im", "rng"]
    assert list(inspect.signature(RandomQuality.__call__).parameters) == ["self", "im"]
    random.seed(0)
    out = [random_degrade(_img(i)) for i in range(20)]
    assert all(o.size == (224, 224) for o in out)


def test_quality_aug_is_off_by_default_and_on_when_requested():
    assert AugConfig().quality_p == 0.0
    assert not any(type(t).__name__ == "RandomQuality" for t in make_transform(True, AugConfig()).transforms)
    assert any(type(t).__name__ == "RandomQuality" for t in make_transform(True, AugConfig(quality_p=0.7)).transforms)


def test_dataset_degrade_hook_is_deterministic_and_changes_input(tmp_path):
    m = make_manifest(tmp_path)
    clean = FaceCropDataset(str(m), ["val"], train=False)
    deg1 = FaceCropDataset(str(m), ["val"], train=False, degrade="mixed")
    deg2 = FaceCropDataset(str(m), ["val"], train=False, degrade="mixed")
    x0, _ = clean[0]; x1, _ = deg1[0]; x2, _ = deg2[0]
    assert (x0 - x1).abs().max() > 0                      # actually degraded
    assert (x1 - x2).abs().max() == 0                     # deterministic
    labels = [int(r["label"]) for r in deg1.rows]
    assert labels == clean.labels()                       # labels untouched


def test_train_with_quality_aug_and_degraded_selection(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    out = tmp_path / "runs"
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "training.train", "--manifest", str(m), "--init", str(random_checkpoint), "--name", "q",
                        "--epochs", "1", "--batch-size", "4", "--quality-aug", "0.8", "--select", "clean_degraded_mean", "--out", str(out)],
                       cwd=BACKEND, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    hist = json.loads((out / "q" / "history.json").read_text())
    assert hist[0]["val_degraded_auc"] is not None and hist[0]["val_select"] is not None
    summ = json.loads((out / "q" / "summary.json").read_text())
    assert summ["selection_criterion"] == "clean_degraded_mean"
    assert json.loads((out / "q" / "config.json").read_text())["quality_aug"] == 0.8
