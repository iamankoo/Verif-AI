"""End-to-end smoke test of the training script on a tiny synthetic manifest (CPU, ~30 s).

Regression: a checkpoint written by training.train must load through the app's strict `weights_only=True`
loader (an earlier version stored torch.__version__ in the metadata and broke loading).
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import BACKEND
from verifai.model import load_model


def make_manifest(tmp: Path) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("train", 12), ("val", 8)):
        for i in range(n):
            label = i % 2
            d = tmp / "crops" / split; d.mkdir(parents=True, exist_ok=True)
            p = d / f"{label}_{i}.jpg"
            Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)).save(p, quality=95)
            rows.append(dict(split=split, label=label, generator="real" if label == 0 else "synthetic", identity=f"{split}{i}", crop_path=str(p).replace("\\", "/")))
    m = tmp / "m.csv"
    with open(m, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return m


def test_train_one_epoch_and_reload(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    out = tmp_path / "runs"
    r = subprocess.run([sys.executable, "-X", "utf8", "-m", "training.train", "--manifest", str(m), "--init", str(random_checkpoint), "--name", "smoke",
                        "--epochs", "1", "--batch-size", "4", "--out", str(out)], cwd=BACKEND, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    run = out / "smoke"
    for f in ("best.pth", "last.pth", "history.csv", "history.json", "config.json", "summary.json"):
        assert (run / f).is_file(), f
    hist = json.loads((run / "history.json").read_text())
    assert hist[0]["epoch"] == 1 and "val_auc" in hist[0] and "train_loss" in hist[0]
    b = load_model(run / "best.pth", "cpu")                       # strict + weights_only
    assert b.info.name == "verifai-two-stream-smoke" and b.info.parameters == 11_972_954
    assert json.loads((run / "config.json").read_text())["seed"] == 42
