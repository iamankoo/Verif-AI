"""E3 data module: file-disjoint seeded splits, face-crop extraction through a detector, and near-duplicate leak removal."""
import csv
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from training import modern_data as md


def _img(path, seed, size=64):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.resize(base, (size, size), interpolation=cv2.INTER_NEAREST))      # blocky pattern: distinct dHash per seed


def _tree(tmp_path):
    for name, n in (("ffhq", 40), ("sfhq/pt1", 20), ("sfhq/pt2", 20), ("dd_train", 30), ("dd_test", 15)):
        d = tmp_path / name; d.mkdir(parents=True)
        for i in range(n):
            _img(d / f"{name.replace('/', '_')}_{i}.jpg", seed=hash((name, i)) % 10_000_000)
    return tmp_path


QUOTAS = dict(ffhq=(20, 5, 8), sfhq=(10, 3, 4), dd_train=(15, 5), dd_test=6)


def test_plan_is_seeded_file_disjoint_and_uses_the_modern_test_split(tmp_path):
    t = _tree(tmp_path)
    args = (str(t / "ffhq"), [str(t / "sfhq/pt1"), str(t / "sfhq/pt2")], str(t / "dd_train"), str(t / "dd_test"))
    a = md.plan_modern(*args, seed=7, quotas=QUOTAS)
    b = md.plan_modern(*args, seed=7, quotas=QUOTAS)
    assert [x["path"] for x in a] == [x["path"] for x in b]                                    # reproducible
    assert len({x["path"] for x in a}) == len(a)                                               # no file used twice
    by = {}
    for x in a:
        by.setdefault((x["split"], x["generator"]), 0); by[(x["split"], x["generator"])] += 1
    assert by[("train", "real_ffhq")] == 20 and by[("val", "sfhq")] == 3 and by[("modern_test", "deepdetect_fake")] == 6
    assert {x["split"] for x in a} == {"train", "val", "modern_test"}
    assert all(x["label"] == (0 if x["generator"] == "real_ffhq" else 1) for x in a)           # FFHQ real, everything else fake
    assert md.plan_modern(*args, seed=8, quotas=QUOTAS) != a                                   # the seed matters


class StubDetector:
    def __init__(self, reject=()):
        self.reject = set(reject)

    def detect(self, rgb):
        if any(int(rgb.mean()) == r for r in self.reject):
            return []
        crop = torch.from_numpy(cv2.resize(rgb, (224, 224))).permute(2, 0, 1).float()
        return [SimpleNamespace(box=[0, 0, rgb.shape[1], rgb.shape[0]], crop=crop)]


def test_extract_saves_q95_crops_and_drops_images_without_a_face(tmp_path):
    items = [dict(path=str(tmp_path / f"a{i}.jpg"), split="train", label=i % 2, source="ffhq" if i % 2 == 0 else "sfhq", generator="g", identity=f"x:{i}") for i in range(4)]
    for i, it in enumerate(items):
        cv2.imwrite(it["path"], np.full((40, 40, 3), 10 + i * 40, dtype=np.uint8))
    det = StubDetector(reject={10 + 2 * 40})                                                     # the third image has "no face"
    rows = md.extract(items, str(tmp_path / "crops"), det, workers=2, log_every=0)
    assert len(rows) == 3 and all(os.path.isfile(r["crop_path"]) for r in rows)
    assert {r["split"] for r in rows} == {"train"} and all(set(md.FIELDS) <= set(r) for r in rows)


def test_near_duplicate_leaks_are_removed_from_val_and_test(tmp_path):
    d = tmp_path / "c"; d.mkdir()
    for n, seed in (("train0", 1), ("train1", 2), ("val_dup", 1), ("val_new", 3), ("test_dup_of_val", 3), ("test_dup_of_train", 2), ("test_new", 4)):
        _img(d / f"{n}.jpg", seed, size=96)
    row = lambda n, sp, lab=0: dict(split=sp, label=lab, generator="g", identity=n, crop_path=str(d / f"{n}.jpg"), source="s", video="", frame="")
    rows = [row("train0", "train"), row("train1", "train"), row("val_dup", "val"), row("val_new", "val"), row("test_dup_of_val", "modern_test"), row("test_dup_of_train", "modern_test"), row("test_new", "modern_test")]
    kept, removed = md.remove_leaks(rows, reference_paths=[], max_hamming=3)
    names = {r["identity"] for r in kept}
    assert names == {"train0", "train1", "val_new", "test_new"}
    assert removed == {"val": 1, "modern_test": 2}
    kept2, _ = md.remove_leaks(rows, reference_paths=[str(d / "test_new.jpg")])                  # reference crops of other sources count as training crops too
    assert "test_new" not in {r["identity"] for r in kept2}


def test_hamming_mask_and_merge(tmp_path):
    ref = np.array([0, 0xFFFF], dtype=np.uint64)
    assert md.near_duplicate_mask(ref, np.array([0b111, 0xFFFF_FFFF_FFFF_FFFF], dtype=np.uint64), 3).tolist() == [True, False]
    base = tmp_path / "base.csv"
    base.write_text("split,label,generator,identity,crop_path,source,video,frame\ntrain,0,real,dff:1,a.jpg,dff,,\n", encoding="utf-8")
    md.write_merged(str(base), [dict(split="val", label=1, generator="sfhq", identity="sfhq:1", crop_path="b.jpg", source="sfhq", video="", frame="")], str(tmp_path / "m.csv"))
    rows = list(csv.DictReader(open(tmp_path / "m.csv", encoding="utf-8")))
    assert [r["generator"] for r in rows] == ["real", "sfhq"]
