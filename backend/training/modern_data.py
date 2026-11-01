"""Modern-generator face data for experiment E3 (third-party image datasets, used privately inside a Kaggle notebook; nothing is exported).

Sources used (see docs/DATASET_LICENCE_REVIEW.md for the review; Kaggle-listed licences were NOT accepted as proof):
  * FFHQ real faces (upstream: NVlabs, CC BY-NC-SA 4.0, non-commercial research use with attribution)      -> label 0, generator "real_ffhq"
  * SFHQ synthetic faces (upstream CC0)                                                                     -> label 1, generator "sfhq"
  * DeepDetect-2025 FAKE side only (modern generators, face-filtered; its 'real' side has unspecified origin and is not used) -> label 1, "deepdetect_fake"
Splits are file-disjoint and seeded. Near-duplicates (64-bit dHash, Hamming <= 3) of any training crop are removed from validation and test.
Crops are produced with the production face detector, so they match what the application feeds the model.
"""
from __future__ import annotations

import csv
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
FIELDS = ["split", "label", "generator", "identity", "crop_path", "source", "video", "frame"]
DEFAULT_QUOTAS = dict(ffhq=(8000, 1000, 1500), sfhq=(3000, 500, 700), dd_train=(9000, 1000), dd_test=2500)   # (train, val, test) image counts before face filtering


def list_images(root: str) -> List[str]:
    return sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def _take(files: List[str], counts: Sequence[int], rng: random.Random) -> List[List[str]]:
    files = list(files)
    rng.shuffle(files)
    out, i = [], 0
    for c in counts:
        out.append(files[i:i + c]); i += c
    return out


def plan_modern(ffhq_root: str, sfhq_roots: Sequence[str], dd_train_fake_root: str, dd_test_fake_root: str, seed: int = 42, quotas: Dict = DEFAULT_QUOTAS) -> List[dict]:
    """File-disjoint, seeded sample. FFHQ and SFHQ test files and DeepDetect test fakes form the split 'modern_test' (never used for training or selection)."""
    rng = random.Random(seed)
    items: List[dict] = []

    def add(files, split, label, source, generator):
        for f in files:
            items.append(dict(path=f, split=split, label=label, source=source, generator=generator, identity=f"{source}:{Path(f).stem}"))

    tr, va, te = _take(list_images(ffhq_root), quotas["ffhq"], rng)
    add(tr, "train", 0, "ffhq", "real_ffhq"); add(va, "val", 0, "ffhq", "real_ffhq"); add(te, "modern_test", 0, "ffhq", "real_ffhq")
    sf = [f for r in sfhq_roots for f in list_images(r)]
    tr, va, te = _take(sf, quotas["sfhq"], rng)
    add(tr, "train", 1, "sfhq", "sfhq"); add(va, "val", 1, "sfhq", "sfhq"); add(te, "modern_test", 1, "sfhq", "sfhq")
    tr, va = _take(list_images(dd_train_fake_root), quotas["dd_train"], rng)
    add(tr, "train", 1, "deepdetect", "deepdetect_fake"); add(va, "val", 1, "deepdetect", "deepdetect_fake")
    (te,) = _take(list_images(dd_test_fake_root), [quotas["dd_test"]], rng)
    add(te, "modern_test", 1, "deepdetect", "deepdetect_fake")
    return items


def _read_rgb(path: str):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def extract(items: List[dict], out_root: str, detector, workers: int = 4, log_every: int = 1000) -> List[dict]:
    """Detect the largest face with the production detector and save a q95 crop. Images without a usable face are dropped (and counted by the caller)."""
    import cv2
    out = Path(out_root)
    rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [(it, ex.submit(_read_rgb, it["path"])) for it in items]
        for k, (it, fut) in enumerate(futs):
            rgb = fut.result()
            if rgb is not None:
                dets = detector.detect(rgb)
                if dets:
                    face = max(dets, key=lambda x: (x.box[2] - x.box[0]) * (x.box[3] - x.box[1]))
                    arr = np.clip(face.crop.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
                    d = out / it["split"] / ("fake" if it["label"] else "real"); d.mkdir(parents=True, exist_ok=True)
                    p = d / f"{it['source']}_{Path(it['path']).stem}.jpg"
                    cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rows.append(dict(split=it["split"], label=it["label"], generator=it["generator"], identity=it["identity"], crop_path=str(p).replace("\\", "/"), source=it["source"], video="", frame=""))
            if log_every and k % log_every == 0:
                print(f"  modern crops {k + 1}/{len(futs)} -> {len(rows)} kept", flush=True)
    return rows


# ------------------------------------------------------------------ leakage control
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash64(path: str) -> int:
    import cv2
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hashes(paths: Sequence[str]) -> np.ndarray:
    return np.array([dhash64(p) for p in paths], dtype=np.uint64)


def near_duplicate_mask(reference: np.ndarray, candidates: np.ndarray, max_hamming: int = 3) -> np.ndarray:
    """True for every candidate that has a reference hash within max_hamming bits."""
    hit = np.zeros(len(candidates), dtype=bool)
    if len(reference) == 0:
        return hit
    for i, c in enumerate(candidates):
        x = np.bitwise_xor(reference, np.uint64(c)).view(np.uint8).reshape(-1, 8)
        hit[i] = bool((_POP[x].sum(1) <= max_hamming).any())
    return hit


def remove_leaks(new_rows: List[dict], reference_paths: Sequence[str], max_hamming: int = 3) -> tuple[List[dict], Dict[str, int]]:
    """Drop validation rows that near-duplicate any training crop, and test rows that near-duplicate any training or validation crop.
    reference_paths = training crops of ALL sources (DeepFakeFace, Celeb-DF and the new data). Returns (kept rows, counts removed per split)."""
    train_h = np.concatenate([hashes(reference_paths), hashes([r["crop_path"] for r in new_rows if r["split"] == "train"])]) if len(reference_paths) or new_rows else np.zeros(0, np.uint64)
    removed: Dict[str, int] = {}
    keep = [r for r in new_rows if r["split"] == "train"]
    val = [r for r in new_rows if r["split"] == "val"]
    vm = near_duplicate_mask(train_h, hashes([r["crop_path"] for r in val]), max_hamming)
    removed["val"] = int(vm.sum()); val = [r for r, m in zip(val, vm) if not m]
    ref2 = np.concatenate([train_h, hashes([r["crop_path"] for r in val])])
    for sp in sorted({r["split"] for r in new_rows} - {"train", "val"}):
        rs = [r for r in new_rows if r["split"] == sp]
        m = near_duplicate_mask(ref2, hashes([r["crop_path"] for r in rs]), max_hamming)
        removed[sp] = int(m.sum()); keep += [r for r, mm in zip(rs, m) if not mm]
    return keep + val, removed


def write_merged(base_manifest: str, modern_rows: List[dict], out: str) -> None:
    rows = list(csv.DictReader(open(base_manifest, encoding="utf-8"))) + modern_rows
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
