"""Face extraction + manifest validation for an IMAGE dataset manifest (e.g. DeepFakeFace sample).

For every raw image: detect the largest face with MTCNN (image_size=224, margin=20 — identical to the
original project and to inference), save the crop as JPEG via cv2.imwrite (quality 95 — the training-time
storage that inference reproduces), compute SHA-256 / difference-hash, then validate the dataset:
  * identity-disjoint splits, pair integrity,
  * exact + near duplicates (perceptual hash) across splits (offending items are DROPPED and logged),
  * per-class face-detection failure rates, class/generator distribution,
  * shortcut diagnostics (AUC of raw file size and of crop JPEG size as label predictors).
Outputs: manifests/<name>_faces_manifest.csv and manifests/<name>_validation_report.json

Usage (from backend/):  python -m training.prepare_faces --manifest manifests/dff_raw_manifest.csv --name dff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

from verifai.config import FACE_MARGIN, INPUT_SIZE, TRAINING_JPEG_QUALITY
from verifai.model import resolve_device

from .data_validation import (auc_of_feature, check_group_disjoint, cross_split_duplicate_pairs, dhash, distribution, find_duplicates)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="manifests/dff_raw_manifest.csv")
    ap.add_argument("--name", default="dff")
    ap.add_argument("--crops", default="data/crops")
    ap.add_argument("--max-hamming", type=int, default=4)
    a = ap.parse_args()

    from facenet_pytorch import MTCNN
    dev = resolve_device("auto")
    mtcnn = MTCNN(image_size=INPUT_SIZE, margin=FACE_MARGIN, keep_all=False, post_process=False, device=dev)   # largest face
    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    out_rows, failures = [], Counter()
    for n, r in enumerate(rows):
        img = cv2.imread(r["raw_path"])
        r["face_detected"] = 0
        r["crop_path"] = ""
        if img is None:
            failures[(r["split"], r["label"], "unreadable")] += 1
            out_rows.append(r); continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_dh = dhash(Image.fromarray(rgb))
        r["raw_dhash"] = str(raw_dh)
        try:
            face = mtcnn(rgb)
        except Exception:
            face = None
        if face is None:
            failures[(r["split"], r["label"], "no_face")] += 1
            out_rows.append(r); continue
        crop = np.clip(face.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
        d = os.path.join(a.crops, r["split"], "fake" if int(r["label"]) else "real")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, os.path.splitext(os.path.basename(r["raw_path"]))[0] + ".jpg").replace("\\", "/")
        cv2.imwrite(p, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, TRAINING_JPEG_QUALITY])
        r["crop_path"], r["face_detected"] = p, 1
        r["crop_bytes"] = os.path.getsize(p)
        r["crop_sha256"] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        r["crop_dhash"] = str(dhash(Image.open(p)))
        out_rows.append(r)
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(rows)}", flush=True)

    ok = [r for r in out_rows if r["face_detected"]]
    report = {"input_manifest": a.manifest, "total_raw_images": len(rows), "faces_kept": len(ok),
              "detection_failures": {f"{k[0]}|{'fake' if int(k[1]) else 'real'}|{k[2]}": v for k, v in failures.items()},
              "mtcnn": {"image_size": INPUT_SIZE, "margin": FACE_MARGIN, "selection": "largest face", "device": str(dev)},
              "crop_jpeg_quality": TRAINING_JPEG_QUALITY}

    # ---- leakage / duplicate validation on crops
    report["group_violations_identity"] = check_group_disjoint(ok, "identity")
    report["group_violations_pair_id"] = check_group_disjoint(ok, "pair_id")
    items = [dict(sha256=r["crop_sha256"], dhash=int(r["crop_dhash"]), split=r["split"], label=int(r["label"])) for r in ok]
    dups = find_duplicates(items, a.max_hamming)
    cross = cross_split_duplicate_pairs(items, dups)
    report["duplicates"] = {"exact_pairs": len(dups["exact"]), "near_pairs": len(dups["near"]), "cross_split_pairs": len(cross),
                            "max_hamming": a.max_hamming}
    drop = set()
    for i, j, d in cross:        # keep the item in the larger-priority split (train < val < test): drop the later-split copy
        order = {"train": 0, "val": 1, "test_internal": 2, "test_crossgen": 3}
        drop.add(i if order[items[i]["split"]] >= order[items[j]["split"]] else j)
    report["dropped_cross_split_duplicates"] = sorted(ok[i]["crop_path"] for i in drop)
    final = [r for k, r in enumerate(ok) if k not in drop]

    # ---- distribution + shortcut diagnostics
    report["distribution_after_validation"] = distribution(final)
    short = {}
    for sp in sorted({r["split"] for r in final}):
        sub = [r for r in final if r["split"] == sp]
        y = [int(r["label"]) for r in sub]
        if len(set(y)) == 2:
            short[sp] = {"auc_raw_file_bytes": round(auc_of_feature([float(r["bytes"]) for r in sub], y), 4),
                         "auc_crop_jpeg_bytes": round(auc_of_feature([float(r["crop_bytes"]) for r in sub], y), 4)}
    report["shortcut_diagnostics_note"] = ("AUC of a single scalar (file size) as a real/fake predictor. 0.5 = no signal; values far from 0.5 mean the "
                                           "classes differ in compression, which a network could exploit instead of manipulation traces.")
    report["shortcut_diagnostics"] = short
    report["identities_per_split"] = {sp: len({r["identity"] for r in final if r["split"] == sp}) for sp in sorted({r["split"] for r in final})}

    fn = f"manifests/{a.name}_faces_manifest.csv"
    keep = ["split", "label", "generator", "identity", "pair_id", "source_zip", "member", "raw_path", "sha256", "bytes", "crop_path", "crop_sha256", "crop_bytes", "crop_dhash"]
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore"); w.writeheader(); w.writerows(final)
    json.dump(report, open(f"manifests/{a.name}_validation_report.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: report[k] for k in ("faces_kept", "detection_failures", "duplicates", "distribution_after_validation", "shortcut_diagnostics", "identities_per_split")}, indent=1))
    print("wrote", fn)
    bad = report["group_violations_identity"] or report["group_violations_pair_id"]
    if bad:
        raise SystemExit("LEAKAGE DETECTED: see validation report")


if __name__ == "__main__":
    main()
