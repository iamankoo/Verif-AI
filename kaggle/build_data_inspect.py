"""Builds a small Kaggle notebook that INSPECTS candidate training datasets (no training, no images in the output).

  python kaggle/build_data_inspect.py --username <you>
  kaggle kernels push -p kaggle/dist/inspect

For every leaf folder it reports image counts, file formats, resolution, file size and sharpness (variance of Laplacian) on a seeded sample, so that
label-correlated shortcuts (format, size, resolution, sharpness) can be found BEFORE any model is trained. Output: results/data_inspect.json only.
All datasets are third-party, used privately inside this notebook only; their Kaggle-listed licences are recorded in the JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "kaggle" / "dist" / "inspect"

DATASETS = {   # ref -> licence label shown by Kaggle (uploader-declared; the underlying image licences are NOT verified)
    "wish096/realvsfake-81k-by-wish": "CC0: Public Domain",
    "ajaysonicu/deepfakefusion-399k-realfake-faces": "MIT",
    "ayushmandatta1/deepdetect-2025": "Apache 2.0",
    "philosopher0808/real-vs-ai-generated-faces-dataset": "MIT",
}

CODE = r'''
import os, json, random, collections, time
import numpy as np, cv2
random.seed(0)
LIC = __LIC__
IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
base = "/kaggle/input"
print(os.listdir(base))
out = {"licences_declared_on_kaggle": LIC, "datasets": {}}
def leaf_stats(files):
    smp = random.sample(files, min(60, len(files)))
    w, h, kb, lap = [], [], [], []
    for f in smp:
        try:
            im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            h.append(im.shape[0]); w.append(im.shape[1]); kb.append(os.path.getsize(f) / 1024)
            lap.append(float(cv2.Laplacian(im, cv2.CV_64F).var()))
        except Exception:
            pass
    q = lambda a: [round(float(x), 1) for x in np.percentile(a, [10, 50, 90])] if a else None
    return dict(n_sampled=len(w), width_p10_50_90=q(w), height_p10_50_90=q(h), kb_p10_50_90=q(kb), lap_var_p10_50_90=q(lap))
t0 = time.time()
BUDGET_S = 540            # wall-clock budget per dataset; the notebook must always finish, partial statistics are fine
MAX_LEAVES = 80
for top in sorted(os.listdir(base)):
    root = os.path.join(base, top); t1 = time.time()
    leaves = collections.defaultdict(list); n_all = 0; truncated = False
    for dp, dn, fn in os.walk(root):
        imgs = [os.path.join(dp, f) for f in fn if f.lower().endswith(IMG)]
        if imgs:
            leaves[os.path.relpath(dp, root)] = imgs; n_all += len(imgs)
        if len(leaves) >= MAX_LEAVES or time.time() - t1 > BUDGET_S / 3:
            truncated = True; break
    rep = {}
    for rel, files in sorted(leaves.items()):
        if time.time() - t1 > BUDGET_S:
            truncated = True; break
        ext = collections.Counter(os.path.splitext(f)[1].lower() for f in files)
        d = dict(n=len(files), ext=dict(ext), example_name=os.path.basename(files[0])[:40]); d.update(leaf_stats(files)); rep[rel] = d
    out["datasets"][top] = dict(n_images_seen=n_all, n_leaf_folders_seen=len(leaves), n_leaf_folders_profiled=len(rep), truncated=truncated, leaves=rep)
    print(top, n_all, "images seen,", len(rep), "folders profiled, truncated:", truncated, round(time.time() - t0), "s", flush=True)
    os.makedirs("/kaggle/working/results", exist_ok=True)
    json.dump(out, open("/kaggle/working/results/data_inspect.json", "w"), indent=1)
os.makedirs
'''


def notebook() -> dict:
    src = CODE.replace("__LIC__", repr(DATASETS)).strip("\n")
    lines = src.split("\n")
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    md = {"cell_type": "markdown", "metadata": {}, "source": ["# Verif-AI data inspection (no training; statistics only)"]}
    return {"cells": [md, cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    a = ap.parse_args()
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    (DIST / "inspect.ipynb").write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    (DIST / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.username}/verif-ai-data-inspect", "title": "Verif-AI data inspect", "code_file": "inspect.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "false", "enable_internet": "false", "dataset_sources": list(DATASETS), "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
    print("kernel folder:", DIST)


if __name__ == "__main__":
    main()
