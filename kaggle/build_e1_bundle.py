"""Builds the Kaggle notebook for E1 (zero-training quality-bin diagnostic) on Celeb-DF.

  python kaggle/build_e1_bundle.py --username <you>
  kaggle kernels push -p kaggle/dist/kernel

Inference only: no training, no checkpoint changes, no threshold tuning. Datasets attached (all private, already uploaded by build_kaggle_bundle.py):
  <you>/verifai-bundle      code, DFF crops and manifests, production checkpoint
  <you>/verifai-candidates  the other candidate checkpoints
  reubensuju/celeb-df-v2    Celeb-DF v2 (third-party copy, licence "unknown"; used only inside this private notebook)
Celeb-DF crops are written to /tmp, so they can never be part of the downloadable notebook output; the output is one JSON file and a ZIP of it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DIST = ROOT / "kaggle" / "dist"

CELLS = [
("md", """# Verif-AI - E1: quality-bin diagnostic on Celeb-DF (no training)
Is image quality a label proxy in Celeb-DF, and how do the production model and the three candidates behave across quality bins (combined, RGB-only, SRM-only)?
Fixed sharpness bins; identity/video-grouped bootstrap CIs; threshold 0.5. Crops live in /tmp and are never exported."""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
import torch
print(dict(python=platform.python_version(), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])
cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True); assert cands
shutil.copytree(os.path.dirname(os.path.dirname(cands[0])), '/kaggle/working/backend', dirs_exist_ok=True)
os.chdir('/kaggle/working/backend'); sys.path.insert(0, '.')
meta = glob.glob('/kaggle/input/**/candidates.json', recursive=True); assert meta
CAND = os.path.dirname(meta[0])
real_dirs = [d for d in glob.glob('/kaggle/input/**/Celeb-real', recursive=True) if os.path.isdir(d)]; assert real_dirs
CDF_ROOT = os.path.dirname(real_dirs[0]); print("Celeb-DF root:", CDF_ROOT)
os.makedirs('/kaggle/working/results', exist_ok=True)
open("training/stream_ablation.py", "w", encoding="utf-8").write(@@STREAM_ABLATION_SOURCE@@)
open("training/quality_bins.py", "w", encoding="utf-8").write(@@QUALITY_BINS_SOURCE@@)"""),
("code", """from collections import Counter
from training import celebdf
from verifai.config import Settings
from verifai.faces import FaceDetector
vids = celebdf.scan(CDF_ROOT)
plan, split_of = celebdf.plan_videos(vids, seed=42, frames_real=12, frames_test=16)      # identity split identical to the training runs (seed 42)
train = [v for v in plan if v["split"] == "train"][::3]                                  # every 3rd training video (deterministic): enough for the distribution, 1/3 of the time
plan = train + [v for v in plan if v["split"] in ("val", "cdf_test")]
print("videos:", Counter((v["split"], v["source"]) for v in plan))
det = FaceDetector(torch.device("cuda" if torch.cuda.is_available() else "cpu"), Settings())
t0 = time.time()
rows = celebdf.extract(plan, "/tmp/crops_e1", det, workers=4, seed=42)                   # /tmp: never part of the notebook output
print("crops:", len(rows), f"{(time.time()-t0)/60:.1f} min", Counter((r["split"], r["source"]) for r in rows))
celebdf.write_manifest(rows, "/tmp/celebdf_e1_manifest.csv")"""),
("code", """from training import quality_bins
MODELS = {
    "1_production_v2": "models/verifai_v2_best.pth",
    "2_dff_only_qualityAug": os.path.join(CAND, "dff_qa_R1.pth"),
    "3_dff_celebdf_noQualityAug": os.path.join(CAND, "cdf_C0_noQA.pth"),
    "4_dff_celebdf_qualityAug": os.path.join(CAND, "cdf_C1_qa.pth"),
}
res = quality_bins.run(MODELS, "/tmp/celebdf_e1_manifest.csv", ["train", "val", "cdf_test"], ["val", "cdf_test"], ref_manifest="manifests/dff_faces_manifest.csv",
                       ref_split="train", n_ref=150, n_boot=1000, device="cuda" if torch.cuda.is_available() else "cpu")
res["crops_extracted"] = {k: v for k, v in Counter((r["split"] + "/" + r["source"]) for r in rows).items()}
json.dump(res, open("/kaggle/working/results/cdf_e1.json", "w"), indent=1)
for sp, d in res["dependence"].items():
    s = d["auc_fake_from_softness_alone"]
    print(sp, "n=", d["n"], "real/fake=", d["n_real"], d["n_fake"], "softness-alone AUC", s["auc"], s["ci95"], "chi2 p", d["p_value"], "V", d["cramers_v"])"""),
("code", """import zipfile
out = "/kaggle/working/verifai_e1_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("/kaggle/working/results/cdf_e1.json", "results/cdf_e1.json")
print(out, round(os.path.getsize(out) / 1e3, 1), "KB (one JSON file; no Celeb-DF frames or crops)")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)"""),
]


def notebook() -> dict:
    cells = []
    for kind, src in CELLS:
        src = src.replace("@@STREAM_ABLATION_SOURCE@@", repr((BACKEND / "training/stream_ablation.py").read_text(encoding="utf-8")))
        src = src.replace("@@QUALITY_BINS_SOURCE@@", repr((BACKEND / "training/quality_bins.py").read_text(encoding="utf-8")))
        lines = src.split("\n")
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    a = ap.parse_args()
    kn = DIST / "kernel"
    if kn.exists():
        shutil.rmtree(kn)
    kn.mkdir(parents=True)
    (kn / "verifai_train_kaggle.ipynb").write_text(json.dumps(notebook(), indent=1))
    (kn / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.username}/verif-ai-e1", "title": "Verif-AI e1", "code_file": "verifai_train_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{a.username}/verifai-bundle", f"{a.username}/verifai-candidates", "reubensuju/celeb-df-v2"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    print("kernel folder:", kn)


if __name__ == "__main__":
    main()
