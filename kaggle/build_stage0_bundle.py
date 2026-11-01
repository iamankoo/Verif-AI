"""Builds the Kaggle notebook for stage-0 gate validation on the Celeb-DF VALIDATION identities (inference only).

  python kaggle/build_stage0_bundle.py --username <you>
  kaggle kernels push -p kaggle/dist/kernel

Protocol: docs/STAGE0_PROTOCOL.md (frozen). Only the Celeb-DF `val` split is extracted and scored (clean and with the fixed class-blind degradations); the Celeb-DF
test identities are never read. Crops are written to /tmp (never part of the notebook output). The output is one JSON file and a ZIP of it.
Datasets (all private): <you>/verifai-bundle, <you>/verifai-candidates, reubensuju/celeb-df-v2 (third-party copy, licence "unknown", used only inside this notebook).
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
("md", """# Verif-AI - stage-0 gate validation on Celeb-DF validation identities (inference only)
Protocol docs/STAGE0_PROTOCOL.md v1 (frozen). Sets: V-CDF-clean and V-CDF-stress (class-blind deterministic degradations). No test identity is read; no training; threshold 0.5."""),
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
CDF_ROOT = os.path.dirname(real_dirs[0])
os.makedirs('/kaggle/working/results', exist_ok=True)
open("training/stream_ablation.py", "w", encoding="utf-8").write(@@STREAM_ABLATION_SOURCE@@)
open("training/quality_bins.py", "w", encoding="utf-8").write(@@QUALITY_BINS_SOURCE@@)
open("training/gate_eval.py", "w", encoding="utf-8").write(@@GATE_EVAL_SOURCE@@)"""),
("code", """from collections import Counter
from training import celebdf
from verifai.config import Settings
from verifai.faces import FaceDetector
vids = celebdf.scan(CDF_ROOT)
plan, split_of = celebdf.plan_videos(vids, seed=42, frames_real=12, frames_test=16)      # identity split identical to the training runs
plan = [v for v in plan if v["split"] == "val"]                                          # VALIDATION ONLY: no test identity is read
print("val videos:", Counter(v["source"] for v in plan))
det = FaceDetector(torch.device("cuda" if torch.cuda.is_available() else "cpu"), Settings())
t0 = time.time()
rows = celebdf.extract(plan, "/tmp/crops_stage0", det, workers=4, seed=42)               # /tmp: never part of the notebook output
print("val crops:", len(rows), f"{(time.time()-t0)/60:.1f} min", Counter(r["source"] for r in rows))
celebdf.write_manifest(rows, "/tmp/cdf_val_manifest.csv")"""),
("code", """from training import gate_eval
MODELS = {
    "1_production_v2": "models/verifai_v2_best.pth",
    "2_dff_only_qualityAug": os.path.join(CAND, "dff_qa_R1.pth"),
    "3_dff_celebdf_noQualityAug": os.path.join(CAND, "cdf_C0_noQA.pth"),
    "4_dff_celebdf_qualityAug": os.path.join(CAND, "cdf_C1_qa.pth"),
}
SETS = {"V-CDF-clean": dict(split="val", degrade=""), "V-CDF-stress": dict(split="val", degrade="mixed")}
res = gate_eval.run(MODELS, "/tmp/cdf_val_manifest.csv", SETS, ref_manifest="manifests/dff_faces_manifest.csv", device="cuda" if torch.cuda.is_available() else "cpu")
res["crops_extracted"] = dict(Counter(r["source"] for r in rows))
json.dump(res, open("/kaggle/working/results/cdf_gate.json", "w"), indent=1)
for n, m in res["models"].items():
    print(n, m["verdict"], m["failed_checks"])"""),
("code", """import zipfile
out = "/kaggle/working/verifai_stage0_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.write("/kaggle/working/results/cdf_gate.json", "results/cdf_gate.json")
print(out, round(os.path.getsize(out) / 1e3, 1), "KB (one JSON file; no Celeb-DF frames or crops)")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)"""),
]


def notebook() -> dict:
    cells = []
    subs = {"@@STREAM_ABLATION_SOURCE@@": "stream_ablation.py", "@@QUALITY_BINS_SOURCE@@": "quality_bins.py", "@@GATE_EVAL_SOURCE@@": "gate_eval.py"}
    for kind, src in CELLS:
        for k, f in subs.items():
            src = src.replace(k, repr((BACKEND / "training" / f).read_text(encoding="utf-8")))
        lines = src.split("\n")
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


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
        "id": f"{a.username}/verif-ai-stage0", "title": "Verif-AI stage0", "code_file": "verifai_train_kaggle.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{a.username}/verifai-bundle", f"{a.username}/verifai-candidates", "reubensuju/celeb-df-v2"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    print("kernel folder:", kn)


if __name__ == "__main__":
    main()
