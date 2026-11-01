"""Builds the Kaggle GPU notebook for experiment E3 (modern-generator training). Protocol: docs/E3_PROTOCOL.md (pre-registered).

  python kaggle/build_e3_bundle.py --username <you>
  kaggle kernels push -p kaggle/dist/e3

Private notebook. Datasets (all private use inside the notebook only, nothing exported): <you>/verifai-bundle, <you>/verifai-candidates,
reubensuju/celeb-df-v2, philosopher0808/real-vs-ai-generated-faces-dataset, ayushmandatta1/deepdetect-2025.
The output zip holds checkpoints and metrics only: no face crops and no third-party images.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
DIST = ROOT / "kaggle" / "dist" / "e3"

CELLS = [
("md", """# Verif-AI - E3: modern-generator training (GPU)
Pre-registered protocol: docs/E3_PROTOCOL.md. Third-party image datasets are used privately inside this notebook; the results zip contains checkpoints and metrics only.
Selection between E3 candidates uses validation data only; test splits (including `modern_test`) are never used for selection or thresholds."""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout[:600])
import torch
ENV = dict(python=platform.python_version(), torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cuda=str(torch.version.cuda))
print(ENV); assert torch.cuda.is_available(), "Enable a GPU accelerator in the notebook settings."
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])"""),
("code", """cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True); assert cands, "Attach the dataset 'verifai-bundle'."
shutil.copytree(os.path.dirname(os.path.dirname(cands[0])), '/kaggle/working/backend', dirs_exist_ok=True)
os.chdir('/kaggle/working/backend'); sys.path.insert(0, '.'); os.makedirs('/kaggle/working/results', exist_ok=True)
meta = glob.glob('/kaggle/input/**/candidates.json', recursive=True); assert meta; CAND = os.path.dirname(meta[0])
real_dirs = [d for d in glob.glob('/kaggle/input/**/Celeb-real', recursive=True) if os.path.isdir(d)]; assert real_dirs; CDF_ROOT = os.path.dirname(real_dirs[0])
ffhq = glob.glob('/kaggle/input/**/data_source/data_source/ffhq', recursive=True); assert ffhq, "Attach philosopher0808/real-vs-ai-generated-faces-dataset"
PH = os.path.dirname(os.path.dirname(ffhq[0]))                                     # .../data_source/data_source/ffhq -> .../data_source/data_source
SFHQ = sorted(glob.glob(os.path.join(PH, 'fake', 'sfhq', 'pt*')))
dd_tr = glob.glob('/kaggle/input/**/ddata/train/fake', recursive=True); dd_te = glob.glob('/kaggle/input/**/ddata/test/fake', recursive=True)
assert SFHQ and dd_tr and dd_te, (SFHQ, dd_tr, dd_te)
FFHQ, DD_TRAIN, DD_TEST = ffhq[0], dd_tr[0], dd_te[0]
print("roots:", dict(cdf=CDF_ROOT, ffhq=FFHQ, sfhq=SFHQ, dd_train=DD_TRAIN, dd_test=DD_TEST))
assert os.path.isfile(os.path.join(CAND, 'cdf_C1_qa.pth')) and os.path.isfile('models/verifai_v2_best.pth')
open("training/modern_data.py", "w", encoding="utf-8").write(@@MODERN_DATA_SOURCE@@)
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_model_package.py", "tests/test_data_pipeline.py", "tests/test_scoring_and_tracking.py", "tests/test_celebdf.py", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True)
print(r.stdout[-600:])"""),
("code", """from collections import Counter
from training import celebdf
from verifai.config import Settings
from verifai.faces import FaceDetector
vids = celebdf.scan(CDF_ROOT)
plan, split_of = celebdf.plan_videos(vids, seed=42, frames_real=12, frames_test=16)        # identical to the Model 3/4 runs
det = FaceDetector(torch.device("cuda"), Settings())
t0 = time.time(); rows = celebdf.extract(plan, "data/crops_cdf", det, workers=4, seed=42)
print("celeb-df crops:", len(rows), f"{(time.time()-t0)/60:.1f} min")
celebdf.write_manifest(rows, "manifests/celebdf_faces_manifest.csv")
celebdf.merge_manifests("manifests/dff_faces_manifest.csv", "manifests/celebdf_faces_manifest.csv", "manifests/combined_manifest.csv")"""),
("code", """import csv, numpy as np, cv2
from training import modern_data as md
items = md.plan_modern(FFHQ, SFHQ, DD_TRAIN, DD_TEST, seed=42)
print("planned:", Counter((i["split"], i["generator"]) for i in items))
t0 = time.time(); mrows = md.extract(items, "data/crops_modern", det, workers=4)
print("modern crops with a usable face:", len(mrows), f"{(time.time()-t0)/60:.1f} min")
planned, kept = Counter((i["split"], i["generator"]) for i in items), Counter((r["split"], r["generator"]) for r in mrows)
print("face-detection yield:", {f"{k[0]}/{k[1]}": f"{kept.get(k,0)}/{v}" for k, v in planned.items()})
base = list(csv.DictReader(open("manifests/combined_manifest.csv", encoding="utf-8")))
ref_train = [r["crop_path"] for r in base if r["split"] == "train"]                         # DeepFakeFace + Celeb-DF training crops: the new val/test data must not duplicate them
mrows2, removed = md.remove_leaks(mrows, ref_train, max_hamming=3)
print("near-duplicates removed (val / test vs training crops):", removed)
md.write_merged("manifests/combined_manifest.csv", mrows2, "manifests/combined_modern_manifest.csv")
allrows = list(csv.DictReader(open("manifests/combined_modern_manifest.csv", encoding="utf-8")))
comp = Counter((r["split"], r["source"], r["generator"], r["label"]) for r in allrows)
print("final composition:", sorted(comp.items()))
def lap(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE); return float(cv2.Laplacian(g, cv2.CV_64F).var())
import random; random.seed(0); q = {}
for k in sorted({(r["split"], r["generator"]) for r in allrows}):
    ps = [r["crop_path"] for r in allrows if (r["split"], r["generator"]) == k]; v = [lap(p) for p in random.sample(ps, min(80, len(ps)))]
    q[f"{k[0]}/{k[1]}"] = dict(n=len([1 for r in allrows if (r["split"], r["generator"]) == k]), lap_var_p10_50_90=[round(float(x), 1) for x in np.percentile(v, [10, 50, 90])])
print("sharpness of the crops actually used (variance of Laplacian):"); [print(" ", a, b) for a, b in q.items()]
assert all(os.path.isfile(r["crop_path"]) for r in allrows)
json.dump(dict(planned={f"{k[0]}/{k[1]}": v for k, v in planned.items()}, kept={f"{k[0]}/{k[1]}": v for k, v in kept.items()}, leaks_removed=removed,
               composition={"|".join(k): v for k, v in comp.items()}, sharpness=q), open("/kaggle/working/results/e3_data.json", "w"), indent=1)"""),
("code", """RUNS = [("e3a_modern_qa0.5", ["--quality-aug", "0.5"]), ("e3b_modern_noQA", ["--quality-aug", "0.0"])]
COMMON = ["--manifest", "manifests/combined_modern_manifest.csv", "--init", "imagenet", "--select", "clean_degraded_mean", "--epochs", "12", "--patience", "4",
          "--batch-size", "32", "--workers", "2", "--lr", "1e-4", "--seed", "42", "--out", "/kaggle/working/runs"]
for name, extra in RUNS:
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", "-m", "training.train", "--name", name, *COMMON, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith("==") or "early stopping" in line or "Traceback" in line or "Error" in line or "done." in line:
            print(name, line.rstrip())
    p.wait(); print(name, "exit", p.returncode, f"{(time.time()-t0)/60:.1f} min"); assert p.returncode == 0"""),
("code", """CANDS = [(n, f"/kaggle/working/runs/{n}/best.pth") for n, _ in RUNS] + [("model4", os.path.join(CAND, "cdf_C1_qa.pth")), ("v2", "models/verifai_v2_best.pth")]
CONDS = ["none", "mixed"]
SPLITS = ["--split", "val:validation_used_for_selection", "--split", "test_internal:dff_internal_test", "--split", "test_crossgen:dff_cross_generator",
          "--split", "cdf_test:celebdf_heldout_identities", "--split", "modern_test:modern_generators_file_disjoint"]
for name, ck in CANDS:
    for cond in CONDS:
        out = f"/kaggle/working/evals/{name}/{cond}"
        args = ["--degrade", cond] if cond != "none" else []
        r = subprocess.run([sys.executable, "-m", "training.evaluate", "--checkpoint", ck, "--manifest", "manifests/combined_modern_manifest.csv", *SPLITS,
                            "--out", out, "--boot", "300", "--no-plots", *args], capture_output=True, text=True)
        print(name, cond, "ok" if r.returncode == 0 else r.stderr[-400:]); assert r.returncode == 0"""),
("code", """# SELECTION AMONG THE E3 CANDIDATES USES VALIDATION ONLY. Model 4 and v2 are references, not candidates.
table, best = [], None
def met(name, cond, split, video=False):
    m = json.load(open(f"/kaggle/working/evals/{name}/{cond}/metrics.json"))["splits"][split]; return m["video_level"] if video else m
for name, _ in CANDS:
    row = dict(run=name)
    if name.startswith("e3"):
        s = json.load(open(f"/kaggle/working/runs/{name}/summary.json")); row.update(best_epoch=s["best_epoch"], val_select=s["best_val_auc"])
    for cond in CONDS:
        for sp in ("val", "test_internal", "test_crossgen", "cdf_test", "modern_test"):
            m = met(name, cond, sp)
            row[f"{sp}_{cond}"] = {k: m[k] for k in ("n", "accuracy", "roc_auc", "false_positive_rate", "false_negative_rate") if k in m}
            if "per_generator" in m: row[f"{sp}_{cond}_per_generator"] = m["per_generator"]
        row[f"cdf_test_{cond}_VIDEO"] = {k: met(name, cond, "cdf_test", True).get(k) for k in ("n", "accuracy", "roc_auc", "false_positive_rate", "false_negative_rate")}
    table.append(row)
    if name.startswith("e3") and (best is None or row["val_select"] > best["val_select"]): best = row
for r in table: print(r["run"], "| modern_test none:", r["modern_test_none"], "| cdf VIDEO none:", r["cdf_test_none_VIDEO"])
print("SELECTED (validation only):", best["run"])
json.dump(dict(selected=best["run"], criterion="max mean(ROC-AUC clean val, ROC-AUC degraded val) on the combined validation split", table=table, env=ENV), open("/kaggle/working/results/e3_selection.json", "w"), indent=2)"""),
("code", """import zipfile
out = "/kaggle/working/verifai_e3_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, _ in RUNS:
        base = f"/kaggle/working/runs/{name}"
        for f in ["best.pth", "history.csv", "history.json", "config.json", "summary.json"]:
            z.write(f"{base}/{f}", f"runs/{name}/{f}")
    for f in glob.glob("/kaggle/working/evals/*/*/metrics.json"): z.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/results/*"): z.write(f, f"results/{os.path.basename(f)}")
    z.write("manifests/combined_modern_manifest.csv", "results/combined_modern_manifest.csv")
print(out, round(os.path.getsize(out) / 1e6, 1), "MB (checkpoints + metrics + manifest paths only; no images or crops)")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)                        # third-party crops are not part of the notebook output
for name, _ in RUNS:
    for f in ("last.pth",):
        try: os.remove(f"/kaggle/working/runs/{name}/{f}")
        except FileNotFoundError: pass
    shutil.rmtree(f"/kaggle/working/runs/{name}/tb", ignore_errors=True)"""),
]


def notebook() -> dict:
    src_md = repr((BACKEND / "training" / "modern_data.py").read_text(encoding="utf-8"))
    cells = []
    for kind, src in CELLS:
        src = src.replace("@@MODERN_DATA_SOURCE@@", src_md)
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
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    (DIST / "e3.ipynb").write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    (DIST / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.username}/verif-ai-e3", "title": "Verif-AI E3", "code_file": "e3.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{a.username}/verifai-bundle", f"{a.username}/verifai-candidates", "reubensuju/celeb-df-v2",
                            "philosopher0808/real-vs-ai-generated-faces-dataset", "ayushmandatta1/deepdetect-2025"],
        "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
    print("kernel folder:", DIST)


if __name__ == "__main__":
    main()
