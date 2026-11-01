"""Build the Kaggle upload package for GPU training (dataset folder + notebook + metadata).

  python kaggle/build_kaggle_bundle.py --username <your-kaggle-username>

Creates kaggle/dist/dataset/  (private dataset: code, tests, manifests, EXACT local face crops, original checkpoint)
        kaggle/dist/kernel/   (notebook + kernel-metadata.json, GPU + internet enabled)
Then (see kaggle/README.md):
  kaggle datasets create -p kaggle/dist/dataset --dir-mode zip
  kaggle kernels push    -p kaggle/dist/kernel
The dataset contains face images (public research dataset, openrail): keep it PRIVATE.
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
("md", """# Verif-AI - GPU training on Kaggle
Trains the two-stream EfficientNet-B0 + SRM model on the validated, identity-disjoint DeepFakeFace sample (same crops/manifests as the local pipeline),
compares three configurations **using validation data only for selection**, evaluates all on the held-out test splits, benchmarks GPU inference and packages
everything into `verifai_results.zip` (download it from the notebook output and place it next to the project).
Settings: *Accelerator = GPU, Internet = On, attach dataset `verifai-bundle`.*"""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout[:900])
import torch
ENV = dict(python=platform.python_version(), torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cuda=str(torch.version.cuda))
print(ENV)
assert torch.cuda.is_available(), "Enable a GPU accelerator in the notebook settings (Settings > Accelerator)."
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])"""),
("code", """# Kaggle unpacks backend.zip at the dataset root, so the package may sit at <dataset>/verifai or <dataset>/backend/verifai.
cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True)
if not cands:
    print(subprocess.run("find /kaggle/input -maxdepth 4 | head -50", shell=True, capture_output=True, text=True).stdout)
assert cands, "Attach the dataset 'verifai-bundle' to this notebook."
src = os.path.dirname(os.path.dirname(cands[0]))
shutil.copytree(src, '/kaggle/working/backend', dirs_exist_ok=True)
os.chdir('/kaggle/working/backend'); sys.path.insert(0, '.')
os.makedirs('/kaggle/working/results', exist_ok=True)
import csv, hashlib
rows = list(csv.DictReader(open('manifests/dff_faces_manifest.csv', encoding='utf-8')))
from collections import Counter
print(Counter((r['split'], r['label']) for r in rows))
assert all(os.path.isfile(r['crop_path']) for r in rows), "crop files missing from the bundle"
print("manifest sha256:", hashlib.sha256(open('manifests/dff_faces_manifest.csv','rb').read()).hexdigest())"""),
("code", """# unit tests that run on the GPU too (includes the CUDA inference test)
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_model_package.py", "tests/test_data_pipeline.py", "tests/test_scoring_and_tracking.py", "-q", "-p", "no:cacheprovider", "-rs"],
                   capture_output=True, text=True)
print(r.stdout[-2500:]); open('/kaggle/working/results/pytest_gpu.log', 'w').write(r.stdout + r.stderr)"""),
("code", """RUNS = [
    ("v2_A_origInit",        ["--init", "scripts/two_stream_model_epoch_10.pth"]),   # fine-tune the existing checkpoint (primary)
    ("v2_B_imagenetInit",    ["--init", "imagenet"]),                                # ablation: ImageNet initialisation
    ("v2_C_origInit_noAug",  ["--init", "scripts/two_stream_model_epoch_10.pth", "--no-aug"]),   # ablation: no augmentation
]
COMMON = ["--manifest", "manifests/dff_faces_manifest.csv", "--epochs", "15", "--patience", "4", "--batch-size", "32",
          "--workers", "2", "--lr", "1e-4", "--seed", "42", "--out", "/kaggle/working/runs"]
for name, extra in RUNS:
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", "-m", "training.train", "--name", name, *COMMON, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith("==") or "early stopping" in line or "Traceback" in line or "Error" in line or "done." in line:
            print(name, line.rstrip())
    p.wait(); print(name, "exit", p.returncode, f"{(time.time()-t0)/60:.1f} min")
    assert p.returncode == 0"""),
("code", """for name, _ in RUNS:
    ck = f"/kaggle/working/runs/{name}/best.pth"
    r = subprocess.run([sys.executable, "-m", "training.evaluate", "--checkpoint", ck, "--manifest", "manifests/dff_faces_manifest.csv",
                        "--split", "val:validation_used_for_selection", "--split", "test_internal:internal_test_unseen_identities",
                        "--split", "test_crossgen:cross_generator_unseen_generator_and_identities", "--out", f"/kaggle/working/runs/{name}/eval", "--boot", "1000"],
                       capture_output=True, text=True)
    print(name, r.stdout[-700:], r.stderr[-300:] if r.returncode else "")
    assert r.returncode == 0"""),
("code", """# MODEL SELECTION USES VALIDATION ONLY. Test numbers are printed for every run for transparency, never for choosing.
table, best = [], None
for name, _ in RUNS:
    s = json.load(open(f"/kaggle/working/runs/{name}/summary.json")); m = json.load(open(f"/kaggle/working/runs/{name}/eval/metrics.json"))["splits"]
    row = dict(run=name, best_epoch=s["best_epoch"], val_auc=s["best_val_auc"], val_acc=m["val"]["accuracy"],
               test_internal_acc=m["test_internal"]["accuracy"], test_internal_auc=m["test_internal"]["roc_auc"],
               test_crossgen_acc=m["test_crossgen"]["accuracy"], test_crossgen_auc=m["test_crossgen"]["roc_auc"])
    table.append(row)
    if best is None or row["val_auc"] > best["val_auc"]:
        best = row
for r in table: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print("SELECTED (by validation ROC-AUC only):", best["run"])
json.dump(dict(selected=best["run"], criterion="best validation ROC-AUC", table=table, env=ENV), open("/kaggle/working/results/selection.json", "w"), indent=2)"""),
("code", """# GPU inference benchmark of the selected model (real measurement on this GPU)
from verifai.model import load_model
b = load_model(f"/kaggle/working/runs/{best['run']}/best.pth", "cuda")
x = torch.randn(32, 3, 224, 224)
for _ in range(5): b.predict_batch(x)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time(); n = 40
for _ in range(n): b.predict_batch(x)
torch.cuda.synchronize(); dt = time.time() - t
bench = dict(gpu=ENV["gpu"], batch=32, images_per_second=round(32 * n / dt, 1), ms_per_batch=round(1000 * dt / n, 1), peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6))
print(bench); json.dump(bench, open("/kaggle/working/results/gpu_benchmark.json", "w"), indent=2)"""),
("code", """import zipfile
out = "/kaggle/working/verifai_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, _ in RUNS:
        base = f"/kaggle/working/runs/{name}"
        for f in ["best.pth", "history.csv", "history.json", "config.json", "summary.json"]:
            z.write(f"{base}/{f}", f"runs/{name}/{f}")
        for f in glob.glob(f"{base}/eval/*"):
            z.write(f, f"runs/{name}/eval/{os.path.basename(f)}")
    for f in glob.glob("/kaggle/working/results/*"):
        z.write(f, f"results/{os.path.basename(f)}")
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)        # keep the notebook output small
for name, _ in RUNS:
    for f in ("last.pth",):
        try: os.remove(f"/kaggle/working/runs/{name}/{f}")
        except FileNotFoundError: pass
    shutil.rmtree(f"/kaggle/working/runs/{name}/tb", ignore_errors=True)"""),
]


def notebook() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        cells.append(c)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True, help="your Kaggle username (the dataset/kernel owner)")
    ap.add_argument("--dataset-slug", default="verifai-bundle")
    ap.add_argument("--kernel-slug", default="verif-ai-gpu-training")
    a = ap.parse_args()

    if DIST.exists():
        shutil.rmtree(DIST)
    ds, kn = DIST / "dataset", DIST / "kernel"
    (ds / "backend").mkdir(parents=True); kn.mkdir(parents=True)
    ig = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("verifai", "training"):
        shutil.copytree(BACKEND / d, ds / "backend" / d, ignore=ig)
    (ds / "backend/tests").mkdir()
    for f in ("conftest.py", "test_model_package.py", "test_data_pipeline.py", "test_scoring_and_tracking.py"):
        shutil.copy(BACKEND / "tests" / f, ds / "backend/tests" / f)
    (ds / "backend/scripts").mkdir()
    for f in ("verifai_inference.py", "two_stream_model_epoch_10.pth"):
        shutil.copy(BACKEND / "scripts" / f, ds / "backend/scripts" / f)
    shutil.copytree(BACKEND / "manifests", ds / "backend/manifests")
    shutil.copytree(BACKEND / "data/crops", ds / "backend/data/crops")
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "verifai-bundle", "id": f"{a.username}/{a.dataset_slug}", "licenses": [{"name": "other"}],
        "subtitle": "Verif-AI training code, manifests and face crops (DeepFakeFace sample, openrail)",
        "description": "Private bundle for GPU training. Face crops come from the DeepFakeFace dataset (openrail). Do not make public."}, indent=2))
    (kn / "verifai_train_kaggle.ipynb").write_text(json.dumps(notebook(), indent=1))
    (kn / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.username}/{a.kernel_slug}", "title": "Verif-AI GPU training", "code_file": "verifai_train_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{a.username}/{a.dataset_slug}"], "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    size = sum(p.stat().st_size for p in ds.rglob("*") if p.is_file()) / 1e6
    print(f"dataset folder: {ds}  ({size:.0f} MB, {sum(1 for p in ds.rglob('*') if p.is_file())} files)\nkernel folder : {kn}")


if __name__ == "__main__":
    main()
