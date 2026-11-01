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


# ---------------------------------------------------------------------------------------------------------------------
# Experiment "robust": retrain with CLASS-BLIND quality degradation (see docs/VIDEO_FALSE_POSITIVE_INVESTIGATION.md).
# Three runs, all ImageNet-initialised (as the selected v2), same seed/splits:
#   R0 control   : no quality augmentation, but the new selection criterion  -> isolates the effect of the augmentation
#   R1 p=0.5     : quality augmentation on half of the training crops
#   R2 p=0.8     : quality augmentation on 80 % of the training crops
# Selection (fixed BEFORE running, validation data only): highest mean of ROC-AUC on clean val and on deterministically degraded val.
CELLS_ROBUST = [
("md", """# Verif-AI - robustness retraining on Kaggle GPU
Retrains the two-stream model with class-blind quality degradation (blur / down-up-scaling / JPEG / noise / low light) to remove the
"soft face => FAKE" shortcut found in the video false-positive investigation. Selection uses **validation data only**:
mean ROC-AUC on clean and on deterministically degraded validation crops. Test splits are evaluated afterwards for reporting only."""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout[:900])
import torch
ENV = dict(python=platform.python_version(), torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cuda=str(torch.version.cuda))
print(ENV)
assert torch.cuda.is_available(), "Enable a GPU accelerator in the notebook settings."
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])"""),
("code", """cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True)
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
("code", """r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_model_package.py", "tests/test_data_pipeline.py", "tests/test_scoring_and_tracking.py", "tests/test_degrade.py", "-q", "-p", "no:cacheprovider", "-rs"],
                   capture_output=True, text=True)
print(r.stdout[-2500:]); open('/kaggle/working/results/pytest_gpu.log', 'w').write(r.stdout + r.stderr)"""),
("code", """RUNS = [
    ("v3_R0_control_noQualityAug", ["--quality-aug", "0.0"]),
    ("v3_R1_qualityAug_p0.5",      ["--quality-aug", "0.5"]),
    ("v3_R2_qualityAug_p0.8",      ["--quality-aug", "0.8"]),
]
COMMON = ["--manifest", "manifests/dff_faces_manifest.csv", "--init", "imagenet", "--select", "clean_degraded_mean", "--epochs", "15", "--patience", "4",
          "--batch-size", "32", "--workers", "2", "--lr", "1e-4", "--seed", "42", "--out", "/kaggle/working/runs"]
for name, extra in RUNS:
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", "-m", "training.train", "--name", name, *COMMON, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith("==") or "early stopping" in line or "Traceback" in line or "Error" in line or "done." in line:
            print(name, line.rstrip())
    p.wait(); print(name, "exit", p.returncode, f"{(time.time()-t0)/60:.1f} min")
    assert p.returncode == 0"""),
("code", """CONDS = ["none", "mixed", "blur_s2.5", "downup_0.3", "phone_soft"]
for name, _ in RUNS:
    ck = f"/kaggle/working/runs/{name}/best.pth"
    for cond in CONDS:
        out = f"/kaggle/working/runs/{name}/eval_{cond}"
        args = ["--degrade", cond] if cond != "none" else []
        r = subprocess.run([sys.executable, "-m", "training.evaluate", "--checkpoint", ck, "--manifest", "manifests/dff_faces_manifest.csv",
                            "--split", "val:validation_used_for_selection", "--split", "test_internal:internal_test_unseen_identities",
                            "--split", "test_crossgen:cross_generator_unseen_generator_and_identities", "--out", out, "--boot", "300", "--no-plots", *args],
                           capture_output=True, text=True)
        print(name, cond, r.stdout.replace("\\n", " | ")[-420:], r.stderr[-200:] if r.returncode else "")
        assert r.returncode == 0"""),
("code", """# MODEL SELECTION USES VALIDATION ONLY (mean of clean and degraded validation ROC-AUC, computed during training).
table, best = [], None
for name, _ in RUNS:
    s = json.load(open(f"/kaggle/working/runs/{name}/summary.json"))
    row = dict(run=name, best_epoch=s["best_epoch"], val_select=s["best_val_auc"])
    for cond in CONDS:
        m = json.load(open(f"/kaggle/working/runs/{name}/eval_{cond}/metrics.json"))["splits"]
        for sp in ("val", "test_internal", "test_crossgen"):
            row[f"{sp}_{cond}_auc"] = m[sp]["roc_auc"]; row[f"{sp}_{cond}_fpr"] = m[sp]["false_positive_rate"]; row[f"{sp}_{cond}_fnr"] = m[sp]["false_negative_rate"]
    table.append(row)
    if best is None or row["val_select"] > best["val_select"]:
        best = row
for r in table: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("run", "best_epoch", "val_select", "val_none_auc", "val_mixed_auc", "test_internal_none_auc", "test_internal_mixed_auc", "test_crossgen_none_auc", "test_crossgen_mixed_auc")})
print("SELECTED (validation only):", best["run"])
json.dump(dict(selected=best["run"], criterion="max mean(ROC-AUC clean val, ROC-AUC degraded val)", table=table, env=ENV), open("/kaggle/working/results/selection.json", "w"), indent=2)"""),
("code", """from verifai.model import load_model
b = load_model(f"/kaggle/working/runs/{best['run']}/best.pth", "cuda")
x = torch.randn(32, 3, 224, 224)
for _ in range(5): b.predict_batch(x)
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.time(); n = 40
for _ in range(n): b.predict_batch(x)
torch.cuda.synchronize(); dt = time.time() - t
bench = dict(gpu=ENV["gpu"], batch=32, images_per_second=round(32 * n / dt, 1), ms_per_batch=round(1000 * dt / n, 1), peak_gpu_mem_mb=round(torch.cuda.max_memory_allocated() / 1e6))
print(bench); json.dump(bench, open("/kaggle/working/results/gpu_benchmark.json", "w"), indent=2)"""),
("code", """import zipfile
out = "/kaggle/working/verifai_robust_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, _ in RUNS:
        base = f"/kaggle/working/runs/{name}"
        for f in ["best.pth", "history.csv", "history.json", "config.json", "summary.json"]:
            z.write(f"{base}/{f}", f"runs/{name}/{f}")
        for f in glob.glob(f"{base}/eval_*/metrics.json"):
            z.write(f, f"runs/{name}/{os.path.basename(os.path.dirname(f))}/metrics.json")
    for f in glob.glob("/kaggle/working/results/*"):
        z.write(f, f"results/{os.path.basename(f)}")
print(out, round(os.path.getsize(out) / 1e6, 1), "MB")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)
for name, _ in RUNS:
    try: os.remove(f"/kaggle/working/runs/{name}/last.pth")
    except FileNotFoundError: pass
    shutil.rmtree(f"/kaggle/working/runs/{name}/tb", ignore_errors=True)"""),
]


# ---------------------------------------------------------------------------------------------------------------------
# Experiment "celebdf": train on DeepFakeFace + Celeb-DF (v2) frames (identity-disjoint), evaluate frame- AND video-level on held-out Celeb-DF
# identities, compare with the current production model v2 inside the same notebook (no Celeb-DF data leaves Kaggle).
CELLS_CELEBDF = [
("md", """# Verif-AI - training with Celeb-DF (v2) on Kaggle GPU
Celeb-DF (v2) is attached as a Kaggle dataset (`reubensuju/celeb-df-v2`, a third-party copy; official terms: research use only). Nothing from it is exported:
the results zip contains checkpoints and metrics only. Identity-disjoint train / val / test, fakes that cross splits are dropped, test is evaluated at video level
and never used for selection."""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
print(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout[:900])
import torch
ENV = dict(python=platform.python_version(), torch=str(torch.__version__), cuda_available=torch.cuda.is_available(),
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cuda=str(torch.version.cuda))
print(ENV)
assert torch.cuda.is_available(), "Enable a GPU accelerator in the notebook settings."
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])"""),
("code", """cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True)
assert cands, "Attach the dataset 'verifai-bundle' to this notebook."
src = os.path.dirname(os.path.dirname(cands[0]))
shutil.copytree(src, '/kaggle/working/backend', dirs_exist_ok=True)
os.chdir('/kaggle/working/backend'); sys.path.insert(0, '.')
os.makedirs('/kaggle/working/results', exist_ok=True)
real_dirs = [d for d in glob.glob('/kaggle/input/**/Celeb-real', recursive=True) if os.path.isdir(d)]
assert real_dirs, "Attach the Kaggle dataset 'reubensuju/celeb-df-v2' to this notebook."
CDF_ROOT = os.path.dirname(real_dirs[0]); print("Celeb-DF root:", CDF_ROOT, os.listdir(CDF_ROOT)[:10])
assert os.path.isfile('models/verifai_v2_best.pth'), "v2 checkpoint missing from the bundle"
r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_model_package.py", "tests/test_data_pipeline.py", "tests/test_scoring_and_tracking.py", "tests/test_degrade.py", "tests/test_celebdf.py", "-q", "-p", "no:cacheprovider", "-rs"],
                   capture_output=True, text=True)
print(r.stdout[-1500:]); open('/kaggle/working/results/pytest_gpu.log', 'w').write(r.stdout + r.stderr)"""),
("code", """from collections import Counter
from training import celebdf
from verifai.config import Settings
from verifai.faces import FaceDetector
vids = celebdf.scan(CDF_ROOT)
print({k: len(v) for k, v in vids.items()})
plan, split_of = celebdf.plan_videos(vids, seed=42, frames_real=12, frames_test=16)
print("videos planned:", Counter((v["split"], v["source"]) for v in plan))
print("identities per split:", Counter(split_of.values()))
det = FaceDetector(torch.device("cuda"), Settings())
t0 = time.time()
rows = celebdf.extract(plan, "data/crops_cdf", det, workers=4, seed=42)
print("crops:", len(rows), f"{(time.time()-t0)/60:.1f} min")
celebdf.write_manifest(rows, "manifests/celebdf_faces_manifest.csv")
celebdf.merge_manifests("manifests/dff_faces_manifest.csv", "manifests/celebdf_faces_manifest.csv", "manifests/combined_manifest.csv")
import csv
mrows = list(csv.DictReader(open("manifests/combined_manifest.csv", encoding="utf-8")))
print("combined:", Counter((r["split"], r["source"], r["label"]) for r in mrows))
assert all(os.path.isfile(r["crop_path"]) for r in mrows)
json.dump(dict(videos={k: len(v) for k, v in vids.items()}, crops=Counter((r["split"], r["source"], r["label"]) for r in mrows).__repr__()), open("/kaggle/working/results/celebdf_extraction.json", "w"))"""),
("code", """RUNS = [
    ("v4_C0_celebdf_dff_noQualityAug", ["--quality-aug", "0.0"]),
    ("v4_C1_celebdf_dff_qualityAug0.5", ["--quality-aug", "0.5"]),
]
COMMON = ["--manifest", "manifests/combined_manifest.csv", "--init", "imagenet", "--select", "clean_degraded_mean", "--epochs", "12", "--patience", "4",
          "--batch-size", "32", "--workers", "2", "--lr", "1e-4", "--seed", "42", "--out", "/kaggle/working/runs"]
for name, extra in RUNS:
    t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", "-m", "training.train", "--name", name, *COMMON, *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if line.startswith("==") or "early stopping" in line or "Traceback" in line or "Error" in line or "done." in line:
            print(name, line.rstrip())
    p.wait(); print(name, "exit", p.returncode, f"{(time.time()-t0)/60:.1f} min")
    assert p.returncode == 0"""),
("code", """CANDS = [(n, f"/kaggle/working/runs/{n}/best.pth") for n, _ in RUNS] + [("v2_production", "models/verifai_v2_best.pth")]
CONDS = ["none", "mixed"]
SPLITS = ["--split", "val:validation_used_for_selection", "--split", "test_internal:dff_internal_test", "--split", "test_crossgen:dff_cross_generator",
          "--split", "cdf_test:celebdf_heldout_identities"]
for name, ck in CANDS:
    for cond in CONDS:
        out = f"/kaggle/working/evals/{name}/{cond}"
        args = ["--degrade", cond] if cond != "none" else []
        r = subprocess.run([sys.executable, "-m", "training.evaluate", "--checkpoint", ck, "--manifest", "manifests/combined_manifest.csv", *SPLITS,
                            "--out", out, "--boot", "300", "--no-plots", *args], capture_output=True, text=True)
        print(name, cond, r.stdout.replace("\\n", " | ")[-900:], r.stderr[-300:] if r.returncode else "")
        assert r.returncode == 0"""),
("code", """# MODEL SELECTION USES VALIDATION ONLY. v2_production is shown for comparison and is not a candidate.
table, best = [], None
def g(name, cond, split, key, video=False):
    m = json.load(open(f"/kaggle/working/evals/{name}/{cond}/metrics.json"))["splits"][split]
    m = m["video_level"] if video else m
    return m[key]
for name, _ in CANDS:
    row = dict(run=name)
    if name != "v2_production":
        s = json.load(open(f"/kaggle/working/runs/{name}/summary.json")); row.update(best_epoch=s["best_epoch"], val_select=s["best_val_auc"])
    for cond in CONDS:
        for sp in ("val", "test_internal", "test_crossgen", "cdf_test"):
            for key in ("accuracy", "roc_auc", "false_positive_rate", "false_negative_rate"):
                row[f"{sp}_{cond}_{key}"] = g(name, cond, sp, key)
        for key in ("accuracy", "roc_auc", "false_positive_rate", "false_negative_rate"):
            row[f"cdf_test_{cond}_VIDEO_{key}"] = g(name, cond, "cdf_test", key, video=True)
    table.append(row)
    if name != "v2_production" and (best is None or row["val_select"] > best["val_select"]):
        best = row
for r in table: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if "VIDEO_roc" in k or "VIDEO_acc" in k or "VIDEO_false" in k or k in ("run", "val_select", "best_epoch")})
print("SELECTED (validation only):", best["run"])
json.dump(dict(selected=best["run"], criterion="max mean(ROC-AUC clean val, ROC-AUC degraded val) on the combined validation split", table=table, env=ENV), open("/kaggle/working/results/selection.json", "w"), indent=2)"""),
("code", """import zipfile
out = "/kaggle/working/verifai_celebdf_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for name, _ in RUNS:
        base = f"/kaggle/working/runs/{name}"
        for f in ["best.pth", "history.csv", "history.json", "config.json", "summary.json"]:
            z.write(f"{base}/{f}", f"runs/{name}/{f}")
    for f in glob.glob("/kaggle/working/evals/*/*/metrics.json"):
        z.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/results/*"):
        z.write(f, f"results/{os.path.basename(f)}")
print(out, round(os.path.getsize(out) / 1e6, 1), "MB  (checkpoints + metrics only; no Celeb-DF frames or crops are exported)")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)       # crops of the third-party dataset are removed from the notebook output
for name, _ in RUNS:
    for f in ("last.pth",):
        try: os.remove(f"/kaggle/working/runs/{name}/{f}")
        except FileNotFoundError: pass
    shutil.rmtree(f"/kaggle/working/runs/{name}/tb", ignore_errors=True)"""),
]


# ---------------------------------------------------------------------------------------------------------------------
# Experiment "analysis": the FOUR-MODEL comparison under identical code (no training). Celeb-DF test crops are re-extracted inside Kaggle;
# nothing from Celeb-DF leaves the notebook (the zip holds metrics only).
CELLS_ANALYSIS = [
("md", """# Verif-AI - four-model comparison (no training)
1 production v2 | 2 DFF-only + quality augmentation | 3 DFF + Celeb-DF, no augmentation | 4 DFF + Celeb-DF + quality augmentation.
Same test crops, same code, same thresholds (0.5). Clean and degraded conditions; frame-level and video-level (Celeb-DF); RGB-only / SRM-only / combined outputs."""),
("code", """import subprocess, sys, os, shutil, glob, json, time, platform
import torch
ENV = dict(python=platform.python_version(), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print(ENV)
assert torch.cuda.is_available()
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "efficientnet-pytorch==0.7.1", "facenet-pytorch==2.5.3"])
cands = glob.glob('/kaggle/input/**/verifai/__init__.py', recursive=True); assert cands
src = os.path.dirname(os.path.dirname(cands[0]))
shutil.copytree(src, '/kaggle/working/backend', dirs_exist_ok=True)
os.chdir('/kaggle/working/backend'); sys.path.insert(0, '.')
meta = glob.glob('/kaggle/input/**/candidates.json', recursive=True); assert meta, "attach dataset verifai-candidates"
CAND = os.path.dirname(meta[0]); print("candidates:", os.listdir(CAND))
open("training/stream_ablation.py", "w", encoding="utf-8").write(@@STREAM_ABLATION_SOURCE@@)      # corrected module embedded in this notebook
real_dirs = [d for d in glob.glob('/kaggle/input/**/Celeb-real', recursive=True) if os.path.isdir(d)]; assert real_dirs
CDF_ROOT = os.path.dirname(real_dirs[0]); print("Celeb-DF root:", CDF_ROOT)
os.makedirs('/kaggle/working/results', exist_ok=True)"""),
("code", """from collections import Counter
from training import celebdf
from verifai.config import Settings
from verifai.faces import FaceDetector
vids = celebdf.scan(CDF_ROOT)
plan, split_of = celebdf.plan_videos(vids, seed=42, frames_real=12, frames_test=16)      # identity split identical to the training run (seed 42)
plan = [v for v in plan if v["split"] == "cdf_test"]
print("test videos:", Counter((v["source"]) for v in plan), "identities in test:", sorted({v["identity"] for v in plan})[:5], "...")
det = FaceDetector(torch.device("cuda"), Settings())
t0 = time.time()
rows = celebdf.extract(plan, "/tmp/crops_cdf", det, workers=4, seed=42)          # /tmp: never part of the notebook output
print("test crops:", len(rows), f"{(time.time()-t0)/60:.1f} min", Counter((r["source"]) for r in rows))
CROP_COUNTS = Counter((r["source"]) for r in rows)
celebdf.write_manifest(rows, "manifests/celebdf_test_manifest.csv")
celebdf.merge_manifests("manifests/dff_faces_manifest.csv", "manifests/celebdf_test_manifest.csv", "manifests/analysis_manifest.csv")"""),
("code", """MODELS = [
    ("1_production_v2", "models/verifai_v2_best.pth"),
    ("2_dff_only_qualityAug", os.path.join(CAND, "dff_qa_R1.pth")),
    ("3_dff_celebdf_noQualityAug", os.path.join(CAND, "cdf_C0_noQA.pth")),
    ("4_dff_celebdf_qualityAug", os.path.join(CAND, "cdf_C1_qa.pth")),
]
CONDS = ["none", "mixed", "blur_s2.5", "downup_0.3", "phone_soft"]
SPLITS = ["--split", "test_internal:dff_internal_test", "--split", "test_crossgen:dff_cross_generator", "--split", "cdf_test:celebdf_heldout_identities"]
for name, ck in MODELS:
    for cond in CONDS:
        args = ["--degrade", cond] if cond != "none" else []
        r = subprocess.run([sys.executable, "-m", "training.evaluate", "--checkpoint", ck, "--manifest", "manifests/analysis_manifest.csv", *SPLITS,
                            "--out", f"/kaggle/working/evals/{name}/{cond}", "--boot", "300", "--no-plots", *args], capture_output=True, text=True)
        print(name, cond, r.stdout.replace("\\n", " | ")[-300:], r.stderr[-200:] if r.returncode else "")
        assert r.returncode == 0"""),
("code", """from training import stream_ablation
ABL = {}
for name, ck in MODELS:
    try:
        ABL[name] = stream_ablation.run(ck, "manifests/analysis_manifest.csv", ["test_internal", "test_crossgen", "cdf_test"], ["none", "mixed"], ref_split="train", n_ref=150,
                                        n_examples=8, seed=0, device="cuda", ref_manifest="manifests/dff_faces_manifest.csv")
        print(name, {k: {v: round(m["roc_auc"], 3) for v, m in b["variants"].items()} for k, b in ABL[name]["results"].items() if k.endswith("/none")})
    except Exception as exc:                      # the metrics are already saved; a failure here must not lose them
        ABL[name] = dict(error=repr(exc)); print(name, "ABLATION FAILED:", repr(exc))
json.dump(ABL, open("/kaggle/working/results/stream_ablation.json", "w"), indent=1)"""),
("code", """import zipfile
rows = []
for name, _ in MODELS:
    for cond in CONDS:
        m = json.load(open(f"/kaggle/working/evals/{name}/{cond}/metrics.json"))["splits"]
        for sp, b in m.items():
            rows.append(dict(model=name, condition=cond, split=sp, level="frame", **{k: b[k] for k in ("n", "accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")}, confusion=b["confusion_matrix"], boot=b["bootstrap"]))
            if "video_level" in b:
                v = b["video_level"]
                rows.append(dict(model=name, condition=cond, split=sp, level="video", **{k: v[k] for k in ("n", "accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate")}, confusion=v["confusion_matrix"]))
json.dump(dict(rows=rows, env=ENV, test_crops=repr(CROP_COUNTS)), open("/kaggle/working/results/comparison.json", "w"), indent=1, default=str)
out = "/kaggle/working/verifai_analysis_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("/kaggle/working/evals/*/*/metrics.json"):
        z.write(f, os.path.relpath(f, "/kaggle/working"))
    for f in glob.glob("/kaggle/working/results/*"):
        z.write(f, f"results/{os.path.basename(f)}")
print(out, round(os.path.getsize(out) / 1e6, 2), "MB (metrics only; no Celeb-DF frames or crops are exported)")
shutil.rmtree("/kaggle/working/backend/data", ignore_errors=True)"""),
]


def notebook(cell_list=None) -> dict:
    cells = []
    for kind, src in (cell_list or CELLS):
        if "@@STREAM_ABLATION_SOURCE@@" in src:
            src = src.replace("@@STREAM_ABLATION_SOURCE@@", repr((BACKEND / "training/stream_ablation.py").read_text(encoding="utf-8")))
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
    ap.add_argument("--kernel-slug", default="")
    ap.add_argument("--experiment", choices=["v2", "robust", "celebdf", "analysis"], default="v2", help="v2 = original three-run comparison; robust = quality-augmentation retraining")
    a = ap.parse_args()
    robust = a.experiment == "robust"
    celeb = a.experiment in ("celebdf", "analysis")
    analysis = a.experiment == "analysis"
    kernel_slug = a.kernel_slug or {"robust": "verif-ai-robust-training", "celebdf": "verif-ai-celebdf-training", "analysis": "verif-ai-analysis"}.get(a.experiment, "verif-ai-gpu-training")
    kernel_title = {"robust": "Verif-AI robust training", "celebdf": "Verif-AI celebdf training", "analysis": "Verif-AI analysis"}.get(a.experiment, "Verif-AI GPU training")

    if DIST.exists():
        shutil.rmtree(DIST)
    ds, kn = DIST / "dataset", DIST / "kernel"
    (ds / "backend").mkdir(parents=True); kn.mkdir(parents=True)
    ig = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("verifai", "training"):
        shutil.copytree(BACKEND / d, ds / "backend" / d, ignore=ig)
    (ds / "backend/tests").mkdir()
    for f in ("conftest.py", "test_model_package.py", "test_data_pipeline.py", "test_scoring_and_tracking.py", "test_degrade.py", "test_training_smoke.py", "test_celebdf.py"):
        shutil.copy(BACKEND / "tests" / f, ds / "backend/tests" / f)
    (ds / "backend/models").mkdir()
    shutil.copy(BACKEND / "models" / "verifai_v2_best.pth", ds / "backend/models" / "verifai_v2_best.pth")      # production model, for same-notebook comparison
    (ds / "backend/scripts").mkdir()
    for f in ("verifai_inference.py", "two_stream_model_epoch_10.pth"):
        shutil.copy(BACKEND / "scripts" / f, ds / "backend/scripts" / f)
    shutil.copytree(BACKEND / "manifests", ds / "backend/manifests")
    shutil.copytree(BACKEND / "data/crops", ds / "backend/data/crops")
    (ds / "dataset-metadata.json").write_text(json.dumps({
        "title": "verifai-bundle", "id": f"{a.username}/{a.dataset_slug}", "licenses": [{"name": "other"}],
        "subtitle": "Verif-AI training code, manifests and face crops (DeepFakeFace sample, openrail)",
        "description": "Private bundle for GPU training. Face crops come from the DeepFakeFace dataset (openrail). Do not make public."}, indent=2))
    (kn / "verifai_train_kaggle.ipynb").write_text(json.dumps(notebook(CELLS_ANALYSIS if analysis else CELLS_CELEBDF if celeb else CELLS_ROBUST if robust else None), indent=1))
    (kn / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{a.username}/{kernel_slug}", "title": kernel_title, "code_file": "verifai_train_kaggle.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": [f"{a.username}/{a.dataset_slug}"] + ([f"{a.username}/verifai-candidates"] if analysis else []) + (["reubensuju/celeb-df-v2"] if celeb else []), "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2))
    if analysis:
        import hashlib
        cd = DIST / "candidates"; cd.mkdir(parents=True)
        cands = {"dff_qa_R1.pth": BACKEND / "runs/candidates/dff_qa_R1.pth", "cdf_C0_noQA.pth": BACKEND / "runs/candidates/cdf_C0_noQA.pth", "cdf_C1_qa.pth": BACKEND / "runs/candidates/cdf_C1_qa.pth"}
        meta = {}
        for n, src in cands.items():
            shutil.copy(src, cd / n); meta[n] = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copy(BACKEND / "training/stream_ablation.py", cd / "stream_ablation.py")
        (cd / "candidates.json").write_text(json.dumps(meta, indent=2))
        (cd / "dataset-metadata.json").write_text(json.dumps({"title": "verifai-candidates", "id": f"{a.username}/verifai-candidates", "licenses": [{"name": "other"}],
                                                              "subtitle": "Candidate Verif-AI checkpoints for comparison", "description": "Private: model checkpoints only (no data)."}, indent=2))
        print("candidates dataset:", cd)
    size = sum(p.stat().st_size for p in ds.rglob("*") if p.is_file()) / 1e6
    print(f"dataset folder: {ds}  ({size:.0f} MB, {sum(1 for p in ds.rglob('*') if p.is_file())} files)\nkernel folder : {kn}")


if __name__ == "__main__":
    main()
