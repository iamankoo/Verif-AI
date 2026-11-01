# Training on Kaggle (GPU) with the Kaggle CLI

CPU training on the development laptop would take several hours, so the model is trained on a free Kaggle GPU and the results are brought back and re-evaluated locally.

## One-time: authenticate the Kaggle CLI (do this yourself; never share the token)
1. kaggle.com → profile picture → **Settings** → **API** → **Create New Token** → a file `kaggle.json` is downloaded.
2. Move it to `C:\Users\<you>\.kaggle\kaggle.json` (Linux/macOS: `~/.kaggle/kaggle.json`, `chmod 600`).
3. The CLI is already installed in `backend/.venv` (`kaggle 1.7.4.5`). Check: `backend\.venv\Scripts\kaggle.exe datasets list --mine`.
4. Also verify your phone number on Kaggle (required for GPU and internet in notebooks).

## Build, upload, run (from the repository root)
```bash
# 1. build the upload package (embeds the exact local face crops, manifests, code, original checkpoint)
backend/.venv/Scripts/python kaggle/build_kaggle_bundle.py --username <your-kaggle-username>

# 2. create the PRIVATE dataset (~116 MB; contains face images from a public research dataset)
backend/.venv/Scripts/kaggle datasets create -p kaggle/dist/dataset --dir-mode zip

# 3. push + run the GPU notebook (waits for the dataset to finish processing first)
backend/.venv/Scripts/kaggle datasets status <your-kaggle-username>/verifai-bundle
backend/.venv/Scripts/kaggle kernels push -p kaggle/dist/kernel

# 4. follow it
backend/.venv/Scripts/kaggle kernels status <your-kaggle-username>/verif-ai-gpu-training

# 5. download the results when it says "complete"
backend/.venv/Scripts/kaggle kernels output <your-kaggle-username>/verif-ai-gpu-training -p kaggle/out
```
`kaggle/out/verifai_results.zip` contains, per run: `best.pth`, training history, config, summary and the full evaluation (metrics JSON + ROC/PR/confusion plots + predictions), plus `selection.json`, `gpu_benchmark.json` and the GPU unit-test log.

## What the notebook does
1. Verifies a GPU is present, records environment info.
2. Runs the unit tests that cover model loading, data validation and scoring **on the GPU** (includes the CUDA inference test).
3. Trains three configurations with the same seed/splits: **A** fine-tune the original checkpoint (primary), **B** ImageNet initialisation, **C** original checkpoint without augmentation. 15 epochs max, early stopping patience 4, AdamW 1e-4, batch 32, BCE-with-logits with class weighting, best epoch by **validation ROC-AUC**.
4. Evaluates every run on `val`, `test_internal` and `test_crossgen` (threshold fixed at 0.5).
5. Selects the run **by validation ROC-AUC only**; test numbers are printed for all runs for transparency and never used to choose.
6. Benchmarks GPU inference throughput and memory, then zips the results.

## After the download (locally)
Unzip into `backend/runs/`, copy the selected `best.pth` to `backend/models/verifai_v2_best.pth`, and re-run `python -m training.evaluate` locally on CPU to confirm the numbers (CPU/GPU results should agree to rounding). Everything reported in `docs/MODEL_EVALUATION_REPORT.md` must come from these re-evaluations; GPU training/benchmark numbers are reported as "measured on Kaggle (GPU model recorded in `selection.json`)".
