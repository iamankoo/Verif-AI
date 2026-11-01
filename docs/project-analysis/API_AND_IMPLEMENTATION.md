# API AND IMPLEMENTATION REFERENCE

## 0. Headline: there is no API

A full search of the 25 repository files found **no web server, no routes, no HTTP client, no frontend, no database, no Docker/CI, and no tests**. The directory is called `backend/`, but it contains a notebook, one plotting script, one empty script, checkpoints and plots. Anything the README or banner implies about an "analyzer" you can send videos to is **not implemented [N]**.

| Item you asked about | Finding |
|---|---|
| API endpoints | none |
| Frontend ↔ backend communication | none |
| Upload / validation | none (only "can OpenCV decode a first frame?") |
| Database / storage | local files only (JPEG crops, `.pth`, PNG) |
| Auth, rate limiting, quotas | none |
| Deployment | none |
| Tests | none (the only "tests" are evaluation cells in the notebook) |

What follows documents the **actual callable surface** (Python functions in the notebook), execution instructions, and — clearly marked as *proposal, not implemented* — what a minimal service wrapper would require.

---
## 1. Module / function reference

Cells are zero-indexed in `backend/scripts/preprocessing.ipynb`.

| Symbol | Cell | Signature | Purpose | Status |
|---|---|---|---|---|
| module-level setup (MTCNN, paths, `LIMIT=50`) | 0 | — | create `MTCNN(image_size=224, margin=20, device, post_process=True)` | **[V]** |
| `extract_faces_balanced` | 0 | `(category, num_frames=10)` | sample ≤`LIMIT` videos (shuffle with `seed 42`), take ~10 frames, crop and save largest face | **[V]** (bug B2) |
| imports / seeds / device | 1 | — | `random.seed(42)`, `torch.manual_seed(42)`, device print "cuda" | **[V]** |
| `SRMLayer` | 2 | `nn.Module` | fixed 3-kernel high-pass conv | **[V]** |
| `TwoStreamNet` | 2 | `nn.Module` | two EfficientNet-B0 + head | **[V]** |
| `CelebDFDataset` | 3 | `Dataset(root_dir, transform)` | frame-path/label list; real=0 fake=1 | **[V]** |
| `transform` | 3 | `Compose` | Resize→ToTensor→Normalize | **[V]** |
| `random_split` / loaders | 3 | 80/20, bs 4 | produce 15,981 / 3,996 | **[V]** |
| `train_model` | 4 | `(epochs=10)` | training loop, checkpoint per epoch (writes to the notebook's working dir) | **[V]** |
| `evaluate_model` | 5 | `(model, loader)` | val accuracy | **[V]** |
| `evaluate_on_new_data` | 6 | `(model, loader)` | accuracy + report + confusion matrix | **[V]** |
| PR-curve block | 7 | — | `precision_recall_curve`, `auc` | **[V]** |
| `SRMLayer`, `TwoStreamNet` (copy) | 8 | — | duplicate of cell 2 (so inference cell is standalone) | duplicated code |
| `initialize_engine` | 8 | `(model_path=None) -> (model, mtcnn, device)` | build model + MTCNN(`post_process=False`), load weights | **[IU]** (B8) |
| `run_forensic_analysis` | 8 | `(video_path, model, mtcnn, device) -> None` | frame 0 → face → prob → Grad-CAM figure | **[IU]** |
| `deep_load_and_test` | 9 | `(weights_path, model, device)` | CPU-safe weight loading attempt, `strict=False`, prints weight sum | **never succeeded** (NameError) |
| *(empty)* | 10 | — | — | — |
| `plot.py` | — | script | hard-coded curves → `training_performance.png` | **[V]** (B14) |
| `pipeline.py` | — | **empty** | — | **[N]** |

### 1.1 Return values / output formats
- `run_forensic_analysis` **returns `None`**. Results exist only as console prints (`DEBUG - Raw Output`, `Final Score for this frame`) and a Matplotlib figure titled `Result: DEEPFAKE|REAL / Prob: x.xxxx`. There is no JSON, no confidence object, no saved heat-map.
- Errors are printed strings (`"❌ Error: Could not read video."`, `"❌ Error: No face detected."`) returned via `print(...)`, not raised exceptions.
- "Confidence" is simply the sigmoid output p(fake). For a REAL verdict the displayed "Prob" is still p(fake) (e.g. `image6.3.png` shows REAL with Prob 0.0000). It is an uncalibrated score.

### 1.2 Confidence calculation
`label = "DEEPFAKE" if prob > 0.50 else "REAL"`. The preceding "logit vs probability" conditional is unreachable (the model already applies `Sigmoid`).

---
## 2. Execution instructions (realistic)

These are inferred from the code; **they were not executed** during this analysis.

1. **Environment.** Python 3.10–3.13 (the notebook kernel metadata records 3.13.12). Install:
   `torch torchvision efficientnet-pytorch facenet-pytorch opencv-python tqdm numpy matplotlib scikit-learn seaborn grad-cam pillow`.
   - Use **`grad-cam`** (provides `pytorch_grad_cam`) — `requirements.txt` lists `pytorch-gradcam`, a different package; I confirmed `pytorch_grad_cam` is not importable in this analysis environment, which has `pytorch-gradcam` installed.
   - Add **`seaborn`** (used, undeclared). `torchaudio`, `albumentations`, `pandas` are unused.
2. **Data.** Obtain Celeb-DF and arrange `raw_videos/Celeb-real`, `raw_videos/Celeb-fake`. Edit `BASE_DIR`, `OUT_DIR`, `DATA_PATH`, `NEW_DATA_PATH` (several are `D:\CHODER\...`).
3. **Create frames** (cell 0) for both classes: set `CATEGORIES = ['Celeb-real','Celeb-fake']` and `LIMIT` as desired.
4. **Train** (cells 1–4). Expect ≈10–20 min/epoch on the author's CUDA GPU at batch 4.
5. **Evaluate** (cells 5–7).
6. **Infer** (cell 8): put the checkpoint path in `MY_WEIGHTS` and a video in `video_to_test`. On CPU, change to `torch.load(model_path, map_location=device)`. First run downloads ImageNet weights (`efficientnet-b0-355c32eb.pth`, ≈20.4 MB).
7. **Checkpoint locations:** `backend/models/two_stream_model_epoch_{1..9}.pth` and `backend/scripts/two_stream_model_epoch_10.pth` (the notebook writes checkpoints to its working directory, i.e. `scripts/`; epochs 1–9 were later moved).

### 2.1 Loading a checkpoint safely (verified approach)
```python
import torch
sd = torch.load("backend/scripts/two_stream_model_epoch_10.pth",
                map_location="cpu", weights_only=True)   # this exact call worked in the analysis
model.load_state_dict(sd)          # use strict=True to catch mismatches
```

---
## 3. Dependencies: where and why

| Package | Used in | Why |
|---|---|---|
| `torch`, `torchvision` | all model code, transforms | tensors, `nn`, `random_split`, `transforms` |
| `efficientnet-pytorch` | cell 2/8 | EfficientNet-B0 + ImageNet weights (downloaded at run time) |
| `facenet-pytorch` | cell 0/8 | MTCNN face detector/cropper |
| `opencv-python` (`cv2`) | cell 0/8 | video decoding, BGR→RGB |
| `Pillow` | cell 3 | image loading (installed via torchvision) |
| `numpy` | cells 2, 6–8 | SRM kernel construction, arrays |
| `scikit-learn` | cells 6–7 | `classification_report`, `confusion_matrix`, `precision_recall_curve`, `auc` |
| `matplotlib`, `seaborn` | cells 6–8, `plot.py` | plots |
| `pytorch_grad_cam` | cell 8 | Grad-CAM |
| `tqdm` | cells 0,4–6 | progress bars |
| `torchaudio`, `albumentations`, `pandas` | **nowhere** | unused |
| TensorFlow / Keras | **nowhere** | README only |

## 4. Security & reliability review of the existing code

| Area | Observation |
|---|---|
| Untrusted checkpoints | `torch.load` without `weights_only=True` (cells 8) executes pickle; only load files you trust |
| Untrusted media | `cv2.VideoCapture` on arbitrary files = codec/parsing attack surface; no size/duration/type limits |
| Path handling | `os.path.join` on user-controlled names would be needed with care if exposed (not currently) |
| Network at runtime | ImageNet weights downloaded from GitHub releases without hash pinning (library default) |
| Secrets | none found in any file |
| Failure handling | silent `except Exception: pass` around MTCNN (cell 0) hides real errors |
| Determinism | CUDA/numpy seeds not set; DataLoader unseeded |

## 5. Proposal only — what a serving layer would need (NOT implemented)

```mermaid
flowchart LR
  UI[Upload UI] -->|multipart video| API[FastAPI /v1/analyze]
  API --> VAL[type, size, duration validation]
  VAL --> Q[job queue]
  Q --> W[worker: sample N frames -> MTCNN -> model]
  W --> AGG[aggregate frame logits -> video score]
  AGG --> RES[(result store)]
  RES --> UI
```
Minimum requirements for it to be honest: aggregate over ≥16 frames, return per-frame scores and the number of frames with a detected face, CPU-safe loading (`map_location`, `weights_only=True`), the same JPEG-or-not preprocessing as training, explicit "unknown/low-quality" outcomes, and a published operating threshold tied to a measured FPR.

## 6. Feature status matrix

| Feature | Status |
|---|---|
| Frame extraction | **[V]** |
| Face detection/crop (MTCNN) | **[V]** |
| SRM noise stream | **[V]** |
| Two-stream fusion model | **[V]** |
| Training / checkpoints | **[V]** |
| Frame-level evaluation (acc, P/R/F1, CM, PR-AUC) | **[V]** |
| Grad-CAM | **[IU]** |
| Video inference | **[P]** (first frame only) |
| Temporal analysis | **[N]** |
| ROC-AUC / robustness / cross-dataset | **[N]** |
| REST API / frontend / DB / deployment / tests | **[N]** |
