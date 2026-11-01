# ML MODELS, DATASETS, TRAINING AND EVALUATION

Everything below is sourced from `backend/scripts/preprocessing.ipynb` (cells zero-indexed), `backend/scripts/plot.py`, the ten checkpoints, and the three PNG plots. Items not evidenced are marked **UNVERIFIED**. No numbers were invented.

---
## 1. Models in the repository

Exactly **one trained model architecture** exists (`TwoStreamNet`), saved at 10 epochs. Components: two ImageNet-pretrained EfficientNet-B0 backbones, one fixed SRM-style filter layer, one MLP head, and (for explanation only) a Grad-CAM wrapper. The MTCNN face detector (facenet-pytorch pretrained P/R/O-Net) is a second neural model used as a pre-processor; it is **not trained or fine-tuned** here.

### 1.1 Model card — `TwoStreamNet`

| Item | Value | Evidence |
|---|---|---|
| Task | Binary classification of a face crop: real (0) vs fake (1) | cell 3 labels |
| Input | 3×224×224 float tensor, ImageNet mean/std normalised | cell 3 transform |
| Output | scalar p ∈ (0,1) = P(fake), sigmoid | cell 2 |
| Decision rule | fake if p > 0.5 | cells 4–8 |
| Backbone ×2 | EfficientNet-B0 (`efficientnet-pytorch`, ImageNet init) | cell 2; 1280-d features |
| Noise front-end | `SRMLayer`, 3 fixed 3×3 zero-sum kernels | checkpoint tensor (3,3,3,3) |
| Head | 2560→512 (ReLU) → Dropout 0.5 → 512→1 → Sigmoid | checkpoint `fc.0`, `fc.3` |
| Stored values | 11,972,954 floats; 5,330,613 per backbone; 1,311,745 head | checkpoint inspection |
| Trainable | everything except the 81 SRM values | `requires_grad=False` |
| File size | ≈48.17 MB per checkpoint | `ls` |
| Rationale for choices | **UNVERIFIED** — no experiments or notes justify B0, 3 kernels, or hidden size 512 (README's "B4 trade-off" text refers to a model that is not the one implemented) | — |

### 1.2 Mathematics

**SRM front-end.** For input x ∈ ℝ^{3×H×W} and kernels K_k (k=1..3), output
y_k[i,j] = Σ_{c=1..3} Σ_{u,v∈{-1,0,1}} K_k[u,v] · x_c[i+u, j+v].
Because K_k is identical for each channel c, this equals K_k ∗ (x_R + x_G + x_B). Every K_k sums to 0, so a constant (flat) region maps to 0: the layer removes image content and keeps local pixel-to-pixel deviations (noise/edges). Steganalysis motivation (Fridrich & Kodovsky, 2012): generative pipelines leave statistics in these residuals that differ from camera-sensor noise.

**Backbone.** EfficientNet-B0 = MBConv blocks (inverted bottlenecks with depthwise conv + squeeze-and-excitation, Swish) scaled by compound scaling (Tan & Le, 2019). `extract_features` returns a 1280×7×7 map for 224² input; global average pooling gives f ∈ ℝ^1280.

**Fusion & output.** z = [f_rgb ; f_noise] ∈ ℝ^2560; h = Dropout(ReLU(W₁z+b₁)); p = σ(W₂h+b₂), σ(t)=1/(1+e^{-t}).

**Loss.** Binary cross-entropy, L = −(1/N) Σ [ y log p + (1−y) log(1−p) ], `nn.BCELoss` (applied to post-sigmoid probabilities; `BCEWithLogitsLoss` would be more stable — the logged mini-batch loss reaching exactly 0 at epoch 9 indicates probability saturation).

**Optimiser.** Adam, lr = 1e-4, default β=(0.9,0.999), ε=1e-8, no weight decay, no LR schedule.

**Grad-CAM.** For target layer activations A^k and score y: α_k = mean_{i,j} ∂y/∂A^k_{ij}; CAM = ReLU(Σ_k α_k A^k), upsampled to 224². Here A = output of `spatial_net._blocks[-1]` (7×7 for B0) and y = the fake-probability output. Consequences: resolution is coarse (7×7), it explains only the RGB stream, and it shows evidence *toward "fake"* regardless of the final label.

### 1.3 How the streams interact
Only through concatenation before the MLP. Gradients from the shared loss update both backbones. There is no weighting, gating, or auxiliary supervision, so **the relative contribution of the SRM stream is unknown** (no ablation; no per-stream accuracy anywhere in the repo).

### 1.4 Strengths and weaknesses
| Strengths | Weaknesses |
|---|---|
| Small, fast backbones (B0) | Frame-wise only: ignores temporal flicker, lip-sync, blink dynamics |
| Complementary views (appearance + high-frequency residual) — a well-motivated design | Residual computed on **normalised** pixels and with channels summed; only 3 filters; no truncation |
| ImageNet transfer learning converges in 1 epoch to 87 % | No augmentation ⇒ brittle to JPEG/blur/resize shifts (**UNVERIFIED**, never tested) |
| End-to-end training | Batch size 4 ⇒ noisy BatchNorm statistics |
| Grad-CAM for qualitative inspection | Train/inference preprocessing mismatch (JPEG vs raw crop) |
| | 21 % of stored weights are dead ImageNet classifier heads |

---
## 2. Datasets

| Property | Value | Verified? |
|---|---|---|
| Name | Celeb-DF (folders `Celeb-real`, `Celeb-fake`); README says "v2" | folder names **[V]**; version **UNVERIFIED** |
| Official size (README, matching the Li et al. 2020 paper from my knowledge) | 590 real + 5,639 fake videos | external; not re-counted (data absent) |
| Classes used | 2 (Celeb-real → 0, Celeb-fake → 1). `YouTube-real` (the dataset's third folder) is **not used** | cell 3 |
| Videos actually used for training | **UNVERIFIED** (the `processed/` folder and the code that made it are not in the repo) | — |
| Frames in training pool | 19,977 (15,981 train + 3,996 val) | **[V]** cell 3 output |
| Frames per video | target 10 (uniform interval from frame 0); effective count ≤10 because of bug B2 | code **[V]** |
| Class balance of the 19,977 frames | **UNVERIFIED** | — |
| Held-out sample | 1,000 frames = 500 real + 500 fake (apparently 50+50 videos × 10) | **[V]** cell 6 |
| Format | JPEG face crops 224×224 | code |
| Labelling | folder name → class index | cell 3 |
| Official Celeb-DF test list (`List_of_testing_videos.txt`) | Not referenced anywhere | — |

### 2.1 Preprocessing pipeline
1. Video → ~10 uniformly spaced frames (frame 0, interval, 2·interval …).
2. BGR→RGB; MTCNN → largest face, margin 20 px, 224×224; saved as JPEG via `cv2.imwrite` (OpenCV default quality **95**; lossy) — verified bit-exact in Phase 1 (facenet-pytorch's `save_img` uses `cv2.imwrite` for numpy frames, as cell 0 passes). *(An earlier version of this document did not state the quality.)*
3. Training-time: `Resize(224)` (no-op), `ToTensor`, ImageNet normalise. **No augmentation** (no flip, colour, blur, JPEG, crop).
4. SRM filtering happens *inside the model* on the normalised tensor.

### 2.2 Data-leakage risk assessment
| Risk | Why it applies here | Severity |
|---|---|---|
| **Same-video leakage in val split** | `random_split` over *frames*; ~10 near-duplicate frames of each video spread over train and val | **High** — val 98.70 % is likely optimistic |
| **Test sample overlaps training videos** | `new_test_data` is sampled (`seed 42`) from the same `raw_videos` folder; nothing excludes videos already in `processed/`. Celeb-real has only 590 videos, so a 50-video draw overlaps a broad training set with high probability if training used most real videos (**cannot be confirmed**) | **High** |
| **Identity leakage** | Celeb-DF has few identities (~59 per the paper — external knowledge); every fake is built from a real video of the same celebrity, so even a video-disjoint split shares identities | Medium–High |
| **Source/pairing leakage** | Fake and its source real video share background, pose, lighting | Medium |
| **Compression-shortcut risk** | Real and fake classes have different codec/bitrate histories in Celeb-DF; a noise-residual stream is precisely the kind of branch that can learn codec traces instead of manipulation | Medium (untested) |
| **JPEG double-compression** | All crops were re-saved as JPEG for both classes, so it is class-neutral during training — but it differs from inference input | Medium |
| **Checkpoint selection on test** | Epoch 10 chosen by default, not by test performance | Low |

Evidence nudging toward overfitting/leakage rather than genuine generalisation: held-out "new" accuracy (99.20 %) is *higher* than the internal validation accuracy (98.70 %) — what one expects if the "new" frames are no harder than validation, i.e., not truly unseen; confusion matrix is perfectly symmetric (4/4 errors). These are hints, not proofs.

### 2.3 Overfitting vs generator/artifact specificity
- The model has seen **one** forgery family (Celeb-DF synthesis) and **one** capture/compression regime. Behaviour on FaceForensics++, DFDC, DeeperForensics, Deepfake-Eval, GAN faces (StyleGAN), diffusion images, or newer face-swap tools is **unknown**.
- Published work consistently finds large drops in cross-dataset detection for frame-level CNNs; nothing in this repo indicates the model escapes this pattern. (General literature knowledge; no cross-dataset experiment here.)

---
## 3. Training configuration (verbatim from cells 1–4)

| Setting | Value |
|---|---|
| Seeds | `random.seed(42)`, `torch.manual_seed(42)`; numpy/CUDA seeds not set; DataLoader shuffle uses the global generator |
| Device | CUDA (GPU model not recorded) |
| Batch size | 4 (train and eval) |
| Epochs | 10 (3,996 steps each) |
| Optimiser / LR | Adam, 1e-4 constant |
| Loss | BCELoss |
| Regularisation | Dropout 0.5 in head only |
| Augmentation | none |
| Pretraining | ImageNet (both backbones) |
| Freezing | none (except SRM weights) |
| Model selection | none — every epoch saved, epoch 10 used |
| Mixed precision / grad clipping | none |

### 3.1 Per-epoch training log (from cell 4 output)
| Epoch | Mean train loss | Train accuracy* | Wall time |
|---:|---:|---:|---:|
| 1 | 0.2883 | 0.8707 | 13:14 |
| 2 | 0.0915 | 0.9653 | 10:39 |
| 3 | 0.0552 | 0.9790 | 15:11 |
| 4 | 0.0436 | 0.9830 | 10:51 |
| 5 | 0.0340 | 0.9867 | 10:25 |
| 6 | 0.0305 | 0.9884 | 11:54 |
| 7 | 0.0294 | 0.9887 | 10:48 |
| 8 | 0.0238 | 0.9900 | 10:21 |
| 9 | 0.0211 | 0.9915 | 10:22 |
| 10 | 0.0214 | 0.9921 | 21:28 |

\*Running accuracy over the epoch with dropout active and weights changing — **training** accuracy, not validation. Total ≈ 2 h 05 min.
Observation: the training curve is still improving at epoch 10 (not converged); no validation curve exists, so over-/under-fitting cannot be diagnosed from the logs.

---
## 4. Evaluation — every metric that actually exists

| Metric | Value | Dataset | Source | Notes |
|---|---|---|---|---|
| Accuracy | 98.70 % | val (3,996 frames) | cell 5 | frame-split, probable leakage |
| Accuracy | 99.20 % | test sample (1,000) | cell 6 | README headline |
| Precision (Real / Fake) | 0.99 / 0.99 | test sample | cell 6 | rounded to 2 dp |
| Recall (Real / Fake) | 0.99 / 0.99 | test sample | cell 6 | |
| F1 (Real / Fake) | 0.99 / 0.99 | test sample | cell 6 | |
| Confusion matrix | [[496, 4],[4, 496]] | test sample | cell 6 + `plots/confussion matrix.png` | rows = actual, cols = predicted, order Real, Fake |
| False-positive rate (real flagged fake) | 0.8 % (4/500) | derived | derived | not printed by notebook |
| False-negative rate | 0.8 % (4/500) | derived | derived | |
| PR-AUC (Fake positive) | 0.9995 | test sample | cell 7 + plot | trapezoid on PR curve; PR curve in plot shows precision ≈1 until recall ≈0.99 |
| ROC-AUC | — | — | **not computed** | |
| Video-level accuracy/AUC | — | — | **not computed** | no video-level inference |
| Calibration / ECE / Brier | — | — | **not computed** | |
| Robustness (JPEG, blur, resize, noise) | — | — | **not computed** | |
| Cross-dataset | — | — | **not computed** | |
| Inference time | ≈33 img/s (val), ≈45 img/s (test) on unspecified CUDA GPU, batch 4 | derived from tqdm | cells 5–6 | includes data loading |
| Memory / VRAM | — | — | **not measured** | |

Statistical uncertainty: 99.20 % on n=1000 has a Wilson 95 % interval ≈ [98.4 %, 99.6 %] if frames were independent; they are not (≤100 videos), so the real uncertainty is larger.

### 4.1 Where the README's headline number comes from
The README's 99.2 % equals the **test-sample accuracy** (cell 6: 99.20 %) and also nearly equals the final training accuracy (0.9921) — two different quantities with almost identical values, which invites confusion. `plot.py` labels the training accuracy "Model Accuracy".

### 4.2 Comparison caveat
Published Celeb-DF numbers are usually **video-level AUC on the official 518-video test list** (a different metric on a different split). The README's "73–90 % AUC" comparison is therefore not like-for-like and is unsourced.

---
## 5. Inference procedure details & train/test skew

| Aspect | Training/eval data | `run_forensic_analysis` |
|---|---|---|
| Crop source | MTCNN `post_process=True`, saved as **JPEG** | MTCNN `post_process=False`, raw tensor, **no JPEG** |
| Scale | `ToTensor` → [0,1] | `/255` → [0,1] |
| Resize | `Resize((224,224))` on PIL | `F.interpolate` bilinear (no-op at 224) |
| Normalise | ImageNet | ImageNet |
| Frame choice | ~10 uniform frames/video | frame 0 only |
| Weights loading | n/a | `torch.load` w/o `map_location` (fails on CPU) |

The JPEG discrepancy matters most for the SRM stream because quantisation noise is precisely high-frequency content. Effect size **UNVERIFIED**.

---
## 6. Recommended additional experiments (prioritised)
1. **Re-split by video and by identity**; report video-level accuracy and ROC-AUC. Use the official Celeb-DF test list.
2. **Ablation:** RGB-only vs SRM-only vs fused (same seed/data) → quantify the SRM stream's real contribution.
3. **Leakage audit:** hash/compare train vs test video IDs; re-evaluate on videos provably absent from training.
4. **Cross-dataset:** FaceForensics++ (c23/c40), DFDC, DFD; and cross-generator image sets.
5. **Robustness sweep:** JPEG quality 95→30, Gaussian blur, resize ↓/↑, noise, re-encoding with H.264 at several CRFs.
6. **Preprocessing parity:** retrain/evaluate with identical JPEG handling at inference.
7. **Video aggregation:** mean/median/top-k of frame logits; then a small temporal head (GRU/Transformer) on frame embeddings.
8. **Better training hygiene:** `BCEWithLogitsLoss`, augmentation, LR schedule, larger batch or GroupNorm, validation each epoch, best-checkpoint selection, multiple seeds with confidence intervals.
9. **Calibration & thresholding:** ECE, reliability diagram, operating point at fixed FPR.
10. **Explainability audit:** Grad-CAM for the SRM stream; check whether attention lies on blend boundaries or background.
11. **Efficiency:** drop unused `_fc` heads (−2.56 M values), export to ONNX, measure CPU latency.

## 7. Reproducibility checklist (current state)
| Item | State |
|---|---|
| Code for training frames | missing |
| Data | not included (Celeb-DF licence required) |
| Linear, re-runnable pipeline | no (out-of-order cells, hard-coded paths) |
| Pinned environment | no |
| Seeds | partial |
| Logged metrics | stdout text only; plot data re-typed by hand |
