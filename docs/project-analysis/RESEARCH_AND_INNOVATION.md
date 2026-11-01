# RESEARCH AND INNOVATION ASSESSMENT

Honest positioning of the project against the field. Statements about the repository are verified (see `PROJECT_MASTER_CONTEXT.md`). Statements about the literature come from general knowledge, **not from a web search during this analysis** — verify every citation before publishing or presenting.

## 1. What problem does the project actually address?

**Narrow statement (what the code does):** *frame-level discrimination of real vs. face-swapped (Celeb-DF-style) face crops using a two-stream CNN that combines RGB appearance with a high-pass noise residual.*

Category separation requested in the brief:

| Category | Addressed by this project? | Why |
|---|---|---|
| AI-generated **image** detection (GAN/diffusion portraits, StyleGAN, Midjourney, SD) | **No** | No such data, no evaluation; the model has never seen a fully synthetic face |
| AI-generated **video** detection (text-to-video, Sora-class) | **No** | Same |
| **Deepfake** (face-swap) detection | **Yes, at frame level on one dataset** | Celeb-DF frames |
| Face **manipulation** detection (reenactment, attribute edit, inpainting) | **No** | Binary label only; no manipulation types |
| **Real vs synthetic human** classification | **Partially / indirectly** | Face-swapped video frames are "synthetic-ish" humans, but whole-face synthesis is untested |

Definitions for viva use: *Face-swap deepfake* — a target video in which the face is replaced by another identity (typically auto-encoder/GAN based). *Reenactment* — keeps identity, transfers expression/pose. *Fully synthetic face* — an image generated from noise/text with no source photo. *Face manipulation* — umbrella term for all of the above plus edits (age, hair, lips). *Real* — unmodified camera capture.

## 2. Why this matters
Face-swap and synthetic media enable fraud (KYC bypass, voice+video impersonation), non-consensual imagery, and disinformation; detection supports platform moderation and forensic triage. No detector gives guarantees; the field is adversarial and generators evolve faster than detectors, which is why **generalisation, not in-distribution accuracy, is the central research question**.

## 3. Existing approaches (context)

| Family | Representative work (verify details) | Idea | Typical weakness |
|---|---|---|---|
| Spatial CNN on faces | MesoNet (Afchar et al., 2018); XceptionNet on FaceForensics++ (Rössler et al., ICCV 2019) | Learn visual artifacts from cropped faces | Poor cross-dataset transfer, compression sensitivity |
| **Two-stream / noise residual** | Zhou et al., *Two-Stream Neural Networks for Tampered Face Detection* (CVPRW **2017** — README says 2019; verify); Bayar & Stamm constrained convolutions (2016); SRM from Fridrich & Kodovsky (2012) | Add a steganalysis-style residual stream | Residuals are fragile to re-compression/resizing |
| Frequency-domain | F3-Net (Qian et al., ECCV 2020); Durall et al. (2019/2020) | DCT/FFT cues of up-sampling | Depends on generator and pipeline |
| Blending / consistency | Face X-ray (Li et al., CVPR 2020); self-blended images | Detect the blend boundary, train on synthetic fakes | Needs careful synthetic-data design |
| Temporal | CNN+LSTM (Güera & Delp, 2018); LipForensics (Haliassos et al., CVPR 2021); 3D-CNN, video transformers | Exploit inter-frame/lip/blink inconsistencies | Compute, longer clips, data |
| Biological signals | Eye-blink, head pose, rPPG | Physiological inconsistencies | Increasingly mimicked |
| Universal/foundation-feature | UnivFD (Ojha et al., CVPR 2023) using CLIP features; diffusion detectors (Corvi et al., 2023) | Linear probes on large pretrained features | Still degrades with new generators |
| Datasets | FF++ (2019), Celeb-DF (Li et al., CVPR 2020), DFDC (Dolhansky et al., 2020), DeeperForensics-1.0, WildDeepfake | — | Dataset-specific artifacts |

Limits of conventional detectors: (1) shortcut learning on compression/resolution/background; (2) poor generalisation to unseen generators; (3) frame-wise models ignore temporal evidence; (4) adversarial vulnerability; (5) calibration — scores are not probabilities.

## 4. Where this project sits

| Dimension | This project | Typical modern baseline |
|---|---|---|
| Backbone | EfficientNet-B0 ×2 (ImageNet) | EfficientNet-B4 / Xception / ViT |
| Noise cue | 3 fixed 3×3 high-pass kernels on normalised RGB, channels summed | 30-filter SRM bank (5×5 variants), or learnable constrained conv |
| Fusion | Late concat, MLP | Often attention/cross-stream fusion |
| Temporal | none | frequently included |
| Evaluation | frame accuracy, one self-sampled 1k set | video-level AUC, official protocol, cross-dataset |
| Augmentation | none | compression/blur/colour augmentations are standard |

**Candid view of novelty.** The *architecture* — an RGB stream plus a noise-residual stream, fused late — is an established pattern (Zhou et al. 2017 and many derivatives). Using EfficientNet-B0 backbones and 3 fixed kernels is an *implementation choice*, not a research contribution. The project's value today is as a **clean, working, small-footprint two-stream baseline** with Grad-CAM, and as a **learning/engineering artefact**. A publishable contribution does not yet exist because (a) the SRM stream's benefit is unablated, (b) the evaluation protocol cannot support generalisation claims, (c) no comparison to published baselines was run on the same split. The README's statement that the result "demonstrates strong generalization" is **not supported** by the evidence.

## 5. Strengths that are real (and defensible)
1. Verified, converged training: training accuracy 0.8707 → 0.9921 in 10 epochs; held-out frame accuracy 98.70 %/99.20 % on its splits.
2. Compact: 11.97 M stored values (≈9.4 M effective after removing the unused ImageNet heads), 48 MB checkpoints.
3. Principled motivation: complementary content-suppressed residual view.
4. Built-in qualitative explainability hook (Grad-CAM) — though limited to the RGB stream.
5. Sensible preprocessing (face-focused crops with margin) and some reproducibility intent (seeds, per-epoch checkpoints).

## 6. Threats to validity (must be disclosed in any write-up)
1. **Frame-level split leakage** (same video in train/val).
2. **Test-sample/training overlap** possible (shared pool, no exclusion).
3. **Identity overlap** inherent to Celeb-DF's design.
4. **Single-dataset, single-generator** evidence.
5. **No variance estimates** — one seed, one run.
6. **Train/inference preprocessing mismatch** (JPEG vs raw).
7. **Compression-shortcut hypothesis** for the noise stream (untested).
8. **Hard-coded plot data** that already contains a wrong value (final loss).
9. **Metric mismatch** with the literature (frame accuracy vs video AUC).

## 7. Experiments that would turn this into research
| # | Experiment | Hypothesis / question | Output |
|---|---|---|---|
| E1 | Video- and identity-disjoint split + official test list | How much of 99.2 % survives? | video-level AUC/accuracy, CI |
| E2 | Ablation: RGB-only / SRM-only / fused | Does the noise stream help? | Δ accuracy, Δ AUC |
| E3 | 30-filter SRM bank, 5×5 kernels, learnable constrained conv, un-summed channels, raw-pixel input | Is the filter design suboptimal? | table |
| E4 | Backbone sweep (B0/B4/Xception/ViT) | Was the README's B4 claim ever worth it? | table |
| E5 | Cross-dataset train→test matrix (Celeb-DF, FF++, DFDC, DFD) | Generalisation gap | heat-map |
| E6 | Compression robustness curves | Is the SRM stream a codec detector? | acc vs JPEG-Q / CRF |
| E7 | Augmentation with compression/blur/noise | Closes the robustness gap? | table |
| E8 | Temporal head on frame embeddings (GRU / Transformer) | Gain over mean-pooling? | video AUC |
| E9 | Aggregation study (mean, max, top-k, trimmed) | Best video score | table |
| E10 | Calibration (temperature scaling), operating-point selection at FPR ≤1 % | Deployable thresholds | ECE, ROC |
| E11 | Explainability: CAM per stream, deletion/insertion tests | Does it attend to blend boundaries? | plots |
| E12 | Fully synthetic faces (StyleGAN/diffusion) as an extra test domain | Real-vs-synthetic-human scope | accuracy/AUC |
| E13 | Adversarial robustness (FGSM/PGD, re-encoding attacks) | How fragile? | attack success rate |
| E14 | Multi-seed runs (≥5) | Variance | mean ± std |

## 8. Possible publication directions (ranked by feasibility)
1. **Reproducibility/benchmark-integrity study:** "How much does frame-level splitting inflate Celeb-DF accuracy?" — cheap, uses this exact model, clearly novel *as an analysis*.
2. **Noise-stream ablation & robustness** under compression: does a fixed-SRM branch add value once compression augmentation is used?
3. **Lightweight two-stream + temporal head** for edge deployment (B0 backbones are compelling here) with latency/accuracy trade-offs.
4. **Cross-domain extension** to fully synthetic faces, if E12 shows partial transfer.
Target venues for such work (workshop-level initially): IEEE WIFS, ACM IH&MMSec, CVPR/ICCV Workshops on media forensics, student-track venues. A full conference paper would require E1–E6 at minimum.

## 9. Resume / presentation-safe claims
Safe (verified): "Implemented a two-stream (RGB + SRM-style noise residual) EfficientNet-B0 classifier in PyTorch for face-swap deepfake frame detection; trained on ~16 k Celeb-DF face crops; 98.7 % validation and 99.2 % held-out-sample frame accuracy (random frame split; not the official protocol)."
Unsafe (do **not** claim): "99.2 % on CelebDF-v2 benchmark", "detects AI-generated videos/images", "EfficientNet-B4", "30 SRM kernels", "real-time", "generalises", "outperforms published methods", "deployed", "full-stack".
