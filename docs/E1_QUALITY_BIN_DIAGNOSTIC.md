# E1: quality-bin diagnostic (zero training)

**Status: results for review.** E1 is a measurement-only study: no training, no checkpoint modification, no threshold tuning (0.5 throughout), no change to the production application or checkpoint, and **no private media of any kind** (only public dataset crops). Data: `docs/results/e1_quality_bins/` (`dff_e1.json` computed locally with `backend/training/quality_bins.py`; `cdf_e1.json` computed inside the private Kaggle notebook `iamankoo/verif-ai-e1`, whose output holds only that JSON; see `SOURCE.txt`). The Celeb-DF copy is a third-party Kaggle mirror with licence field "unknown"; no Celeb-DF frame or crop left Kaggle.

**The question:** is image quality acting as a **label proxy** (does it predict REAL/FAKE by itself), how do the four models (1 production, 2 DFF-only + quality aug, 3 DFF+Celeb-DF, 4 DFF+Celeb-DF + quality aug) behave across quality, and what do the RGB and SRM streams do?

## 1. Definitions
* **Crop.** The stored 224x224 face crop, i.e. exactly what the model sees (MTCNN crop, margin 20, JPEG q95).
* **Sharpness** = variance of the Laplacian of the grey-scale crop (higher = more fine detail). **HF-noise** = mean absolute high-pass residual (grey minus Gaussian-blurred grey, sigma 1).
* **Quality bins (fixed in advance, factor-of-two spacing, not derived from any result):** `<40`, `40-80`, `80-160`, `160-320`, `>=320` (sharpness).
* **Label-proxy tests.** (a) Real/fake counts and fake share per bin, chi-square test of bin x label independence and Cramer's V. (b) ROC-AUC of *softness alone* as a classifier of FAKE (score = -sharpness): 0.5 = quality carries no label information; **< 0.5 = fakes are sharper than reals; > 0.5 = fakes are softer**. Same for low HF-noise. 95% CIs come from a bootstrap that resamples whole identities (DFF) or videos (Celeb-DF) so correlated crops do not shrink the interval.
* **Model behaviour per bin.** FPR (real called FAKE) and FNR (fake called REAL) with Wilson 95% CIs and the counts behind them, within-bin ROC-AUC (only where both classes have >= 20 crops), and the Spearman correlation between sharpness and score among reals and among fakes. RGB-only / SRM-only scores neutralise the other stream with the mean features of 300 class-balanced clean DFF training crops (conditional scores, not independent classifiers).

## 2. Sample counts
| Set | Real crops | Fake crops | Notes |
|---|---:|---:|---|
| DFF train | 1,170 | 1,145 | identity-disjoint from val/test |
| DFF val | 239 | 232 | |
| DFF internal test | 292 | 283 | unseen identities, seen generators |
| DFF cross-generator test | 281 | 287 | unseen identities **and** unseen generator (text2img) |
| Celeb-DF train (every 3rd training video) | 1,644 + 1,008 YouTube-real | 2,808 | a deterministic third of the training videos (enough for a distribution) |
| Celeb-DF val | 540 + 300 YouTube-real | 825 | |
| Celeb-DF held-out test | 1,454 | 895 | 9 held-out identities, 147 videos (91 real / 56 fake); one crop fewer than in the four-model comparison (a frame without a detectable face) |

The Celeb-DF crops are a **different sample** from those used in the four-model comparison (re-extracted with the same seed and rules inside Kaggle), so their numbers agree in spirit but not digit for digit.

## 3. Is quality a label proxy? (no model involved)
### 3.1 DeepFakeFace
| Split | n (real / fake) | median sharpness real / fake | median HF-noise real / fake | ROC-AUC of **softness alone** for FAKE (95% CI) | ROC-AUC of **low HF-noise alone** (95% CI) | chi-square p (bin x label) | Cramer's V |
|---|---|---|---|---|---|---|---|
| DFF train | 2315 (1170 / 1145) | 221.9 / 112.3 | 2.654 / 2.001 | 0.712 (0.696-0.729) | 0.683 (0.665-0.699) | 1.4e-65 | 0.365 |
| DFF val | 471 (239 / 232) | 211.3 / 108.9 | 2.644 / 2.053 | 0.692 (0.651-0.735) | 0.658 (0.616-0.703) | 5.9e-11 | 0.338 |
| DFF internal test | 575 (292 / 283) | 212.5 / 117.0 | 2.646 / 2.108 | 0.680 (0.645-0.713) | 0.647 (0.610-0.682) | 4.6e-11 | 0.307 |
| DFF cross-generator test | 568 (281 / 287) | 200.1 / 259.2 | 2.509 / 2.943 | 0.432 (0.385-0.475) | 0.399 (0.354-0.443) | 0.074 | 0.123 |

Fake share per fixed sharpness bin (real / fake counts):

| Split | <40: real / fake (fake %) | 40-80: real / fake (fake %) | 80-160: real / fake (fake %) | 160-320: real / fake (fake %) | >=320: real / fake (fake %) |
|---|---|---|---|---|---|
| DFF train | 29 / 159 (85%) | 103 / 254 (71%) | 282 / 352 (56%) | 369 / 246 (40%) | 387 / 134 (26%) |
| DFF val | 12 / 30 (71%) | 23 / 58 (72%) | 46 / 63 (58%) | 76 / 50 (40%) | 82 / 31 (27%) |
| DFF internal test | 10 / 30 (75%) | 28 / 69 (71%) | 77 / 88 (53%) | 91 / 59 (39%) | 86 / 37 (30%) |
| DFF cross-generator test | 7 / 2 (22%) | 20 / 15 (43%) | 77 / 60 (44%) | 89 / 101 (53%) | 88 / 109 (55%) |

### 3.2 Celeb-DF v2 (computed inside the private Kaggle notebook; crops never left it)
| Split | n (real / fake) | median sharpness real / fake | median HF-noise real / fake | ROC-AUC of **softness alone** for FAKE (95% CI) | ROC-AUC of **low HF-noise alone** (95% CI) | chi-square p (bin x label) | Cramer's V |
|---|---|---|---|---|---|---|---|
| Celeb-DF train (every 3rd video) | 5460 (2652 / 2808) | 18.4 / 14.8 | 0.878 / 0.801 | 0.618 (0.577-0.656) | 0.622 (0.582-0.661) | 1.2e-16 | 0.119 |
| Celeb-DF val | 1665 (840 / 825) | 19.9 / 16.1 | 0.916 / 0.83 | 0.632 (0.561-0.702) | 0.639 (0.570-0.709) | 1.5e-08 | 0.147 |
| Celeb-DF held-out test | 2349 (1454 / 895) | 18.0 / 14.6 | 0.863 / 0.771 | 0.613 (0.525-0.692) | 0.645 (0.563-0.723) | 1.3e-06 | 0.113 |

Fake share per fixed sharpness bin (real / fake counts):

| Split | <40: real / fake (fake %) | 40-80: real / fake (fake %) | 80-160: real / fake (fake %) | 160-320: real / fake (fake %) | >=320: real / fake (fake %) |
|---|---|---|---|---|---|
| Celeb-DF train (every 3rd video) | 2392 / 2699 (53%) | 226 / 96 (30%) | 30 / 13 (30%) | 4 / 0 (0%) | 0 / 0 (-%) |
| Celeb-DF val | 761 / 803 (51%) | 66 / 22 (25%) | 13 / 0 (0%) | 0 / 0 (-%) | 0 / 0 (-%) |
| Celeb-DF held-out test | 1334 / 870 (39%) | 98 / 17 (15%) | 18 / 7 (28%) | 4 / 1 (20%) | 0 / 0 (-%) |

## 4. Model behaviour across quality bins (threshold 0.5)
FPR = real crops called FAKE; FNR = fake crops called REAL; Wilson 95% CIs; bracketed counts are the numerators and denominators. Within-bin ROC-AUC is shown only where both classes have at least 20 crops.

### 4.1 DFF internal test (unseen identities, seen generators)
| Model | Bin | real / fake | FPR (95% CI) [FP/real] | FNR (95% CI) [FN/fake] | within-bin ROC-AUC |
|---|---|---|---|---|---|
| 1 Production v2 | <40 | 10 / 30 | 0.40 (0.17-0.69) [4/10] | 0.07 (0.02-0.21) [2/30] | - |
| 1 Production v2 | 40-80 | 28 / 69 | 0.32 (0.18-0.51) [9/28] | 0.06 (0.02-0.14) [4/69] | 0.905 |
| 1 Production v2 | 80-160 | 77 / 88 | 0.21 (0.13-0.31) [16/77] | 0.09 (0.05-0.17) [8/88] | 0.940 |
| 1 Production v2 | 160-320 | 91 / 59 | 0.09 (0.05-0.16) [8/91] | 0.15 (0.08-0.27) [9/59] | 0.942 |
| 1 Production v2 | >=320 | 86 / 37 | 0.05 (0.02-0.11) [4/86] | 0.22 (0.11-0.37) [8/37] | 0.935 |
| 2 DFF-only + quality aug | <40 | 10 / 30 | 0.40 (0.17-0.69) [4/10] | 0.10 (0.03-0.26) [3/30] | - |
| 2 DFF-only + quality aug | 40-80 | 28 / 69 | 0.29 (0.15-0.47) [8/28] | 0.10 (0.05-0.19) [7/69] | 0.900 |
| 2 DFF-only + quality aug | 80-160 | 77 / 88 | 0.17 (0.10-0.27) [13/77] | 0.11 (0.06-0.20) [10/88] | 0.940 |
| 2 DFF-only + quality aug | 160-320 | 91 / 59 | 0.13 (0.08-0.22) [12/91] | 0.19 (0.11-0.30) [11/59] | 0.926 |
| 2 DFF-only + quality aug | >=320 | 86 / 37 | 0.08 (0.04-0.16) [7/86] | 0.22 (0.11-0.37) [8/37] | 0.934 |
| 3 DFF+Celeb-DF, no aug | <40 | 10 / 30 | 0.30 (0.11-0.60) [3/10] | 0.07 (0.02-0.21) [2/30] | - |
| 3 DFF+Celeb-DF, no aug | 40-80 | 28 / 69 | 0.32 (0.18-0.51) [9/28] | 0.14 (0.08-0.25) [10/69] | 0.873 |
| 3 DFF+Celeb-DF, no aug | 80-160 | 77 / 88 | 0.19 (0.12-0.30) [15/77] | 0.11 (0.06-0.20) [10/88] | 0.922 |
| 3 DFF+Celeb-DF, no aug | 160-320 | 91 / 59 | 0.08 (0.04-0.15) [7/91] | 0.24 (0.15-0.36) [14/59] | 0.943 |
| 3 DFF+Celeb-DF, no aug | >=320 | 86 / 37 | 0.03 (0.01-0.10) [3/86] | 0.32 (0.20-0.49) [12/37] | 0.931 |
| 4 DFF+Celeb-DF + quality aug | <40 | 10 / 30 | 0.40 (0.17-0.69) [4/10] | 0.07 (0.02-0.21) [2/30] | - |
| 4 DFF+Celeb-DF + quality aug | 40-80 | 28 / 69 | 0.25 (0.13-0.43) [7/28] | 0.20 (0.12-0.31) [14/69] | 0.859 |
| 4 DFF+Celeb-DF + quality aug | 80-160 | 77 / 88 | 0.21 (0.13-0.31) [16/77] | 0.18 (0.12-0.28) [16/88] | 0.909 |
| 4 DFF+Celeb-DF + quality aug | 160-320 | 91 / 59 | 0.10 (0.05-0.18) [9/91] | 0.19 (0.11-0.30) [11/59] | 0.935 |
| 4 DFF+Celeb-DF + quality aug | >=320 | 86 / 37 | 0.02 (0.01-0.08) [2/86] | 0.38 (0.24-0.54) [14/37] | 0.940 |

### 4.2 DFF cross-generator test (unseen generator)
| Model | Bin | real / fake | FPR (95% CI) [FP/real] | FNR (95% CI) [FN/fake] | within-bin ROC-AUC |
|---|---|---|---|---|---|
| 1 Production v2 | <40 | 7 / 2 | 0.14 (0.03-0.51) [1/7] | 0.50 (0.09-0.91) [1/2] | - |
| 1 Production v2 | 40-80 | 20 / 15 | 0.30 (0.15-0.52) [6/20] | 0.20 (0.07-0.45) [3/15] | - |
| 1 Production v2 | 80-160 | 77 / 60 | 0.18 (0.11-0.28) [14/77] | 0.27 (0.17-0.39) [16/60] | 0.850 |
| 1 Production v2 | 160-320 | 89 / 101 | 0.11 (0.06-0.19) [10/89] | 0.36 (0.27-0.45) [36/101] | 0.883 |
| 1 Production v2 | >=320 | 88 / 109 | 0.11 (0.06-0.20) [10/88] | 0.43 (0.34-0.52) [47/109] | 0.860 |
| 2 DFF-only + quality aug | <40 | 7 / 2 | 0.43 (0.16-0.75) [3/7] | 1.00 (0.34-1.00) [2/2] | - |
| 2 DFF-only + quality aug | 40-80 | 20 / 15 | 0.25 (0.11-0.47) [5/20] | 0.33 (0.15-0.58) [5/15] | - |
| 2 DFF-only + quality aug | 80-160 | 77 / 60 | 0.13 (0.07-0.22) [10/77] | 0.32 (0.21-0.44) [19/60] | 0.836 |
| 2 DFF-only + quality aug | 160-320 | 89 / 101 | 0.11 (0.06-0.19) [10/89] | 0.42 (0.32-0.51) [42/101] | 0.853 |
| 2 DFF-only + quality aug | >=320 | 88 / 109 | 0.07 (0.03-0.14) [6/88] | 0.51 (0.42-0.61) [56/109] | 0.850 |
| 3 DFF+Celeb-DF, no aug | <40 | 7 / 2 | 0.29 (0.08-0.64) [2/7] | 0.50 (0.09-0.91) [1/2] | - |
| 3 DFF+Celeb-DF, no aug | 40-80 | 20 / 15 | 0.25 (0.11-0.47) [5/20] | 0.40 (0.20-0.64) [6/15] | - |
| 3 DFF+Celeb-DF, no aug | 80-160 | 77 / 60 | 0.16 (0.09-0.25) [12/77] | 0.45 (0.33-0.58) [27/60] | 0.823 |
| 3 DFF+Celeb-DF, no aug | 160-320 | 89 / 101 | 0.09 (0.05-0.17) [8/89] | 0.53 (0.44-0.63) [54/101] | 0.827 |
| 3 DFF+Celeb-DF, no aug | >=320 | 88 / 109 | 0.06 (0.02-0.13) [5/88] | 0.55 (0.46-0.64) [60/109] | 0.860 |
| 4 DFF+Celeb-DF + quality aug | <40 | 7 / 2 | 0.57 (0.25-0.84) [4/7] | 0.50 (0.09-0.91) [1/2] | - |
| 4 DFF+Celeb-DF + quality aug | 40-80 | 20 / 15 | 0.20 (0.08-0.42) [4/20] | 0.67 (0.42-0.85) [10/15] | - |
| 4 DFF+Celeb-DF + quality aug | 80-160 | 77 / 60 | 0.14 (0.08-0.24) [11/77] | 0.58 (0.46-0.70) [35/60] | 0.774 |
| 4 DFF+Celeb-DF + quality aug | 160-320 | 89 / 101 | 0.08 (0.04-0.15) [7/89] | 0.64 (0.55-0.73) [65/101] | 0.822 |
| 4 DFF+Celeb-DF + quality aug | >=320 | 88 / 109 | 0.05 (0.02-0.11) [4/88] | 0.65 (0.56-0.73) [71/109] | 0.861 |

### 4.3 Celeb-DF held-out identities (frame level)
| Model | Bin | real / fake | FPR (95% CI) [FP/real] | FNR (95% CI) [FN/fake] | within-bin ROC-AUC |
|---|---|---|---|---|---|
| 1 Production v2 | <40 | 1334 / 870 | 0.59 (0.57-0.62) [792/1334] | 0.43 (0.40-0.46) [375/870] | 0.482 |
| 1 Production v2 | 40-80 | 98 / 17 | 0.77 (0.67-0.84) [75/98] | 0.24 (0.10-0.47) [4/17] | - |
| 1 Production v2 | 80-160 | 18 / 7 | 0.50 (0.29-0.71) [9/18] | 0.71 (0.36-0.92) [5/7] | - |
| 1 Production v2 | 160-320 | 4 / 1 | 0.75 (0.30-0.95) [3/4] | 0.00 (0.00-0.79) [0/1] | - |
| 1 Production v2 | >=320 | 0 / 0 | - (-) [0/0] | - (-) [0/0] | - |
| 2 DFF-only + quality aug | <40 | 1334 / 870 | 0.24 (0.21-0.26) [314/1334] | 0.74 (0.71-0.77) [642/870] | 0.532 |
| 2 DFF-only + quality aug | 40-80 | 98 / 17 | 0.61 (0.51-0.70) [60/98] | 0.47 (0.26-0.69) [8/17] | - |
| 2 DFF-only + quality aug | 80-160 | 18 / 7 | 0.44 (0.25-0.66) [8/18] | 0.86 (0.49-0.97) [6/7] | - |
| 2 DFF-only + quality aug | 160-320 | 4 / 1 | 0.00 (0.00-0.49) [0/4] | 1.00 (0.21-1.00) [1/1] | - |
| 2 DFF-only + quality aug | >=320 | 0 / 0 | - (-) [0/0] | - (-) [0/0] | - |
| 3 DFF+Celeb-DF, no aug | <40 | 1334 / 870 | 0.04 (0.03-0.05) [50/1334] | 0.03 (0.02-0.05) [30/870] | 0.994 |
| 3 DFF+Celeb-DF, no aug | 40-80 | 98 / 17 | 0.22 (0.15-0.32) [22/98] | 0.00 (0.00-0.18) [0/17] | - |
| 3 DFF+Celeb-DF, no aug | 80-160 | 18 / 7 | 0.39 (0.20-0.61) [7/18] | 0.29 (0.08-0.64) [2/7] | - |
| 3 DFF+Celeb-DF, no aug | 160-320 | 4 / 1 | 0.00 (0.00-0.49) [0/4] | 0.00 (0.00-0.79) [0/1] | - |
| 3 DFF+Celeb-DF, no aug | >=320 | 0 / 0 | - (-) [0/0] | - (-) [0/0] | - |
| 4 DFF+Celeb-DF + quality aug | <40 | 1334 / 870 | 0.03 (0.02-0.04) [39/1334] | 0.05 (0.03-0.06) [41/870] | 0.994 |
| 4 DFF+Celeb-DF + quality aug | 40-80 | 98 / 17 | 0.18 (0.12-0.27) [18/98] | 0.00 (0.00-0.18) [0/17] | - |
| 4 DFF+Celeb-DF + quality aug | 80-160 | 18 / 7 | 0.28 (0.12-0.51) [5/18] | 0.57 (0.25-0.84) [4/7] | - |
| 4 DFF+Celeb-DF + quality aug | 160-320 | 4 / 1 | 0.00 (0.00-0.49) [0/4] | 0.00 (0.00-0.79) [0/1] | - |
| 4 DFF+Celeb-DF + quality aug | >=320 | 0 / 0 | - (-) [0/0] | - (-) [0/0] | - |

## 5. RGB-only vs SRM-only across quality
Spearman correlation between sharpness and score, among REAL and among FAKE crops separately (a value near 0 means the stream's score does not move with sharpness). Single-stream scores neutralise the other stream with the mean features of 300 class-balanced DFF training crops (see FOUR_MODEL_COMPARISON.md section 5).

### 5.1 DFF internal test
| Model | combined: real / fake | RGB-only: real / fake | SRM-only: real / fake |
|---|---|---|---|
| 1 Production v2 | -0.30 / -0.07 | -0.07 / 0.09 | -0.36 / -0.52 |
| 2 DFF-only + quality aug | -0.26 / -0.08 | -0.03 / 0.03 | -0.41 / -0.52 |
| 3 DFF+Celeb-DF, no aug | -0.29 / -0.29 | -0.22 / -0.23 | -0.25 / -0.23 |
| 4 DFF+Celeb-DF + quality aug | -0.31 / -0.16 | -0.12 / 0.02 | -0.53 / -0.47 |

### 5.2 DFF cross-generator test
| Model | combined: real / fake | RGB-only: real / fake | SRM-only: real / fake |
|---|---|---|---|
| 1 Production v2 | -0.27 / -0.17 | -0.09 / -0.10 | -0.34 / -0.24 |
| 2 DFF-only + quality aug | -0.29 / -0.12 | -0.09 / -0.05 | -0.41 / -0.33 |
| 3 DFF+Celeb-DF, no aug | -0.19 / -0.08 | -0.10 / -0.03 | -0.23 / -0.20 |
| 4 DFF+Celeb-DF + quality aug | -0.28 / -0.06 | -0.16 / 0.17 | -0.47 / -0.59 |

### 5.3 Celeb-DF held-out identities
| Model | combined: real / fake | RGB-only: real / fake | SRM-only: real / fake |
|---|---|---|---|
| 1 Production v2 | -0.03 / -0.09 | 0.09 / 0.12 | -0.32 / -0.41 |
| 2 DFF-only + quality aug | 0.06 / -0.03 | 0.13 / 0.07 | -0.08 / -0.22 |
| 3 DFF+Celeb-DF, no aug | 0.20 / -0.10 | 0.28 / -0.06 | -0.02 / -0.25 |
| 4 DFF+Celeb-DF + quality aug | 0.13 / -0.12 | 0.13 / -0.09 | 0.15 / -0.17 |

### 5.4 Per-bin single-stream FPR / FNR, DFF internal test
| Model | Bin | real / fake | combined FPR / FNR | RGB-only FPR / FNR | SRM-only FPR / FNR |
|---|---|---|---|---|---|
| 1 Production v2 | <40 | 10 / 30 | 0.40 / 0.07 | 0.20 / 0.30 | 1.00 / 0.00 |
| 1 Production v2 | 40-80 | 28 / 69 | 0.32 / 0.06 | 0.18 / 0.16 | 0.79 / 0.04 |
| 1 Production v2 | 80-160 | 77 / 88 | 0.21 / 0.09 | 0.22 / 0.14 | 0.47 / 0.05 |
| 1 Production v2 | 160-320 | 91 / 59 | 0.09 / 0.15 | 0.16 / 0.24 | 0.56 / 0.07 |
| 1 Production v2 | >=320 | 86 / 37 | 0.05 / 0.22 | 0.14 / 0.19 | 0.36 / 0.24 |
| 2 DFF-only + quality aug | <40 | 10 / 30 | 0.40 / 0.10 | 0.00 / 0.30 | 1.00 / 0.00 |
| 2 DFF-only + quality aug | 40-80 | 28 / 69 | 0.29 / 0.10 | 0.18 / 0.14 | 0.86 / 0.01 |
| 2 DFF-only + quality aug | 80-160 | 77 / 88 | 0.17 / 0.11 | 0.16 / 0.16 | 0.61 / 0.02 |
| 2 DFF-only + quality aug | 160-320 | 91 / 59 | 0.13 / 0.19 | 0.18 / 0.22 | 0.56 / 0.08 |
| 2 DFF-only + quality aug | >=320 | 86 / 37 | 0.08 / 0.22 | 0.09 / 0.22 | 0.36 / 0.19 |
| 3 DFF+Celeb-DF, no aug | <40 | 10 / 30 | 0.30 / 0.07 | 0.20 / 0.13 | 0.30 / 0.17 |
| 3 DFF+Celeb-DF, no aug | 40-80 | 28 / 69 | 0.32 / 0.14 | 0.29 / 0.23 | 0.18 / 0.28 |
| 3 DFF+Celeb-DF, no aug | 80-160 | 77 / 88 | 0.19 / 0.11 | 0.25 / 0.16 | 0.17 / 0.24 |
| 3 DFF+Celeb-DF, no aug | 160-320 | 91 / 59 | 0.08 / 0.24 | 0.09 / 0.32 | 0.10 / 0.27 |
| 3 DFF+Celeb-DF, no aug | >=320 | 86 / 37 | 0.03 / 0.32 | 0.03 / 0.49 | 0.05 / 0.41 |
| 4 DFF+Celeb-DF + quality aug | <40 | 10 / 30 | 0.40 / 0.07 | 0.40 / 0.27 | 0.10 / 0.47 |
| 4 DFF+Celeb-DF + quality aug | 40-80 | 28 / 69 | 0.25 / 0.20 | 0.21 / 0.32 | 0.04 / 0.58 |
| 4 DFF+Celeb-DF + quality aug | 80-160 | 77 / 88 | 0.21 / 0.18 | 0.17 / 0.24 | 0.05 / 0.60 |
| 4 DFF+Celeb-DF + quality aug | 160-320 | 91 / 59 | 0.10 / 0.19 | 0.13 / 0.20 | 0.03 / 0.85 |
| 4 DFF+Celeb-DF + quality aug | >=320 | 86 / 37 | 0.02 / 0.38 | 0.09 / 0.30 | 0.01 / 0.97 |

### 5.5 Per-bin single-stream FPR / FNR, Celeb-DF held-out
| Model | Bin | real / fake | combined FPR / FNR | RGB-only FPR / FNR | SRM-only FPR / FNR |
|---|---|---|---|---|---|
| 1 Production v2 | <40 | 1334 / 870 | 0.59 / 0.43 | 0.22 / 0.84 | 1.00 / 0.00 |
| 1 Production v2 | 40-80 | 98 / 17 | 0.77 / 0.24 | 0.59 / 0.47 | 0.99 / 0.00 |
| 1 Production v2 | 80-160 | 18 / 7 | 0.50 / 0.71 | 0.39 / 1.00 | 1.00 / 0.00 |
| 1 Production v2 | 160-320 | 4 / 1 | 0.75 / 0.00 | 0.00 / 1.00 | 1.00 / 0.00 |
| 1 Production v2 | >=320 | 0 / 0 | - / - | - / - | - / - |
| 2 DFF-only + quality aug | <40 | 1334 / 870 | 0.24 / 0.74 | 0.07 / 0.92 | 1.00 / 0.00 |
| 2 DFF-only + quality aug | 40-80 | 98 / 17 | 0.61 / 0.47 | 0.42 / 0.59 | 1.00 / 0.00 |
| 2 DFF-only + quality aug | 80-160 | 18 / 7 | 0.44 / 0.86 | 0.44 / 0.86 | 1.00 / 0.00 |
| 2 DFF-only + quality aug | 160-320 | 4 / 1 | 0.00 / 1.00 | 0.00 / 1.00 | 1.00 / 0.00 |
| 2 DFF-only + quality aug | >=320 | 0 / 0 | - / - | - / - | - / - |
| 3 DFF+Celeb-DF, no aug | <40 | 1334 / 870 | 0.04 / 0.03 | 0.02 / 0.09 | 0.15 / 0.11 |
| 3 DFF+Celeb-DF, no aug | 40-80 | 98 / 17 | 0.22 / 0.00 | 0.21 / 0.00 | 0.35 / 0.24 |
| 3 DFF+Celeb-DF, no aug | 80-160 | 18 / 7 | 0.39 / 0.29 | 0.28 / 0.57 | 0.33 / 0.14 |
| 3 DFF+Celeb-DF, no aug | 160-320 | 4 / 1 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 3 DFF+Celeb-DF, no aug | >=320 | 0 / 0 | - / - | - / - | - / - |
| 4 DFF+Celeb-DF + quality aug | <40 | 1334 / 870 | 0.03 / 0.05 | 0.02 / 0.07 | 0.00 / 0.83 |
| 4 DFF+Celeb-DF + quality aug | 40-80 | 98 / 17 | 0.18 / 0.00 | 0.13 / 0.00 | 0.06 / 1.00 |
| 4 DFF+Celeb-DF + quality aug | 80-160 | 18 / 7 | 0.28 / 0.57 | 0.28 / 0.57 | 0.00 / 1.00 |
| 4 DFF+Celeb-DF + quality aug | 160-320 | 4 / 1 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 DFF+Celeb-DF + quality aug | >=320 | 0 / 0 | - / - | - / - | - / - |

## 6. Findings
### 6.1 Is quality a label proxy? Yes in DeepFakeFace, weaker in Celeb-DF, and the direction is not stable across generators
* **DFF train:** softness alone predicts FAKE with ROC-AUC **0.712 (95% CI 0.696-0.729)**; chi-square p ~ 1e-65, Cramer's V 0.37. The fake share rises from 26% in the sharpest bin (>=320) to **85% in the softest** (<40: 29 real vs 159 fake). Median sharpness is 222 for reals and 112 for fakes. The same holds on DFF val (0.692, CI 0.651-0.735) and DFF internal test (0.680, CI 0.646-0.713), i.e. the proxy is a property of the dataset, not of one split. Low HF-noise behaves the same (train 0.683).
* **DFF cross-generator test: the relationship reverses.** Softness alone gives AUC **0.432 (CI 0.385-0.475)**: the unseen text-to-image fakes are *sharper* than their reals (median 259 vs 200). A model that has partly learned "soft = fake" will therefore miss exactly these fakes, which is where recall collapses (section 4.2).
* **Celeb-DF:** quality is a weaker proxy: softness-alone AUC 0.618 on train (CI 0.577-0.656), 0.632 on val and **0.613 on the held-out test (CI 0.525-0.692)**; Cramer's V 0.11-0.15; fakes are only slightly softer (median 14.6 vs 18.0 on test). The interval on test includes values close to chance, so this is a modest effect on 9 identities.
* **The two datasets occupy almost disjoint quality ranges.** About **94% of Celeb-DF crops fall in the softest bin** (<40) against **8% of DFF training crops**; DFF has only 29 + 103 real training crops below sharpness 80 (and 413 fake). The softest regime, in which essentially all of Celeb-DF lives, is **85% fake in the DFF training data**.

### 6.2 Model behaviour across quality
* **DFF internal test (all four models):** FPR falls and FNR rises as images get sharper. Production: FPR 0.40 (4/10) in the softest bin -> 0.05 (4/86) in the sharpest; FNR 0.07 (2/30) -> 0.22 (8/37). Model 4: FPR 0.40 -> 0.02 (2/86); FNR 0.07 -> **0.38 (14/37)**. Models 2 and 3 show the same shape. **Quality augmentation (2, 4) did not remove the dependence**; model 4 misses sharp fakes more than production does. The softest bins hold few crops (10 real / 30 fake), so their CIs are wide (see 4.1); the trend across all five bins is what matters (Spearman of score with sharpness among reals: production -0.30, model 2 -0.26, model 3 -0.29, model 4 -0.31).
* **Within a quality bin the models still discriminate** (combined within-bin ROC-AUC 0.86-0.94 in every DFF bin with enough data, for all four models), so they are not using quality alone; but the *base rate* of FAKE shifts with sharpness, which moves the operating point.
* **DFF cross-generator test:** FNR is highest in the sharpest bins for every model (e.g. production 0.43 (47/109) at >=320; model 4 0.65 (71/109)) - consistent with 6.1.
* **Celeb-DF (almost one quality regime):** production has **no discrimination within the softest bin (within-bin ROC-AUC 0.48)** and calls 59% of real crops FAKE there (0.567-0.620); model 2 is also near chance (0.53). Models 3 and 4 separate the classes inside that bin (within-bin ROC-AUC **0.994 / 0.994**, FPR 0.04 / 0.03, FNR 0.03 / 0.05). In the next bin up (40-80, only 98 real and 17 fake crops) models 3 and 4 have an **FPR of 0.22 (0.15-0.32) and 0.18 (0.12-0.27), about six times their FPR in the softest bin** (0.04 and 0.03): the Celeb-DF-trained models are tuned to the one soft regime they were trained on, and genuine faces slightly outside it are flagged more often. The CIs do not overlap, but the bin is small.

### 6.3 RGB vs SRM across quality
* The **SRM stream carries the quality dependence; the RGB stream mostly does not.** Spearman correlation of sharpness with the SRM-only score on DFF internal test: production -0.36 (real) / -0.52 (fake), model 4 -0.53 / -0.47; with the RGB-only score: production -0.07 / +0.09, model 4 -0.12 / +0.02.
* On Celeb-DF the production SRM-only score is also quality-driven (-0.32 / -0.41) and saturates (section 5 of FOUR_MODEL_COMPARISON.md); in model 4 the SRM-only correlations are small and of mixed sign (+0.15 / -0.17), and the RGB stream does most of the work.
* Practical reading: the SRM stream is where a quality-based shortcut lives in the production model; any fix that only augments inputs leaves the stream free to find another proxy.

## 7. Direct answers
| Question | Answer (with the evidence above) |
|---|---|
| Is quality a label proxy in DeepFakeFace? | **Yes.** Softness alone gives ROC-AUC 0.71 (CI 0.70-0.73) on train, 0.68 on internal test; fake share 26% -> 85% from sharp to soft. |
| Does it hold for the unseen generator? | **No - it reverses** (AUC 0.43, CI 0.39-0.48). The shortcut helps on seen generators and hurts on this one. |
| Is it a proxy in Celeb-DF? | **Weakly** (AUC 0.61, CI 0.52-0.69; V ~ 0.11), in a regime that is almost entirely very soft. |
| Do the models depend on quality? | **Yes, all four on DFF**: FPR falls and FNR rises with sharpness; quality augmentation did not remove it. They still discriminate within bins (within-bin AUC 0.86-0.94 on DFF; 0.994 within the Celeb-DF soft bin for models 3-4; 0.48 for production). |
| Which stream? | SRM carries the quality dependence; RGB does not (Spearman ~0). |
| Why does production fail on video? | Its training data tie "soft" to FAKE (85% fake in the softest bin) and all video frames are soft. |

## 8. Limits and consequences for the next experiments
* **Counts in the extreme bins are small** (DFF test: 7-10 real crops in the softest bin; Celeb-DF test: 98 real / 17 fake in 40-80 and almost nothing above). Per-bin FPR/FNR there have wide CIs (given in the tables); do not read single-bin numbers as precise.
* **One quality measure.** Sharpness and HF-noise are proxies for capture quality; they do not capture codec blocking, noise patterns or resolution history, so "quality is not a proxy" would not be established by these two measures alone.
* **Celeb-DF test = 9 identities / 147 videos / one face-swap method**, and ~94% of its crops sit in one quality bin; it cannot say whether a model is robust across the quality range. **No claim of general video reliability follows from it.**
* **DFF has a resolution confound** (real photographs of varied size, fakes all 512 px), which is part of why its quality proxy is strong.
* Everything here is measured on fixed crops; no new model was trained.

**Consequences (inputs to the research plan, `docs/RESEARCH_DIRECTIONS.md`):**
1. Real and fake data are needed **across the whole quality range, with the fake share balanced inside every bin** - today the soft regime is 85% fake in DFF and almost absent for sharp Celeb-DF-like footage. A pre-training check should compute this table for any candidate dataset.
2. Test sets must cover **both** regimes (sharp DFF-like and soft video-like) for **real and fake**, or low-quality false positives and high-quality false negatives stay hidden.
3. Quality-stratified sampling weights (equalising the fake share per bin) and a quality-invariance objective are better matched to these findings than more random degradation, and a **training-time gate** on stream behaviour (per-bin FPR/FNR and Spearman of each stream) can be derived directly from sections 4-5.
4. Acceptance criteria stay as agreed: unseen-generator FNR increase <= 0.05 absolute and genuine-video FPR <= 0.10, reported with CIs and sample counts, and **without claiming general video reliability from the small Celeb-DF evaluation**.
