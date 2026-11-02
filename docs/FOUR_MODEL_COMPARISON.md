# Four-model comparison (for review - nothing here is integrated)

**Status: results for review.** No production file, checkpoint or application code was changed. No private media of any kind is used in any number or example in this document. Model 2 is **not recommended** for production (see section 6).
All numbers come from the verified metrics archive `docs/results/four_model_comparison/` (Kaggle notebook `iamankoo/verif-ai-analysis`, version 2, Tesla T4; archive SHA-256 `956ec7d55b2a803b...`; 22 JSON files; checked to contain no images, face crops, checkpoints or personal strings). The threshold is fixed at 0.5 everywhere; no model or threshold was chosen on any test number.

## 1. The four models
| # | Model | Initialisation | Training data | Selection (validation only) |
|---|---|---|---|---|
| 1 | **Production v2** (`verifai_v2_best.pth`, unchanged) | ImageNet | DeepFakeFace (DFF) | best validation ROC-AUC (epoch 9) |
| 2 | DFF-only + quality augmentation | ImageNet | DFF only, class-blind blur / down-up-scaling / JPEG / noise / low-light on 50% of crops | best mean of clean and degraded validation ROC-AUC (run R1, epoch 15 = the epoch cap) |
| 3 | DFF + Celeb-DF, no augmentation | ImageNet | DFF + Celeb-DF v2 frames (identity-disjoint) | same rule (epoch 4) |
| 4 | DFF + Celeb-DF + quality augmentation | ImageNet | same as 3, with the augmentation | same rule (epoch 12 = the epoch cap) |

Models 3 and 4 were trained on 2,315 DFF + about 16,300 Celeb-DF crops (41 training identities); validation used 9 identities and **testing used 9 other identities** (`cdf_test`). A fake video was kept only if all identities involved belonged to the same split; YouTube-real videos were never used in testing. The Celeb-DF data came from a third-party Kaggle copy (licence field "unknown"; official terms: research use only), used only inside a private notebook; no frames left Kaggle.

## 2. Sample counts (identical for every model)
| Evaluation set | Level | Real | Fake | Total |
|---|---|---:|---:|---:|
| DFF internal test | face crops | 292 | 283 | 575 |
| DFF cross-generator test (text2img = unseen generator) | face crops | 281 | 287 | 568 |
| Celeb-DF held-out identities | face crops (16 per video) | 1,454 | 896 | 2,350 |
| Celeb-DF held-out identities | **videos** | 91 | 56 | **147** |

Video-level score = median of the video's frame scores (for single-face videos this equals the production per-track rule; production takes the maximum over tracks, which matters only with several faces). Frames were sampled randomly across each video (16 per video), not at the production 2 fps. The 147 videos come from only 9 identities, so video-level confidence intervals are wide (section 4).

## 3. Clean results (threshold 0.5; confusion cells are TN / FP / FN / TP with FAKE = positive)
Precision, recall and F1 are for the FAKE class. 95% bootstrap CIs (grouped by identity, 300 resamples) are shown for frame-level ROC-AUC.

### 3.1 DFF internal test
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 575 (292 / 283) | 251 / 41 / 31 / 252 | 0.875 | 0.860 | 0.890 | 0.875 | 0.941 (0.922-0.960) | 0.942 | 0.140 | 0.110 |
| 2 DFF-only + quality aug | 575 (292 / 283) | 248 / 44 / 39 / 244 | 0.856 | 0.847 | 0.862 | 0.855 | 0.936 (0.912-0.955) | 0.937 | 0.151 | 0.138 |
| 3 DFF+Celeb-DF, no aug | 575 (292 / 283) | 255 / 37 / 48 / 235 | 0.852 | 0.864 | 0.830 | 0.847 | 0.931 (0.910-0.950) | 0.931 | 0.127 | 0.170 |
| 4 DFF+Celeb-DF + quality aug | 575 (292 / 283) | 254 / 38 / 57 / 226 | 0.835 | 0.856 | 0.799 | 0.826 | 0.921 (0.899-0.943) | 0.925 | 0.130 | 0.201 |

### 3.2 DFF cross-generator test (unseen generator)
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 568 (281 / 287) | 240 / 41 / 103 / 184 | 0.746 | 0.818 | 0.641 | 0.719 | 0.849 (0.820-0.880) | 0.841 | 0.146 | 0.359 |
| 2 DFF-only + quality aug | 568 (281 / 287) | 247 / 34 / 124 / 163 | 0.722 | 0.827 | 0.568 | 0.674 | 0.831 (0.800-0.860) | 0.830 | 0.121 | 0.432 |
| 3 DFF+Celeb-DF, no aug | 568 (281 / 287) | 249 / 32 / 148 / 139 | 0.683 | 0.813 | 0.484 | 0.607 | 0.819 (0.782-0.854) | 0.800 | 0.114 | 0.516 |
| 4 DFF+Celeb-DF + quality aug | 568 (281 / 287) | 251 / 30 / 182 / 105 | 0.627 | 0.778 | 0.366 | 0.498 | 0.795 (0.758-0.830) | 0.761 | 0.107 | 0.634 |

### 3.3 Celeb-DF held-out identities - frame level
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 2350 (1454 / 896) | 612 / 842 / 383 / 513 | 0.479 | 0.379 | 0.573 | 0.456 | 0.478 (0.370-0.606) | 0.353 | 0.579 | 0.427 |
| 2 DFF-only + quality aug | 2350 (1454 / 896) | 1074 / 380 / 667 / 229 | 0.554 | 0.376 | 0.256 | 0.304 | 0.517 (0.423-0.622) | 0.378 | 0.261 | 0.744 |
| 3 DFF+Celeb-DF, no aug | 2350 (1454 / 896) | 1368 / 86 / 29 / 867 | 0.951 | 0.910 | 0.968 | 0.938 | 0.991 (0.984-0.997) | 0.987 | 0.059 | 0.032 |
| 4 DFF+Celeb-DF + quality aug | 2350 (1454 / 896) | 1390 / 64 / 48 / 848 | 0.952 | 0.930 | 0.946 | 0.938 | 0.990 (0.980-0.996) | 0.986 | 0.044 | 0.054 |

### 3.4 Celeb-DF held-out identities - **video level** (reported separately from frame level)
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 147 (91 / 56) | 36 / 55 / 21 / 35 | 0.483 | 0.389 | 0.625 | 0.479 | 0.466 | 0.349 | 0.604 | 0.375 |
| 2 DFF-only + quality aug | 147 (91 / 56) | 75 / 16 / 46 / 10 | 0.578 | 0.385 | 0.179 | 0.244 | 0.516 | 0.389 | 0.176 | 0.821 |
| 3 DFF+Celeb-DF, no aug | 147 (91 / 56) | 87 / 4 / 0 / 56 | 0.973 | 0.933 | 1.000 | 0.966 | 0.999 | 0.998 | 0.044 | 0.000 |
| 4 DFF+Celeb-DF + quality aug | 147 (91 / 56) | 89 / 2 / 1 / 55 | 0.980 | 0.965 | 0.982 | 0.973 | 0.999 | 0.998 | 0.022 | 0.018 |

Video-level uncertainty (95% Wilson intervals, 147 videos):

| Model | video accuracy (95% Wilson CI) | FPR = real videos called FAKE | FNR = fake videos called REAL |
|---|---|---|---|
| 1 Production v2 | 0.483 (0.404-0.563) | 55/91 = 0.604 (0.502-0.699) | 21/56 = 0.375 (0.260-0.506) |
| 2 DFF-only + quality aug | 0.578 (0.497-0.655) | 16/91 = 0.176 (0.111-0.267) | 46/56 = 0.821 (0.702-0.900) |
| 3 DFF+Celeb-DF, no aug | 0.973 (0.932-0.989) | 4/91 = 0.044 (0.017-0.108) | 0/56 = 0.000 (0.000-0.064) |
| 4 DFF+Celeb-DF + quality aug | 0.980 (0.942-0.993) | 2/91 = 0.022 (0.006-0.077) | 1/56 = 0.018 (0.003-0.094) |

## 4. Degraded results (the same degradation is applied to every crop of every class, deterministically)
### 4.1 'mixed' degradation (each crop gets one of seven fixed degradations in rotation)
#### DFF internal test
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 575 (292 / 283) | 161 / 131 / 55 / 228 | 0.677 | 0.635 | 0.806 | 0.710 | 0.785 (0.754-0.818) | 0.799 | 0.449 | 0.194 |
| 2 DFF-only + quality aug | 575 (292 / 283) | 230 / 62 / 66 / 217 | 0.777 | 0.778 | 0.767 | 0.772 | 0.873 (0.845-0.901) | 0.885 | 0.212 | 0.233 |
| 3 DFF+Celeb-DF, no aug | 575 (292 / 283) | 161 / 131 / 69 / 214 | 0.652 | 0.620 | 0.756 | 0.682 | 0.748 (0.718-0.785) | 0.759 | 0.449 | 0.244 |
| 4 DFF+Celeb-DF + quality aug | 575 (292 / 283) | 242 / 50 / 73 / 210 | 0.786 | 0.808 | 0.742 | 0.773 | 0.862 (0.831-0.889) | 0.868 | 0.171 | 0.258 |

#### DFF cross-generator test
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 568 (281 / 287) | 162 / 119 / 83 / 204 | 0.644 | 0.632 | 0.711 | 0.669 | 0.709 (0.666-0.743) | 0.700 | 0.423 | 0.289 |
| 2 DFF-only + quality aug | 568 (281 / 287) | 234 / 47 / 136 / 151 | 0.678 | 0.763 | 0.526 | 0.623 | 0.752 (0.714-0.786) | 0.768 | 0.167 | 0.474 |
| 3 DFF+Celeb-DF, no aug | 568 (281 / 287) | 169 / 112 / 144 / 143 | 0.549 | 0.561 | 0.498 | 0.528 | 0.582 (0.533-0.625) | 0.571 | 0.399 | 0.502 |
| 4 DFF+Celeb-DF + quality aug | 568 (281 / 287) | 238 / 43 / 173 / 114 | 0.620 | 0.726 | 0.397 | 0.514 | 0.721 (0.675-0.760) | 0.703 | 0.153 | 0.603 |

#### Celeb-DF frames
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 2350 (1454 / 896) | 609 / 845 / 398 / 498 | 0.471 | 0.371 | 0.556 | 0.445 | 0.469 (0.399-0.550) | 0.347 | 0.581 | 0.444 |
| 2 DFF-only + quality aug | 2350 (1454 / 896) | 1199 / 255 / 747 / 149 | 0.574 | 0.369 | 0.166 | 0.229 | 0.524 (0.436-0.629) | 0.390 | 0.175 | 0.834 |
| 3 DFF+Celeb-DF, no aug | 2350 (1454 / 896) | 1346 / 108 / 233 / 663 | 0.855 | 0.860 | 0.740 | 0.795 | 0.940 (0.914-0.964) | 0.917 | 0.074 | 0.260 |
| 4 DFF+Celeb-DF + quality aug | 2350 (1454 / 896) | 1384 / 70 / 72 / 824 | 0.940 | 0.922 | 0.920 | 0.921 | 0.983 (0.966-0.993) | 0.977 | 0.048 | 0.080 |

#### Celeb-DF videos
| Model | n (real / fake) | TN / FP / FN / TP | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | 147 (91 / 56) | 28 / 63 / 22 / 34 | 0.422 | 0.351 | 0.607 | 0.444 | 0.436 | 0.337 | 0.692 | 0.393 |
| 2 DFF-only + quality aug | 147 (91 / 56) | 84 / 7 / 50 / 6 | 0.612 | 0.462 | 0.107 | 0.174 | 0.515 | 0.405 | 0.077 | 0.893 |
| 3 DFF+Celeb-DF, no aug | 147 (91 / 56) | 89 / 2 / 4 / 52 | 0.959 | 0.963 | 0.929 | 0.945 | 0.991 | 0.984 | 0.022 | 0.071 |
| 4 DFF+Celeb-DF + quality aug | 147 (91 / 56) | 90 / 1 / 2 / 54 | 0.980 | 0.982 | 0.964 | 0.973 | 0.997 | 0.996 | 0.011 | 0.036 |

| Model | video accuracy (95% Wilson CI) | FPR = real videos called FAKE | FNR = fake videos called REAL |
|---|---|---|---|
| 1 Production v2 | 0.422 (0.345-0.503) | 63/91 = 0.692 (0.591-0.778) | 22/56 = 0.393 (0.276-0.524) |
| 2 DFF-only + quality aug | 0.612 (0.532-0.687) | 7/91 = 0.077 (0.038-0.150) | 50/56 = 0.893 (0.785-0.950) |
| 3 DFF+Celeb-DF, no aug | 0.959 (0.914-0.981) | 2/91 = 0.022 (0.006-0.077) | 4/56 = 0.071 (0.028-0.170) |
| 4 DFF+Celeb-DF + quality aug | 0.980 (0.942-0.993) | 1/91 = 0.011 (0.002-0.060) | 2/56 = 0.036 (0.010-0.121) |

### 4.2 Other degraded conditions (compact)
#### Gaussian blur sigma 2.5
| Model | Split | n | TN/FP/FN/TP | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | DFF internal | 575 | 40/252/6/277 | 0.551 | 0.524 | 0.979 | 0.682 | 0.682 | 0.863 | 0.021 |
| 1 Production v2 | DFF cross-gen | 568 | 38/243/14/273 | 0.548 | 0.529 | 0.951 | 0.680 | 0.654 | 0.865 | 0.049 |
| 1 Production v2 | Celeb-DF frames | 2350 | 68/1386/77/819 | 0.377 | 0.371 | 0.914 | 0.528 | 0.372 | 0.953 | 0.086 |
| 1 Production v2 | Celeb-DF VIDEO | 147 | 1/90/2/54 | 0.374 | 0.375 | 0.964 | 0.540 | 0.336 | 0.989 | 0.036 |
| 2 DFF-only + quality aug | DFF internal | 575 | 218/74/63/220 | 0.762 | 0.748 | 0.777 | 0.763 | 0.858 | 0.253 | 0.223 |
| 2 DFF-only + quality aug | DFF cross-gen | 568 | 214/67/151/136 | 0.616 | 0.670 | 0.474 | 0.555 | 0.684 | 0.238 | 0.526 |
| 2 DFF-only + quality aug | Celeb-DF frames | 2350 | 1175/279/688/208 | 0.589 | 0.427 | 0.232 | 0.301 | 0.565 | 0.192 | 0.768 |
| 2 DFF-only + quality aug | Celeb-DF VIDEO | 147 | 78/13/48/8 | 0.585 | 0.381 | 0.143 | 0.208 | 0.573 | 0.143 | 0.857 |
| 3 DFF+Celeb-DF, no aug | DFF internal | 575 | 135/157/49/234 | 0.642 | 0.598 | 0.827 | 0.694 | 0.712 | 0.538 | 0.173 |
| 3 DFF+Celeb-DF, no aug | DFF cross-gen | 568 | 129/152/149/138 | 0.470 | 0.476 | 0.481 | 0.478 | 0.484 | 0.541 | 0.519 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF frames | 2350 | 1211/243/206/690 | 0.809 | 0.740 | 0.770 | 0.755 | 0.874 | 0.167 | 0.230 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF VIDEO | 147 | 79/12/11/45 | 0.844 | 0.789 | 0.804 | 0.796 | 0.914 | 0.132 | 0.196 |
| 4 DFF+Celeb-DF + quality aug | DFF internal | 575 | 212/80/66/217 | 0.746 | 0.731 | 0.767 | 0.748 | 0.822 | 0.274 | 0.233 |
| 4 DFF+Celeb-DF + quality aug | DFF cross-gen | 568 | 229/52/165/122 | 0.618 | 0.701 | 0.425 | 0.529 | 0.647 | 0.185 | 0.575 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF frames | 2350 | 1365/89/111/785 | 0.915 | 0.898 | 0.876 | 0.887 | 0.973 | 0.061 | 0.124 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF VIDEO | 147 | 89/2/4/52 | 0.959 | 0.963 | 0.929 | 0.945 | 0.992 | 0.022 | 0.071 |

#### Down/up-scaling x0.3
| Model | Split | n | TN/FP/FN/TP | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | DFF internal | 575 | 136/156/25/258 | 0.685 | 0.623 | 0.912 | 0.740 | 0.825 | 0.534 | 0.088 |
| 1 Production v2 | DFF cross-gen | 568 | 140/141/64/223 | 0.639 | 0.613 | 0.777 | 0.685 | 0.689 | 0.502 | 0.223 |
| 1 Production v2 | Celeb-DF frames | 2350 | 346/1108/185/711 | 0.450 | 0.391 | 0.794 | 0.524 | 0.487 | 0.762 | 0.206 |
| 1 Production v2 | Celeb-DF VIDEO | 147 | 21/70/10/46 | 0.456 | 0.397 | 0.821 | 0.535 | 0.458 | 0.769 | 0.179 |
| 2 DFF-only + quality aug | DFF internal | 575 | 236/56/69/214 | 0.783 | 0.793 | 0.756 | 0.774 | 0.872 | 0.192 | 0.244 |
| 2 DFF-only + quality aug | DFF cross-gen | 568 | 237/44/153/134 | 0.653 | 0.753 | 0.467 | 0.576 | 0.736 | 0.157 | 0.533 |
| 2 DFF-only + quality aug | Celeb-DF frames | 2350 | 1269/185/789/107 | 0.586 | 0.366 | 0.119 | 0.180 | 0.559 | 0.127 | 0.881 |
| 2 DFF-only + quality aug | Celeb-DF VIDEO | 147 | 84/7/50/6 | 0.612 | 0.462 | 0.107 | 0.174 | 0.563 | 0.077 | 0.893 |
| 3 DFF+Celeb-DF, no aug | DFF internal | 575 | 127/165/29/254 | 0.663 | 0.606 | 0.898 | 0.724 | 0.799 | 0.565 | 0.102 |
| 3 DFF+Celeb-DF, no aug | DFF cross-gen | 568 | 134/147/101/186 | 0.563 | 0.559 | 0.648 | 0.600 | 0.576 | 0.523 | 0.352 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF frames | 2350 | 1344/110/100/796 | 0.911 | 0.879 | 0.888 | 0.883 | 0.965 | 0.076 | 0.112 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF VIDEO | 147 | 88/3/4/52 | 0.952 | 0.945 | 0.929 | 0.937 | 0.987 | 0.033 | 0.071 |
| 4 DFF+Celeb-DF + quality aug | DFF internal | 575 | 244/48/82/201 | 0.774 | 0.807 | 0.710 | 0.756 | 0.859 | 0.164 | 0.290 |
| 4 DFF+Celeb-DF + quality aug | DFF cross-gen | 568 | 240/41/172/115 | 0.625 | 0.737 | 0.401 | 0.519 | 0.715 | 0.146 | 0.599 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF frames | 2350 | 1399/55/78/818 | 0.943 | 0.937 | 0.913 | 0.925 | 0.984 | 0.038 | 0.087 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF VIDEO | 147 | 90/1/3/53 | 0.973 | 0.981 | 0.946 | 0.964 | 0.997 | 0.011 | 0.054 |

#### 'phone_soft' (blur 1.2 + x0.6 down/up + JPEG q60)
| Model | Split | n | TN/FP/FN/TP | Acc | Prec | Rec | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Production v2 | DFF internal | 575 | 111/181/14/269 | 0.661 | 0.598 | 0.951 | 0.734 | 0.836 | 0.620 | 0.049 |
| 1 Production v2 | DFF cross-gen | 568 | 119/162/32/255 | 0.658 | 0.612 | 0.889 | 0.724 | 0.750 | 0.577 | 0.111 |
| 1 Production v2 | Celeb-DF frames | 2350 | 505/949/457/439 | 0.402 | 0.316 | 0.490 | 0.384 | 0.389 | 0.653 | 0.510 |
| 1 Production v2 | Celeb-DF VIDEO | 147 | 28/63/27/29 | 0.388 | 0.315 | 0.518 | 0.392 | 0.334 | 0.692 | 0.482 |
| 2 DFF-only + quality aug | DFF internal | 575 | 215/77/45/238 | 0.788 | 0.756 | 0.841 | 0.796 | 0.884 | 0.264 | 0.159 |
| 2 DFF-only + quality aug | DFF cross-gen | 568 | 224/57/118/169 | 0.692 | 0.748 | 0.589 | 0.659 | 0.770 | 0.203 | 0.411 |
| 2 DFF-only + quality aug | Celeb-DF frames | 2350 | 1194/260/764/132 | 0.564 | 0.337 | 0.147 | 0.205 | 0.474 | 0.179 | 0.853 |
| 2 DFF-only + quality aug | Celeb-DF VIDEO | 147 | 84/7/50/6 | 0.612 | 0.462 | 0.107 | 0.174 | 0.444 | 0.077 | 0.893 |
| 3 DFF+Celeb-DF, no aug | DFF internal | 575 | 90/202/23/260 | 0.609 | 0.563 | 0.919 | 0.698 | 0.784 | 0.692 | 0.081 |
| 3 DFF+Celeb-DF, no aug | DFF cross-gen | 568 | 107/174/108/179 | 0.504 | 0.507 | 0.624 | 0.559 | 0.513 | 0.619 | 0.376 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF frames | 2350 | 1367/87/67/829 | 0.934 | 0.905 | 0.925 | 0.915 | 0.980 | 0.060 | 0.075 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF VIDEO | 147 | 89/2/3/53 | 0.966 | 0.964 | 0.946 | 0.955 | 0.996 | 0.022 | 0.054 |
| 4 DFF+Celeb-DF + quality aug | DFF internal | 575 | 230/62/65/218 | 0.779 | 0.779 | 0.770 | 0.774 | 0.854 | 0.212 | 0.230 |
| 4 DFF+Celeb-DF + quality aug | DFF cross-gen | 568 | 224/57/151/136 | 0.634 | 0.705 | 0.474 | 0.567 | 0.698 | 0.203 | 0.526 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF frames | 2350 | 1387/67/73/823 | 0.940 | 0.925 | 0.919 | 0.922 | 0.986 | 0.046 | 0.081 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF VIDEO | 147 | 90/1/1/55 | 0.986 | 0.982 | 0.982 | 0.982 | 0.999 | 0.011 | 0.018 |

## 5. RGB-only / SRM-only / combined
The head is joint, so a stream cannot run alone. Each single-stream score neutralises the other stream by replacing its features with the mean over 300 class-balanced clean DFF training crops (the same reference for every model). These are **conditional** scores that show which stream pushes the decision; they are not independent classifiers. Higher score = more FAKE; FPR/FNR use threshold 0.5.

### 5.1 Clean
| Model | Split | combined AUC / FPR / FNR | RGB-only AUC / FPR / FNR | SRM-only AUC / FPR / FNR | mean score REAL: comb / RGB / SRM | mean score FAKE: comb / RGB / SRM |
|---|---|---|---|---|---|---|
| 1 Production v2 | DFF internal test | 0.941 / 0.14 / 0.11 | 0.895 / 0.17 / 0.19 | 0.855 / 0.51 / 0.07 | 0.17 / 0.21 / 0.49 | 0.88 / 0.80 / 0.89 |
| 1 Production v2 | DFF cross-generator test | 0.849 / 0.15 / 0.36 | 0.787 / 0.15 / 0.40 | 0.779 / 0.49 / 0.10 | 0.16 / 0.18 / 0.48 | 0.62 / 0.59 / 0.83 |
| 1 Production v2 | Celeb-DF held-out identities | 0.478 / 0.58 / 0.43 | 0.461 / 0.26 / 0.84 | 0.550 / 1.00 / 0.01 | 0.58 / 0.28 / 0.97 | 0.57 / 0.21 / 0.97 |
| 2 DFF-only + quality aug | DFF internal test | 0.936 / 0.15 / 0.14 | 0.913 / 0.14 / 0.19 | 0.843 / 0.56 / 0.05 | 0.15 / 0.16 / 0.55 | 0.84 / 0.79 / 0.89 |
| 2 DFF-only + quality aug | DFF cross-generator test | 0.831 / 0.12 / 0.43 | 0.790 / 0.11 / 0.48 | 0.751 / 0.61 / 0.08 | 0.14 / 0.15 / 0.57 | 0.57 / 0.52 / 0.85 |
| 2 DFF-only + quality aug | Celeb-DF held-out identities | 0.517 / 0.26 / 0.74 | 0.509 / 0.09 / 0.90 | 0.536 / 1.00 / 0.00 | 0.30 / 0.13 / 0.97 | 0.30 / 0.12 / 0.97 |
| 3 DFF+Celeb-DF, no aug | DFF internal test | 0.931 / 0.13 / 0.17 | 0.903 / 0.14 / 0.25 | 0.883 / 0.12 / 0.27 | 0.15 / 0.17 / 0.22 | 0.79 / 0.71 / 0.62 |
| 3 DFF+Celeb-DF, no aug | DFF cross-generator test | 0.819 / 0.11 / 0.52 | 0.747 / 0.11 / 0.66 | 0.808 / 0.11 / 0.47 | 0.14 / 0.16 / 0.23 | 0.47 / 0.38 / 0.49 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF held-out identities | 0.991 / 0.06 / 0.03 | 0.983 / 0.04 / 0.09 | 0.941 / 0.17 / 0.12 | 0.07 / 0.06 / 0.26 | 0.96 / 0.90 / 0.79 |
| 4 DFF+Celeb-DF + quality aug | DFF internal test | 0.921 / 0.13 / 0.20 | 0.885 / 0.15 / 0.26 | 0.848 / 0.03 / 0.68 | 0.13 / 0.15 / 0.07 | 0.78 / 0.72 / 0.37 |
| 4 DFF+Celeb-DF + quality aug | DFF cross-generator test | 0.795 / 0.11 / 0.63 | 0.780 / 0.11 / 0.60 | 0.665 / 0.02 / 0.95 | 0.11 / 0.12 / 0.06 | 0.38 / 0.41 / 0.11 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF held-out identities | 0.990 / 0.04 / 0.05 | 0.988 / 0.04 / 0.07 | 0.836 / 0.00 / 0.83 | 0.05 / 0.04 / 0.06 | 0.94 / 0.92 / 0.26 |

### 5.2 Degraded ('mixed')
| Model | Split | combined AUC / FPR / FNR | RGB-only AUC / FPR / FNR | SRM-only AUC / FPR / FNR | mean score REAL: comb / RGB / SRM | mean score FAKE: comb / RGB / SRM |
|---|---|---|---|---|---|---|
| 1 Production v2 | DFF internal test | 0.785 / 0.45 / 0.19 | 0.798 / 0.17 / 0.36 | 0.595 / 0.89 / 0.03 | 0.44 / 0.22 / 0.86 | 0.79 / 0.63 / 0.94 |
| 1 Production v2 | DFF cross-generator test | 0.709 / 0.42 / 0.29 | 0.715 / 0.22 / 0.53 | 0.550 / 0.91 / 0.01 | 0.41 / 0.23 / 0.86 | 0.68 / 0.49 / 0.94 |
| 1 Production v2 | Celeb-DF held-out identities | 0.469 / 0.58 / 0.44 | 0.461 / 0.16 / 0.90 | 0.528 / 1.00 / 0.00 | 0.55 / 0.21 / 0.97 | 0.52 / 0.16 / 0.97 |
| 2 DFF-only + quality aug | DFF internal test | 0.873 / 0.21 / 0.23 | 0.860 / 0.13 / 0.34 | 0.662 / 0.93 / 0.02 | 0.25 / 0.16 / 0.86 | 0.76 / 0.66 / 0.93 |
| 2 DFF-only + quality aug | DFF cross-generator test | 0.752 / 0.17 / 0.47 | 0.738 / 0.10 / 0.59 | 0.618 / 0.94 / 0.01 | 0.22 / 0.13 / 0.87 | 0.53 / 0.42 / 0.92 |
| 2 DFF-only + quality aug | Celeb-DF held-out identities | 0.524 / 0.18 / 0.83 | 0.501 / 0.06 / 0.94 | 0.550 / 1.00 / 0.00 | 0.24 / 0.09 / 0.95 | 0.25 / 0.08 / 0.95 |
| 3 DFF+Celeb-DF, no aug | DFF internal test | 0.748 / 0.45 / 0.24 | 0.752 / 0.40 / 0.26 | 0.595 / 0.29 / 0.59 | 0.44 / 0.42 / 0.38 | 0.74 / 0.73 / 0.45 |
| 3 DFF+Celeb-DF, no aug | DFF cross-generator test | 0.582 / 0.40 / 0.50 | 0.558 / 0.37 / 0.56 | 0.621 / 0.25 / 0.63 | 0.40 / 0.38 / 0.35 | 0.49 / 0.44 / 0.43 |
| 3 DFF+Celeb-DF, no aug | Celeb-DF held-out identities | 0.940 / 0.07 / 0.26 | 0.930 / 0.07 / 0.29 | 0.739 / 0.08 / 0.64 | 0.13 / 0.13 / 0.32 | 0.78 / 0.75 / 0.47 |
| 4 DFF+Celeb-DF + quality aug | DFF internal test | 0.862 / 0.17 / 0.26 | 0.849 / 0.21 / 0.24 | 0.665 / 0.00 / 0.95 | 0.19 / 0.22 / 0.08 | 0.72 / 0.73 / 0.14 |
| 4 DFF+Celeb-DF + quality aug | DFF cross-generator test | 0.721 / 0.15 / 0.60 | 0.700 / 0.16 / 0.60 | 0.615 / 0.01 / 0.99 | 0.17 / 0.19 / 0.08 | 0.40 / 0.40 / 0.10 |
| 4 DFF+Celeb-DF + quality aug | Celeb-DF held-out identities | 0.983 / 0.05 / 0.08 | 0.979 / 0.04 / 0.10 | 0.812 / 0.00 / 0.89 | 0.07 / 0.06 / 0.07 | 0.91 / 0.89 / 0.21 |

### 5.3 Example crops, Celeb-DF held-out (clean) - Model 1 (production)
| true label | clip (public Celeb-DF file id) | combined | RGB-only | SRM-only |
|---|---|---:|---:|---:|
| REAL | id23_0005 | 0.991 | 0.804 | 0.994 |
| REAL | id57_0006 | 0.022 | 0.029 | 0.870 |
| REAL | id56_0009 | 0.988 | 0.840 | 0.990 |
| REAL | id57_0004 | 0.936 | 0.107 | 0.996 |
| REAL | id26_0005 | 0.115 | 0.016 | 0.975 |
| FAKE | id57_id56_0004 | 0.164 | 0.021 | 0.983 |
| FAKE | id23_id26_0003 | 0.830 | 0.288 | 0.983 |
| FAKE | id23_id19_0009 | 0.347 | 0.007 | 0.995 |
| FAKE | id26_id23_0004 | 0.981 | 0.374 | 0.996 |
| FAKE | id19_id26_0009 | 0.932 | 0.044 | 0.997 |

### 5.4 Example crops, Celeb-DF held-out (clean) - Model 4
| true label | clip (public Celeb-DF file id) | combined | RGB-only | SRM-only |
|---|---|---:|---:|---:|
| REAL | id23_0005 | 0.000 | 0.000 | 0.038 |
| REAL | id57_0006 | 0.000 | 0.000 | 0.115 |
| REAL | id56_0009 | 0.000 | 0.001 | 0.006 |
| REAL | id57_0004 | 0.000 | 0.000 | 0.002 |
| REAL | id26_0005 | 0.000 | 0.000 | 0.038 |
| FAKE | id57_id56_0004 | 1.000 | 1.000 | 0.458 |
| FAKE | id23_id26_0003 | 1.000 | 1.000 | 0.096 |
| FAKE | id23_id19_0009 | 1.000 | 1.000 | 0.075 |
| FAKE | id26_id23_0004 | 1.000 | 1.000 | 0.055 |
| FAKE | id19_id26_0009 | 1.000 | 1.000 | 0.986 |

### 5.5 Example crops, DFF internal test (clean) - Model 1 (production)
| true label | generator | combined | RGB-only | SRM-only |
|---|---|---:|---:|---:|
| REAL | real | 0.001 | 0.038 | 0.039 |
| REAL | real | 0.009 | 0.011 | 0.806 |
| REAL | real | 0.993 | 0.987 | 0.958 |
| REAL | real | 0.201 | 0.932 | 0.089 |
| FAKE | inpainting | 0.969 | 0.867 | 0.978 |
| FAKE | inpainting | 0.909 | 0.979 | 0.505 |
| FAKE | inpainting | 0.999 | 0.997 | 0.988 |
| FAKE | insight | 0.951 | 0.985 | 0.601 |

### 5.6 Example crops, DFF internal test (clean) - Model 4
| true label | generator | combined | RGB-only | SRM-only |
|---|---|---:|---:|---:|
| REAL | real | 0.000 | 0.000 | 0.000 |
| REAL | real | 0.000 | 0.001 | 0.012 |
| REAL | real | 0.845 | 0.175 | 0.546 |
| REAL | real | 0.055 | 0.021 | 0.126 |
| FAKE | inpainting | 0.168 | 0.013 | 0.533 |
| FAKE | inpainting | 0.050 | 0.068 | 0.033 |
| FAKE | inpainting | 0.997 | 0.997 | 0.027 |
| FAKE | insight | 0.997 | 0.998 | 0.044 |

(The examples are the first rows of a seeded random sample, not hand-picked; the clip ids are public Celeb-DF file names; no image appears in this report.)

## 6. What the results show
**Model 1 (production) is at chance on held-out Celeb-DF video** (video accuracy 0.483, ROC-AUC 0.466, 55 of 91 real videos called FAKE). The stream breakdown shows why: on Celeb-DF frames its **SRM stream alone says FAKE for about 97% of both real and fake faces** (mean score 0.97 for each class, FPR 1.00), so it carries no information there, while its RGB stream alone says REAL for most frames (FNR 0.84). The combination is a coin flip that leans FAKE whenever the SRM stream saturates. This pattern - an RGB stream that says REAL while a saturated SRM stream pushes the combination toward FAKE - is the mechanism behind the investigated genuine-video false positive. The SRM stream of this model also over-flags soft or degraded faces on DFF (SRM-only FPR 0.51 clean, 0.89 under 'mixed').

**Model 2 (DFF-only + quality augmentation) - not recommended.** Its Celeb-DF video FPR is 0.176 (16/91) against 0.604 for model 1, but its FNR is **0.821 (46/56 fake videos called REAL)** and its video ROC-AUC is 0.516, i.e. it cannot separate real from fake on face-swap video. The lower false-positive rate is therefore **not** evidence of a fix: model 2 never saw face-swap videos, its SRM stream still outputs FAKE for essentially every Celeb-DF frame (SRM-only mean 0.97 for both classes), and its RGB stream says REAL for nearly everything (mean score 0.13 real vs 0.12 fake). The quality augmentation shifted its overall scores downward, so it calls fewer faces FAKE of either kind. This is a model that has been made quieter, not more discriminating. It also scores worse than model 1 on clean DFF cross-generator data (FNR 0.432 vs 0.359). It does help on degraded DFF crops (internal test FPR under blur sigma 2.5: 0.253 vs 0.863 for model 1; under the mixed degradation 0.212 vs 0.449) - a real but narrow benefit.

**Models 3 and 4 are the only models that separate real from fake on Celeb-DF.** Video accuracy 0.973 / 0.980, ROC-AUC 0.999, with the 95% intervals above. Model 4 (with augmentation) has video FPR 2/91 and FNR 1/56 clean, and stays strong under every degradation (video accuracy 0.959-0.986), while model 3 falls to 0.844 under blur sigma 2.5 (video FPR 0.132). In both, the SRM stream no longer saturates: in model 4 the RGB stream alone reaches ROC-AUC 0.988 on Celeb-DF frames (combined 0.990) and the SRM stream is conservative (SRM-only mean score 0.06 real / 0.26 fake), so the SRM-driven "everything is fake" behaviour is gone.

### Trade-offs between false positives and false negatives
| | FPR (real called FAKE) | FNR (fake called REAL) |
|---|---|---|
| Who pays | genuine media (the investigated failure: a genuine video accused of being fake) | manipulated media passes as genuine |
| Model 1 on Celeb-DF video | 0.604 | 0.375 |
| Model 4 on Celeb-DF video | 0.022 | 0.018 |
| Model 4 vs model 1 on DFF unseen generator (frames) | 0.107 vs 0.146 (better) | **0.634 vs 0.359 (much worse)** |
| Model 4 vs model 1 on DFF internal test | 0.130 vs 0.140 | **0.201 vs 0.110 (worse)** |

Adding Celeb-DF fixes the genuine-video false positives but **costs recall on DeepFakeFace fakes, especially the unseen generator** (accuracy 0.746 -> 0.627; 63% of text-to-image fakes now pass as real). Quality augmentation moves the operating point further toward REAL on clean data (model 4 vs 3: DFF cross-generator FNR 0.634 vs 0.516) in exchange for robustness under degradation (internal 'mixed' FPR 0.171 vs 0.449). A system that blocks fewer genuine users but lets more synthetic faces through is not "better"; it is a different operating point, and accuracy alone hides that. None of these models should be chosen on accuracy alone, and none is a strict improvement over production.

### What is and is not supported
* **Supported:** (a) the production model is at chance on held-out Celeb-DF video and its SRM stream is the main reason; (b) training with Celeb-DF frames removes that failure on held-out Celeb-DF identities; (c) class-blind quality augmentation makes the models markedly more robust to blur / down-up-scaling / compression on held-out DFF and Celeb-DF data; (d) model 2 does not solve the problem.
* **Not supported / unknown:** (a) **Celeb-DF video-level accuracy is within-dataset**: held-out identities, but the same face-swap method and the same kind of YouTube footage as the training videos, so 0.98 will overstate performance on other deepfake methods, other datasets (FaceForensics++, DFDC) or phone-recorded video. (b) Only 9 test identities / 147 videos (56 fake), so the intervals are wide (see 3.4). (c) Whether model 4 fixes **phone-camera** genuine video is **not established** by any number in this document (it contains no phone-recorded evaluation set). (d) Both Celeb-DF models might partly rely on Celeb-DF-specific compression or processing cues. (e) Models 2 and 4 stopped at the 15 / 12-epoch cap while still improving, so they may be under-trained. (f) The 'mixed' and other degradations are synthetic approximations of real capture pipelines. (g) The earlier local H.264 clip benchmark built from DFF photographs is **confounded** (real photographs are upscaled and fakes downscaled to a common canvas) and is not used as evidence here.

## 7. Recommendation for review (no action taken)
1. **Do not use model 2.**
2. **Do not replace the production checkpoint yet.** Model 4 is the only candidate that addresses the genuine-video failure, and it is the model the pre-declared validation rule selects among the Celeb-DF candidates, but it weakens detection of DeepFakeFace fakes (notably the unseen generator). Whether that trade is acceptable is a product decision for you.
3. If you choose model 4 (or 3), the next steps I would propose, each needing your approval: test on a second, independent labelled video dataset if one can be legitimately obtained; consider a mixed training set with more generators to recover unseen-generator recall; and add an explicit INCONCLUSIVE rule for low-detail faces.
4. Whatever is decided, the application should keep reporting scores as uncalibrated, and the README / evaluation report must state the Celeb-DF licence status and the within-dataset limitation.

## 8. Later candidates (E3A, E4): neither promoted
After this comparison two further candidates were trained and evaluated with the same protocol and the frozen promotion rule: **E3A** (modern-generator data, [E3_RESULTS.md](E3_RESULTS.md)) and **E4** (DeepFakeFace rows repeated x4, [E4_RESULTS.md](E4_RESULTS.md)). Both pass the Celeb-DF video criterion and the `modern_test` criterion but **fail criterion 2** (DeepFakeFace missed-fake rate: E4 0.208 internal / 0.509 cross-generator against limits 0.160 / 0.409). **Model 4 remains the default, v2 the rollback; E4 is an explicitly selectable candidate.** E4 held-out accuracy: DeepFakeFace internal 0.831, cross-generator 0.688, Celeb-DF video 0.980, `modern_test` 0.983 (the last is not generator-disjoint; Celeb-DF is within-dataset).
