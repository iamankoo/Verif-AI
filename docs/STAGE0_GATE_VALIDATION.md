# Stage-0 gate validation: results

Protocol: `docs/STAGE0_PROTOCOL.md` v1, frozen before any run, sha256 `05b02ed713e992ba333ec199aea9dea78ad267d6c8481b78709fb64df36278cc` (re-checked when this report was generated). Inference only: no training, no fine-tuning, no checkpoint or production change, no threshold tuning, no test-split data, no private media. Raw outputs: `docs/results/stage0_gate/{dff,cdf}_gate.json`; this file is generated from them by `backend/training/make_stage0_report.py`.

## 1. What was run

| Set | Data | n real | n fake | Condition |
|---|---|---|---|---|
| V-DFF-clean | DeepfakeFace validation identities | 239 | 232 | clean |
| V-DFF-stress | DeepfakeFace validation identities | 239 | 232 | class-blind degradation: mixed |
| V-CDF-clean | Celeb-DF v2 validation identities | 840 | 825 | clean |
| V-CDF-stress | Celeb-DF v2 validation identities | 840 | 825 | class-blind degradation: mixed |

The four candidates are the production model (v2) and the three experimental models of `docs/FOUR_MODEL_COMPARISON.md`, all at threshold 0.5. Quality = variance of the Laplacian in the fixed bins <40, 40-80, 80-160, 160-320, >=320 (measured after the degradation). A bin is *testable* with >=30 samples per class; the thresholds are g1_auc 0.70, g1_fpr 0.50, g2_rho 0.25, g3_dev 0.15, g4_auc 0.80, min 2 testable bins.

## 2. Model-by-model results (ungated metrics, then gate checks)

Rates are with Wilson 95% intervals computed from the exact counts. `ns` = not evaluable (fewer than 2 testable bins); never counted as a pass.

| Model | Set | ROC-AUC | FPR (FP/n real) | FNR (FN/n fake) | G1 | G2 | G3 | G4 | testable bins | Gate verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 1_production_v2 | V-DFF-clean | 0.948 | 0.167 [0.125, 0.220] (40/239) | 0.078 [0.050, 0.119] (18/232) | fail | fail | fail | pass | 3 | **fail** |
| 1_production_v2 | V-DFF-stress | 0.758 | 0.469 [0.406, 0.532] (112/239) | 0.216 [0.167, 0.273] (50/232) | fail | fail | fail | fail | 2 | **fail** |
| 2_dff_only_qualityAug | V-DFF-clean | 0.939 | 0.126 [0.089, 0.174] (30/239) | 0.129 [0.092, 0.179] (30/232) | fail | fail | fail | pass | 3 | **fail** |
| 2_dff_only_qualityAug | V-DFF-stress | 0.888 | 0.188 [0.144, 0.243] (45/239) | 0.185 [0.141, 0.240] (43/232) | fail | pass | pass | pass | 2 | **fail** |
| 3_dff_celebdf_noQualityAug | V-DFF-clean | 0.928 | 0.117 [0.082, 0.164] (28/239) | 0.159 [0.118, 0.212] (37/232) | pass | pass | fail | pass | 3 | **fail** |
| 3_dff_celebdf_noQualityAug | V-DFF-stress | 0.748 | 0.397 [0.338, 0.461] (95/239) | 0.284 [0.230, 0.346] (66/232) | fail | fail | fail | fail | 2 | **fail** |
| 4_dff_celebdf_qualityAug | V-DFF-clean | 0.926 | 0.121 [0.086, 0.169] (29/239) | 0.228 [0.179, 0.287] (53/232) | pass | fail | pass | pass | 3 | **fail** |
| 4_dff_celebdf_qualityAug | V-DFF-stress | 0.856 | 0.176 [0.133, 0.229] (42/239) | 0.272 [0.218, 0.332] (63/232) | fail | pass | pass | pass | 2 | **fail** |
| 1_production_v2 | V-CDF-clean | 0.502 | 0.629 [0.595, 0.661] (528/840) | 0.364 [0.332, 0.397] (300/825) | fail | fail | ns | ns | 1 | **fail** |
| 1_production_v2 | V-CDF-stress | 0.495 | 0.583 [0.550, 0.616] (490/840) | 0.435 [0.402, 0.469] (359/825) | fail | fail | fail | fail | 2 | **fail** |
| 2_dff_only_qualityAug | V-CDF-clean | 0.494 | 0.305 [0.275, 0.337] (256/840) | 0.721 [0.690, 0.751] (595/825) | fail | fail | ns | ns | 1 | **fail** |
| 2_dff_only_qualityAug | V-CDF-stress | 0.516 | 0.138 [0.116, 0.163] (116/840) | 0.873 [0.848, 0.894] (720/825) | fail | fail | pass | fail | 2 | **fail** |
| 3_dff_celebdf_noQualityAug | V-CDF-clean | 0.992 | 0.050 [0.037, 0.067] (42/840) | 0.035 [0.025, 0.050] (29/825) | pass | pass | ns | ns | 1 | **no_failure** |
| 3_dff_celebdf_noQualityAug | V-CDF-stress | 0.949 | 0.056 [0.042, 0.074] (47/840) | 0.255 [0.226, 0.285] (210/825) | pass | fail | fail | fail | 2 | **fail** |
| 4_dff_celebdf_qualityAug | V-CDF-clean | 0.989 | 0.035 [0.024, 0.049] (29/840) | 0.057 [0.043, 0.075] (47/825) | pass | pass | ns | ns | 1 | **no_failure** |
| 4_dff_celebdf_qualityAug | V-CDF-stress | 0.976 | 0.057 [0.043, 0.075] (48/840) | 0.099 [0.081, 0.122] (82/825) | pass | fail | pass | pass | 2 | **fail** |

Per-model verdict (worst over its sets): DFF: 1_production_v2 = FAIL; DFF: 2_dff_only_qualityAug = FAIL; DFF: 3_dff_celebdf_noQualityAug = FAIL; DFF: 4_dff_celebdf_qualityAug = FAIL; CDF: 1_production_v2 = FAIL; CDF: 2_dff_only_qualityAug = FAIL; CDF: 3_dff_celebdf_noQualityAug = FAIL; CDF: 4_dff_celebdf_qualityAug = FAIL.

### Stream sanity (G1) and quality dependence (G2)

| Model | Set | RGB-only AUC | SRM-only AUC | SRM-only FPR | Spearman(sharpness, score) real / fake |
|---|---|---|---|---|---|
| 1_production_v2 | V-DFF-clean | 0.902 | 0.845 | 0.502 | -0.264 / -0.119 |
| 1_production_v2 | V-DFF-stress | 0.757 | 0.636 | 0.870 | -0.530 / -0.202 |
| 2_dff_only_qualityAug | V-DFF-clean | 0.911 | 0.826 | 0.594 | -0.226 / -0.075 |
| 2_dff_only_qualityAug | V-DFF-stress | 0.870 | 0.708 | 0.900 | -0.208 / +0.083 |
| 3_dff_celebdf_noQualityAug | V-DFF-clean | 0.898 | 0.879 | 0.142 | -0.187 / -0.317 |
| 3_dff_celebdf_noQualityAug | V-DFF-stress | 0.752 | 0.618 | 0.272 | -0.304 / -0.315 |
| 4_dff_celebdf_qualityAug | V-DFF-clean | 0.890 | 0.847 | 0.029 | -0.239 / -0.205 |
| 4_dff_celebdf_qualityAug | V-DFF-stress | 0.850 | 0.656 | 0.008 | -0.190 / +0.072 |
| 1_production_v2 | V-CDF-clean | 0.489 | 0.530 | 0.987 | -0.072 / -0.040 |
| 1_production_v2 | V-CDF-stress | 0.499 | 0.508 | 1.000 | -0.495 / -0.486 |
| 2_dff_only_qualityAug | V-CDF-clean | 0.494 | 0.519 | 1.000 | +0.066 / -0.070 |
| 2_dff_only_qualityAug | V-CDF-stress | 0.507 | 0.537 | 1.000 | -0.078 / -0.052 |
| 3_dff_celebdf_noQualityAug | V-CDF-clean | 0.987 | 0.944 | 0.158 | +0.000 / +0.061 |
| 3_dff_celebdf_noQualityAug | V-CDF-stress | 0.939 | 0.784 | 0.098 | +0.127 / -0.216 |
| 4_dff_celebdf_qualityAug | V-CDF-clean | 0.987 | 0.855 | 0.010 | +0.004 / +0.158 |
| 4_dff_celebdf_qualityAug | V-CDF-stress | 0.972 | 0.813 | 0.004 | +0.005 / -0.067 |

## 3. Per-quality-bin results with sample counts

Only bins with >=30 samples per class are *testable* and enter G3/G4; the others are shown for transparency.


**1_production_v2 - V-DFF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 12 | 30 | no | 0.167 [0.047, 0.448] | 0.033 [0.006, 0.167] | n/a |
| 40-80 | 23 | 58 | no | 0.391 [0.222, 0.592] | 0.017 [0.003, 0.091] | 0.929 |
| 80-160 | 46 | 63 | yes | 0.217 [0.123, 0.356] | 0.064 [0.025, 0.152] | 0.935 |
| 160-320 | 76 | 50 | yes | 0.158 [0.093, 0.256] | 0.080 [0.032, 0.188] | 0.947 |
| >=320 | 82 | 31 | yes | 0.085 [0.042, 0.166] | 0.258 [0.137, 0.432] | 0.930 |

**1_production_v2 - V-DFF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 157 | 166 | yes | 0.605 [0.527, 0.678] | 0.084 [0.051, 0.137] | 0.790 |
| 40-80 | 17 | 12 | no | 0.412 [0.216, 0.640] | 0.000 [0.000, 0.242] | n/a |
| 80-160 | 9 | 9 | no | 0.444 [0.189, 0.733] | 0.556 [0.267, 0.811] | n/a |
| 160-320 | 14 | 8 | no | 0.286 [0.117, 0.546] | 0.500 [0.215, 0.785] | n/a |
| >=320 | 42 | 37 | yes | 0.048 [0.013, 0.158] | 0.730 [0.570, 0.846] | 0.774 |

**2_dff_only_qualityAug - V-DFF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 12 | 30 | no | 0.250 [0.089, 0.532] | 0.067 [0.018, 0.213] | n/a |
| 40-80 | 23 | 58 | no | 0.217 [0.097, 0.419] | 0.103 [0.048, 0.208] | 0.917 |
| 80-160 | 46 | 63 | yes | 0.152 [0.076, 0.282] | 0.079 [0.034, 0.173] | 0.938 |
| 160-320 | 76 | 50 | yes | 0.092 [0.045, 0.178] | 0.100 [0.043, 0.214] | 0.960 |
| >=320 | 82 | 31 | yes | 0.098 [0.050, 0.181] | 0.387 [0.237, 0.562] | 0.891 |

**2_dff_only_qualityAug - V-DFF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 157 | 166 | yes | 0.185 [0.132, 0.253] | 0.193 [0.140, 0.260] | 0.886 |
| 40-80 | 17 | 12 | no | 0.235 [0.096, 0.473] | 0.167 [0.047, 0.448] | n/a |
| 80-160 | 9 | 9 | no | 0.333 [0.121, 0.646] | 0.111 [0.020, 0.435] | n/a |
| 160-320 | 14 | 8 | no | 0.143 [0.040, 0.399] | 0.250 [0.071, 0.591] | n/a |
| >=320 | 42 | 37 | yes | 0.167 [0.083, 0.306] | 0.162 [0.076, 0.311] | 0.913 |

**3_dff_celebdf_noQualityAug - V-DFF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 12 | 30 | no | 0.250 [0.089, 0.532] | 0.100 [0.035, 0.256] | n/a |
| 40-80 | 23 | 58 | no | 0.217 [0.097, 0.419] | 0.103 [0.048, 0.208] | 0.923 |
| 80-160 | 46 | 63 | yes | 0.130 [0.061, 0.257] | 0.143 [0.077, 0.250] | 0.911 |
| 160-320 | 76 | 50 | yes | 0.105 [0.054, 0.194] | 0.180 [0.098, 0.308] | 0.933 |
| >=320 | 82 | 31 | yes | 0.073 [0.034, 0.151] | 0.323 [0.186, 0.499] | 0.898 |

**3_dff_celebdf_noQualityAug - V-DFF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 157 | 166 | yes | 0.522 [0.445, 0.599] | 0.145 [0.099, 0.206] | 0.784 |
| 40-80 | 17 | 12 | no | 0.471 [0.262, 0.690] | 0.083 [0.015, 0.354] | n/a |
| 80-160 | 9 | 9 | no | 0.333 [0.121, 0.646] | 0.444 [0.189, 0.733] | n/a |
| 160-320 | 14 | 8 | no | 0.143 [0.040, 0.399] | 0.500 [0.215, 0.785] | n/a |
| >=320 | 42 | 37 | yes | 0.000 [0.000, 0.084] | 0.892 [0.753, 0.957] | 0.763 |

**4_dff_celebdf_qualityAug - V-DFF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 12 | 30 | no | 0.250 [0.089, 0.532] | 0.167 [0.073, 0.336] | n/a |
| 40-80 | 23 | 58 | no | 0.217 [0.097, 0.419] | 0.224 [0.136, 0.347] | 0.894 |
| 80-160 | 46 | 63 | yes | 0.174 [0.091, 0.307] | 0.222 [0.137, 0.339] | 0.892 |
| 160-320 | 76 | 50 | yes | 0.092 [0.045, 0.178] | 0.200 [0.112, 0.330] | 0.944 |
| >=320 | 82 | 31 | yes | 0.073 [0.034, 0.151] | 0.355 [0.211, 0.530] | 0.925 |

**4_dff_celebdf_qualityAug - V-DFF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 157 | 166 | yes | 0.172 [0.121, 0.239] | 0.277 [0.215, 0.350] | 0.851 |
| 40-80 | 17 | 12 | no | 0.294 [0.133, 0.531] | 0.167 [0.047, 0.448] | n/a |
| 80-160 | 9 | 9 | no | 0.222 [0.063, 0.547] | 0.333 [0.121, 0.646] | n/a |
| 160-320 | 14 | 8 | no | 0.214 [0.076, 0.476] | 0.250 [0.071, 0.591] | n/a |
| >=320 | 42 | 37 | yes | 0.119 [0.052, 0.250] | 0.270 [0.154, 0.430] | 0.901 |

**1_production_v2 - V-CDF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 764 | 799 | yes | 0.628 [0.593, 0.662] | 0.359 [0.327, 0.393] | 0.503 |
| 40-80 | 62 | 25 | no | 0.597 [0.472, 0.710] | 0.480 [0.300, 0.665] | 0.574 |
| 80-160 | 14 | 1 | no | 0.786 [0.524, 0.924] | 1.000 [0.206, 1.000] | n/a |
| 160-320 | 0 | 0 | no | n/a | n/a | n/a |
| >=320 | 0 | 0 | no | n/a | n/a | n/a |

**1_production_v2 - V-CDF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 702 | 698 | yes | 0.682 [0.647, 0.716] | 0.337 [0.303, 0.372] | 0.489 |
| 40-80 | 16 | 9 | no | 0.188 [0.066, 0.430] | 0.889 [0.565, 0.980] | n/a |
| 80-160 | 2 | 0 | no | 0.000 [0.000, 0.658] | n/a | n/a |
| 160-320 | 21 | 16 | no | 0.048 [0.009, 0.227] | 1.000 [0.806, 1.000] | n/a |
| >=320 | 99 | 102 | yes | 0.071 [0.035, 0.139] | 0.980 [0.931, 0.995] | 0.577 |

**2_dff_only_qualityAug - V-CDF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 764 | 799 | yes | 0.293 [0.262, 0.326] | 0.727 [0.695, 0.757] | 0.503 |
| 40-80 | 62 | 25 | no | 0.355 [0.247, 0.479] | 0.560 [0.371, 0.733] | 0.483 |
| 80-160 | 14 | 1 | no | 0.714 [0.454, 0.883] | 0.000 [0.000, 0.793] | n/a |
| 160-320 | 0 | 0 | no | n/a | n/a | n/a |
| >=320 | 0 | 0 | no | n/a | n/a | n/a |

**2_dff_only_qualityAug - V-CDF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 702 | 698 | yes | 0.142 [0.118, 0.170] | 0.874 [0.847, 0.896] | 0.512 |
| 40-80 | 16 | 9 | no | 0.125 [0.035, 0.360] | 1.000 [0.701, 1.000] | n/a |
| 80-160 | 2 | 0 | no | 0.000 [0.000, 0.658] | n/a | n/a |
| 160-320 | 21 | 16 | no | 0.095 [0.026, 0.289] | 0.812 [0.570, 0.934] | n/a |
| >=320 | 99 | 102 | yes | 0.121 [0.071, 0.200] | 0.863 [0.783, 0.916] | 0.550 |

**3_dff_celebdf_noQualityAug - V-CDF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 764 | 799 | yes | 0.048 [0.035, 0.066] | 0.035 [0.024, 0.050] | 0.992 |
| 40-80 | 62 | 25 | no | 0.048 [0.017, 0.133] | 0.040 [0.007, 0.195] | 0.996 |
| 80-160 | 14 | 1 | no | 0.143 [0.040, 0.399] | 0.000 [0.000, 0.793] | n/a |
| 160-320 | 0 | 0 | no | n/a | n/a | n/a |
| >=320 | 0 | 0 | no | n/a | n/a | n/a |

**3_dff_celebdf_noQualityAug - V-CDF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 702 | 698 | yes | 0.066 [0.050, 0.086] | 0.143 [0.119, 0.171] | 0.967 |
| 40-80 | 16 | 9 | no | 0.000 [0.000, 0.194] | 0.111 [0.020, 0.435] | n/a |
| 80-160 | 2 | 0 | no | 0.000 [0.000, 0.658] | n/a | n/a |
| 160-320 | 21 | 16 | no | 0.000 [0.000, 0.155] | 0.938 [0.717, 0.989] | n/a |
| >=320 | 99 | 102 | yes | 0.010 [0.002, 0.055] | 0.922 [0.853, 0.960] | 0.695 |

**4_dff_celebdf_qualityAug - V-CDF-clean**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 764 | 799 | yes | 0.033 [0.022, 0.048] | 0.059 [0.044, 0.077] | 0.989 |
| 40-80 | 62 | 25 | no | 0.016 [0.003, 0.086] | 0.000 [0.000, 0.133] | 1.000 |
| 80-160 | 14 | 1 | no | 0.214 [0.076, 0.476] | 0.000 [0.000, 0.793] | n/a |
| 160-320 | 0 | 0 | no | n/a | n/a | n/a |
| >=320 | 0 | 0 | no | n/a | n/a | n/a |

**4_dff_celebdf_qualityAug - V-CDF-stress**

| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |
|---|---|---|---|---|---|---|
| <40 | 702 | 698 | yes | 0.056 [0.041, 0.075] | 0.087 [0.069, 0.111] | 0.979 |
| 40-80 | 16 | 9 | no | 0.062 [0.011, 0.283] | 0.000 [0.000, 0.299] | n/a |
| 80-160 | 2 | 0 | no | 0.000 [0.000, 0.658] | n/a | n/a |
| 160-320 | 21 | 16 | no | 0.095 [0.026, 0.289] | 0.062 [0.011, 0.283] | n/a |
| >=320 | 99 | 102 | yes | 0.061 [0.028, 0.126] | 0.196 [0.131, 0.283] | 0.944 |

## 4. Synthetic controls (logic validation, no model)

The controls are model-free logic checks (a softness-only classifier must fail; a quality-independent scorer must not). They ran on the two DeepfakeFace validation sets only; the Celeb-DF notebook did not run them.

| Control | Set | Verdict | Failed/passed checks |
|---|---|---|---|
| NEG_softness_classifier | V-DFF-clean | fail | G1 fail, G2 fail, G3 fail, G4 fail |
| NEG_softness_classifier | V-DFF-stress | fail | G1 fail, G2 fail, G3 fail, G4 fail |
| POS_quality_independent | V-DFF-clean | no_failure | G1 pass, G2 pass, G3 pass, G4 pass |
| POS_quality_independent | V-DFF-stress | no_failure | G1 pass, G2 pass, G3 pass, G4 pass |

## 5. Decision rules of the frozen protocol

* **V1 sensitivity** (NEG control fails; production fails G2 or G3 on a DeepfakeFace validation set): NEG fails on every set = True; production fails G2/G3 = True. **MET.**
* **V2 logic** (POS control does not fail on any set): **MET.**
* **V3 incremental information over the ungated screen** (screen = combined ROC-AUC >= 0.90 on both clean validation sets):

| Model | Passes ungated screen | Gate failures |
|---|---|---|
| 1_production_v2 | no | V-DFF-clean:G1, V-DFF-clean:G2, V-DFF-clean:G3, V-DFF-stress:G1, V-DFF-stress:G2, V-DFF-stress:G3, V-DFF-stress:G4, V-CDF-clean:G1, V-CDF-clean:G2, V-CDF-stress:G1, V-CDF-stress:G2, V-CDF-stress:G3, V-CDF-stress:G4 |
| 2_dff_only_qualityAug | no | V-DFF-clean:G1, V-DFF-clean:G2, V-DFF-clean:G3, V-DFF-stress:G1, V-CDF-clean:G1, V-CDF-clean:G2, V-CDF-stress:G1, V-CDF-stress:G2, V-CDF-stress:G4 |
| 3_dff_celebdf_noQualityAug | yes | V-DFF-clean:G3, V-DFF-stress:G1, V-DFF-stress:G2, V-DFF-stress:G3, V-DFF-stress:G4, V-CDF-stress:G2, V-CDF-stress:G3, V-CDF-stress:G4 |
| 4_dff_celebdf_qualityAug | yes | V-DFF-clean:G2, V-DFF-stress:G1, V-CDF-stress:G2 |

  **MET**: 3_dff_celebdf_noQualityAug, 4_dff_celebdf_qualityAug pass the ungated screen but fail the gate.

## 6. Does the gate add value over the ungated models?

Reading of the numbers above (interpretation, not a new decision rule):

* The gate **confirms and explains** what the ungated metrics already show for the production model and experiment 2: both are at chance on Celeb-DF validation (AUC about 0.50) and the SRM-only stream calls almost every Celeb-DF face fake. The ungated screen already rejects them; the gate adds the *reason* (stream collapse, G1; quality dependence, G2).
* For models 3 and 4 the ungated screen passes (AUC about 0.99 on Celeb-DF validation, 0.93 on DeepfakeFace validation), yet the gate fails them: mostly on the **degraded** sets, plus one check each on clean DeepfakeFace validation (model 3: G3 per-bin error deviation; model 4: G2 quality correlation). That is information the clean AUC does not contain: it matches the earlier finding that robustness to blur/down-up/phone-like softening is weaker than clean accuracy, and that the SRM stream's behaviour depends on image quality. Whether those failures translate into worse behaviour on real, unseen footage is **not established here**.
* Clean Celeb-DF validation has one testable bin only (about 94% of crops are in the softest bin), so G3/G4 are *not evaluable* there and models 3 and 4 get 'no failure' on that set by default, not a pass.

## 7. Limitations and inconclusive findings

* **Models 3 and 4 were trained on Celeb-DF training identities.** V-CDF is validation data of the same dataset and generator family (identity-disjoint, not generator-disjoint), so the high CDF numbers say nothing about unseen generators or general video reliability. Celeb-DF results must not be presented as proof of general video reliability.
* **Threshold ranges were chosen while reading E1 results, which included DFF test splits and training sets** (disclosed in the protocol). Stage-0 therefore cannot prove the thresholds are independent of the test measurements; a future gate version has to be derived from validation-only evidence before any E2 candidate is evaluated.
* The degradations are class-blind synthetic proxies for soft/compressed footage; they are not real phone video.
* Several per-bin FPR/FNR estimates rest on 30-100 samples; their Wilson intervals (section 3) are wide, and many checks are marginal. Gate decisions use point estimates by protocol.
* The gate evaluates a model offline on validation data; it is **not** an application feature and nothing was integrated into the production code or API.
* No labelled real-video benchmark exists in this project, so the gate's relation to false positives on genuine phone videos could not be tested here.

## 8. What this does and does not change

* Production model: **unchanged** (no model met the E2/promotion criteria; no checkpoint, threshold or application code was altered).
* E2 training: **not started**. These results are inputs to the E2 go/no-go discussion, not an approval.
