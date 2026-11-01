# Stage-0 gate validation: protocol (FROZEN before any run)

Version 1. This file is written **before** stage-0 is executed; its SHA-256 is recorded in the results report so that any later edit is detectable. Thresholds below are frozen: they are not changed after seeing stage-0 results (a change = a new protocol version, reported as such).
Scope (approved): inference-only validation of the per-bin acceptance gate defined in `docs/E2_EXPERIMENTAL_DESIGN.md` section 6, on models 1-4. **No** training or fine-tuning, no checkpoint modification, no threshold tuning on test data, no application integration, no private personal media, no change to any Kaggle notebook or dataset (a new inference-only notebook may be created), no production change.

## 1. Candidates (inference only)
1 Production v2 (`verifai_v2_best.pth`, sha256 prefix 3b947755); 2 DFF-only + quality augmentation (dff_qa_R1); 3 DFF+Celeb-DF, no augmentation (cdf_C0_noQA); 4 DFF+Celeb-DF + quality augmentation (cdf_C1_qa). Threshold 0.5 everywhere.

## 2. Validation split V (exact)
| Set | Source | Content | Notes |
|---|---|---|---|
| V-DFF-clean | DFF **val** (identity-disjoint from train and test) | 239 real / 232 fake crops | clean |
| V-DFF-stress | the same DFF val crops | each crop receives one of the 7 fixed degradations of `training/degrade.py` ("mixed": blur 1.5, blur 2.5, down/up 0.5, down/up 0.3, JPEG q25, low-light+noise, phone_soft), deterministic, class-blind | **synthetic**; sharpness is re-measured on the degraded crop |
| V-CDF-clean | Celeb-DF **val** identities (9, disjoint from train and test), all val videos incl. YouTube-real | about 840 real / 825 fake crops | computed inside a private Kaggle notebook; crops stay in /tmp |
| V-CDF-stress | the same Celeb-DF val crops with the same deterministic degradations | synthetic | same notebook |
**Never used:** DFF internal test, DFF cross-generator test, Celeb-DF test, any private media. (The E1 JSON files contain test results; the gate code in this stage reads none of them.) Neutral reference for single-stream scores: 300 class-balanced clean DFF **train** crops (same as E1).

## 3. Quality measure and bins (fixed in E1, unchanged)
Sharpness = variance of the Laplacian of the grey 224x224 crop as the model sees it; bins `<40, 40-80, 80-160, 160-320, >=320`.

## 4. Gate checks and thresholds (frozen)
Applied per model and per set in V. Scores: combined, RGB-only and SRM-only (the other stream neutralised).
| Check | Rule | Evaluability |
|---|---|---|
| G1 stream sanity | RGB-only and SRM-only each: ROC-AUC >= 0.70 **and** FPR <= 0.50 | always |
| G2 stream-quality independence | \|Spearman(sharpness, score)\| <= 0.25 among reals and among fakes, for RGB-only and for SRM-only | always |
| G3 per-bin stability (combined score) | in every **testable** bin (>= 30 real and >= 30 fake): \|FPR_bin - FPR_all\| <= 0.15 and \|FNR_bin - FNR_all\| <= 0.15 | needs >= 2 testable bins, else "not evaluable" |
| G4 within-bin discrimination (combined) | within-bin ROC-AUC >= 0.80 in every testable bin | needs >= 2 testable bins, else "not evaluable" |
| G5 data balance (report only) | softness-alone ROC-AUC of the training distributions (from E1 train: DFF 0.712, Celeb-DF 0.618): not a model check | reported, not gating |
Decisions use **point estimates**; each is reported with Wilson / bootstrap CIs and the counts, and a check is flagged **marginal** when its CI crosses the threshold. "Not evaluable" is never counted as a pass.
**Model verdict:** FAIL if any evaluable check fails on any set; PASS if no check fails, G1 and G2 are evaluated on every set, and G3/G4 are evaluable on at least one set; otherwise INCONCLUSIVE.

## 5. Synthetic controls (logic validation, no model)
Built from the V-DFF crops' labels and measured sharpness:
* **NEG control "softness classifier":** score = sigmoid(3 * (ln(median sharpness) - ln(sharpness))) (softer -> FAKE), used for combined, RGB-only and SRM-only. Must **FAIL** the gate.
* **POS control "quality-independent scorer":** score = clip(0.5 + 0.7 * (label - 0.5) + N(0, 0.12), 0, 1), noise independent of quality (fixed seed), for all three scores. Must **PASS**.
If NEG passes or POS fails, the gate is mis-specified and is reported as such (not tuned).

## 6. Decision rules for "does the gate add value?" (frozen)
* **V1 sensitivity:** NEG control FAILS; and production FAILS at least one of G2/G3 on V-DFF-clean or V-DFF-stress.
* **V2 logic:** POS control PASSES.
* **V3 incremental information over ungated metrics:** the **ungated screen** is "combined ROC-AUC >= 0.90 on V-DFF-clean and on V-CDF-clean". The gate **adds value** if at least one model passes the ungated screen but FAILS the gate on checks that are measurable from validation data alone (G1-G4), i.e. the gate flags something the aggregate metric does not. If no such model exists, the gate adds no information over the ungated screen in this stage.
* **External consistency (descriptive only, never used to set thresholds):** after the run, gate verdicts are compared with the already-published test findings (`FOUR_MODEL_COMPARISON.md`, `E1_QUALITY_BIN_DIAGNOSTIC.md`).
**Disclosure (leakage):** the ranges of the thresholds in section 4 were chosen while reading E1 results, which included DFF **test** splits and the training sets. They are not derived from any stage-0 validation number and are frozen as written; stage-0 therefore cannot prove independence from the E1 test measurements, and the report says so. A future gate version must be derived from validation-only evidence before E2 candidates are evaluated.

## 7. Outputs
`docs/results/stage0_gate/*.json` (per model, per set: all numbers, counts, CIs) and `docs/STAGE0_GATE_VALIDATION.md` (model-by-model results, per-bin tables, verdicts, limitations).
