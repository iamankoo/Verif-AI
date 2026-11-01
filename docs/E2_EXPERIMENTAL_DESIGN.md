# E2 experimental design: stream-supervised, quality-balanced training (proposal, not run)

**Status: design for review only.** Nothing in this document has been run. No training, no per-bin gate code, no checkpoint change, no threshold tuning and no application change is authorised until this design is approved. Production stays Model 1 (`verifai_v2_best.pth`). No private personal media or scores are used anywhere in E2.
Inputs: `docs/FOUR_MODEL_COMPARISON.md`, `docs/E1_QUALITY_BIN_DIAGNOSTIC.md`, `docs/RESEARCH_DIRECTIONS.md`.

## 1. Why E2: what E1 established
| E1 finding | Consequence for the design |
|---|---|
| In DFF, softness alone predicts FAKE (ROC-AUC 0.71, CI 0.70-0.73); fake share 26% (sharp) -> 85% (soft). It **reverses** for the unseen generator (AUC 0.43). | Training data must be balanced so quality carries no label information, and unseen-generator recall must be protected explicitly. |
| The SRM stream carries the quality dependence (Spearman of sharpness with SRM-only score -0.36 / -0.52 production; RGB-only about 0). | Each stream must be forced to carry label evidence on its own; the SRM stream must not be free to act as a quality detector. |
| Celeb-DF lives almost entirely in one very soft bin (94% of crops < 40 vs 8% of DFF training crops). | Celeb-DF cannot teach quality-robustness; it can only supply soft reals and one face-swap method. Overfitting to that regime is a primary risk. |
| Random degradation (models 2, 4) did not remove the quality dependence. | E2 changes the *objective and the sampling*, not only the inputs. |

**Hypotheses (fixed in advance).**
H1: auxiliary single-stream supervision plus stream dropout makes each stream label-informative and lowers |Spearman(sharpness, stream score)|.
H2: quality-stratified, method-balanced sampling lowers the label-quality dependence of the training distribution to near zero and flattens per-bin FPR/FNR.
H3: H1 and H2 together reduce genuine-video false positives **without** the unseen-generator recall loss seen in models 3 and 4.
Null: no difference from the baseline recipe within the pre-declared margins (section 8).

## 2. Models and baselines (identical code path, data splits, seeds, 15-epoch cap, patience 4, AdamW 1e-4, batch 32)
| ID | Description | Purpose |
|---|---|---|
| P | Production v2 (frozen, evaluated only) | reference |
| B0 | Recipe baseline: same recipe as model 3 (DFF + Celeb-DF, ImageNet init, no augmentation) retrained with the E2 code and seeds | isolates the effect of each E2 component from recipe/seed noise |
| B1 | Model 4 recipe (quality augmentation p = 0.5) retrained | the "random augmentation" baseline |
| S-RGB, S-SRM | **Single-stream networks** (EfficientNet-B0 on RGB only; on the SRM residual only), same data | shows what each stream can do alone and whether the SRM stream's shortcut is intrinsic to the fixed filter bank |
Existing models 1-4 are kept as evaluated numbers (not retrained) for context.

## 3. Components
### 3.1 Auxiliary single-stream heads
Two linear heads on the pooled 1280-d features: `head_rgb` (RGB stream) and `head_srm` (SRM stream), alongside the unchanged fusion head (2560 -> 512 -> 1). Loss: `L = L_fusion + lambda_r L_rgb + lambda_s L_srm` (BCE with logits, same class weights). The aux heads are trained on the **un-dropped** features, so each backbone receives its own gradient toward the label. Pre-declared default `lambda_r = lambda_s = 0.3`; a validation-only sensitivity check over {0.1, 0.3, 1.0} for the full model (section 6.1). The heads live only in a training-side subclass; the fusion head and checkpoint keys of the deployed architecture are unchanged, so any later integration is a separate, reviewed step.

### 3.2 Stream dropout
Per training sample: drop the RGB features with p = 0.2, the SRM features with p = 0.2, none with p = 0.6 (never both). A dropped stream's feature vector is replaced by its running class-agnostic mean (EMA of the batch mean, momentum 0.99) - the same neutralisation that E1 and the four-model comparison use for the "RGB-only / SRM-only" scores - and the fusion loss is computed on the result. This trains the fusion head to work with a missing or uninformative stream and keeps training and the evaluation diagnostics consistent. No dropout at inference.

### 3.3 Balanced data across quality bins and manipulation methods
Fixed sharpness bins from E1 (`<40, 40-80, 80-160, 160-320, >=320`). Sources: DFF train (real; fakes: inpainting, insight), Celeb-DF train (real, YouTube-real; fakes: Celeb-synthesis). **Held out and never trained on:** DFF text2img (unseen generator), all test identities.
* **Batch composition:** 50% real / 50% fake. Among fakes, each manipulation method supplies 1/3 (inpainting, insight, Celeb-synthesis) so Celeb-DF (about 8.4k of the 9.6k training fakes) cannot dominate. Among reals, DFF-real and Celeb/YouTube-real supply 1/2 each.
* **Quality-stratified weights:** iterative proportional fitting of sampling weights over the (bin x label) table so that the fake share equals 0.5 in every bin; bins with fewer than 30 samples per class are merged with the neighbouring bin; per-sample repeat factor capped at 4; the effective sample size `(sum w)^2 / sum w^2` is reported and the design is rejected (not tuned) if it falls below 50% of N.
* **Quality-targeted paired degradation (component Q).** DFF is paired (a real image and its fake counterparts). For each pair draw a target bin uniformly, find the blur / down-up strength that brings the *real* crop into that bin (bisection), and apply **the same** parameters to its fake counterpart, so soft bins contain both labels by construction. Celeb-DF crops are not degraded further.
* **Pre-training check (must pass before any GPU time):** on the weighted training distribution, the ROC-AUC of softness alone for FAKE must lie in [0.45, 0.55] (E1 measured 0.712 for DFF train). The realised (source x method x bin x label) counts before and after weighting are published.
* **Known gap:** sharp bins (>= 160) exist only in DFF for both labels, because Celeb-DF has essentially none; E2 cannot create sharp Celeb-DF-like footage. This is a stated limit, not something E2 solves.

## 4. Ablation matrix (each row adds one thing to B0; single seed first, see 7)
| Run | Heads | Stream dropout | Quality-stratified sampling | Paired degradation Q |
|---|---|---|---|---|
| A0 = B0 | | | | |
| A1 | yes | | | |
| A2 | | yes | | |
| A3 | yes | yes | | |
| A4 | | | yes | |
| A5 | yes | yes | yes | |
| A6 | yes | yes | yes | yes |
| C-rand | yes | yes | **random** weights with the same effective sample size | |
C-rand is a negative control: any benefit of A5 over it must come from *quality-targeted* balancing, not from resampling noise. S-RGB and S-SRM and B1 are run alongside.

## 5. Evaluation (frozen before the first run)
Splits are identity-disjoint everywhere; the test sets are touched **once per frozen design** and each access is logged. Counts (from E1 / the comparison): DFF internal test 292 real / 283 fake; DFF cross-generator test 281 / 287; Celeb-DF held-out 1,454 / 895 frames and **91 real / 56 fake videos from 9 identities**.
1. **Frame level and video level, reported separately.** Video score = median of the video's frame scores (single-face videos; equals the production per-track rule); 16 random frames per video.
2. **Unseen generator:** DFF text2img (never trained on). In stage 4 also leave-one-generator-out over {inpainting, insight, Celeb-synthesis} for the baseline and the final candidate.
3. **Cross-dataset (independent of the training source):** train on **DFF only** (all E2 components) and test on Celeb-DF; train on **Celeb-DF only** and test on DFF. These are the only genuinely cross-dataset results available today and are reported as such.
4. **Independent video datasets:** none beyond Celeb-DF is available. If another labelled video dataset is obtained legitimately (licence checked), it is added as a held-out test; until then **no claim of general video reliability is made**, and Celeb-DF results are labelled within-dataset.
5. **Genuine phone video:** an evaluation-only set of consented recordings (many people, devices and lighting; never used for training or selection) would be needed to measure phone-footage FPR; it requires separate approval and is not part of E2. The existing private clips remain out of scope.
6. Every table lists TN/FP/FN/TP, accuracy, precision, recall, F1, ROC-AUC, FPR and FNR with Wilson or bootstrap CIs and the exact n; E1-style per-bin tables (counts, FPR/FNR, within-bin AUC, Spearman per stream) for every candidate.

## 6. Per-bin gate: definition and leak-free validation (specification only; no code is written until approved)
The gate is a **model-acceptance test run on validation data**, not an inference rule. It uses fixed thresholds taken from the ranges E1 observed, declared here, and never adjusted after seeing E2 results (any change = a new, versioned gate, reported).
| Check | Rule (per candidate, on the validation gate set V) |
|---|---|
| G1 stream sanity | with the other stream neutralised, RGB-only and SRM-only each: ROC-AUC >= 0.70 and FPR <= 0.50 (E1: production SRM-only FPR 0.51 clean, 1.00 on Celeb-DF) |
| G2 stream-quality independence | \|Spearman(sharpness, stream score)\| <= 0.25 among reals and among fakes, for each stream (E1: SRM -0.36 to -0.53, RGB <= 0.12) |
| G3 per-bin stability | in every bin with >= 30 crops per class: \|FPR_bin - FPR_all\| <= 0.15 and \|FNR_bin - FNR_all\| <= 0.15; bins below the count are reported "untestable", never passed |
| G4 within-bin discrimination | within-bin ROC-AUC >= 0.80 in testable bins |
| G5 training-distribution balance | softness-alone AUC on the weighted training set in [0.45, 0.55] (pre-training check, section 3.3) |
**Validation without leakage.** V = DFF val (identity-disjoint from train and test) + Celeb-DF val (9 identities, disjoint from train and test) + a **synthetic quality-stress copy** of both (fixed deterministic degradations applied class-blind, flagged as synthetic) so that bins above 80 become testable. Test sets, test identities and the private clips never enter V; bin edges and gate thresholds are fixed before any run; epoch selection and the lambda / dropout sensitivity checks use V only. **The gate is validated before it is trusted:** it is first applied to the existing models 1-4 on V (negative controls: production and the augmentation models are expected to fail G2/G3; positive control: the balanced recipe must be able to pass), and the result is published before E2 runs. If the gate cannot tell the known-bad models from the candidates, it is redesigned before use.
A separate, later question - an inference-time INCONCLUSIVE rule for low-detail faces - is **not** part of E2 and would need its own validation (coverage-vs-error curve on V) and approval.

## 7. Compute plan (Kaggle T4, about 45-50 min per run, 30 GPU-hours per week)
| Stage | Runs | Approx. GPU time | Decision after |
|---|---|---|---|
| 0 | Pre-training checks (G5, balance tables), gate validation on models 1-4 (inference only) | ~1 h | gate and data design frozen |
| 1 | A0-A6, C-rand, S-RGB, S-SRM, B1 - one seed (42) | ~8 h | drop components that do not help on V |
| 2 | Baseline + best two variants, 3 seeds (42-44) | ~6 h | mean +/- sd; choose the candidate on V only |
| 3 | Cross-dataset runs (DFF-only -> Celeb-DF; Celeb-DF-only -> DFF) for baseline and candidate | ~3 h | generalisation evidence |
| 4 | Leave-one-generator-out for baseline and candidate | ~5 h | unseen-method evidence |
| T | **One** test evaluation of the frozen candidate and baseline | ~0.5 h | report, review, approval |
Total about 22 h, spread over at least two weeks. Each stage is a go/no-go checkpoint reviewed with you.

## 8. Statistics and acceptance criteria
* **Primary endpoints (pre-declared):**
  **P1 unseen generator:** FNR on DFF text2img must increase by **no more than 0.05 absolute** vs production (production 0.359, so <= 0.409). **P2 genuine video:** FPR on Celeb-DF held-out real videos **<= 0.10**.
* **Decision rules given the small samples.** P1 is judged on a paired bootstrap over identities (same crops as production): **pass** if the point estimate of the increase is <= 0.05 **and** the upper 95% bound is <= 0.10; otherwise **inconclusive** (not pass). With 287 unseen-generator fakes the 95% Wilson half-width of an FNR near 0.4 is about +/-0.06, so a point estimate alone cannot establish the criterion. P2 is judged on 91 real videos (Wilson half-width up to about +/-0.06): **pass** if the point estimate is <= 0.10 and the Wilson upper bound is <= 0.15; the bound is always reported and **no general video reliability is claimed from this set**.
* **Secondary endpoints:** DFF internal-test FNR increase <= 0.03 vs production; gate G1-G4 pass on V; cross-dataset (stage 3) results; video-level ROC-AUC; per-bin curves.
* **Seed variance and multiplicity:** stage-2 results are mean +/- sd over 3 seeds; a difference is called real only if it exceeds both the paired CI and the seed sd. Two primary endpoints only; everything else is descriptive.
* **Selection:** candidate choice on V only (gate pass, then V performance); never on the test sets or any private media.

## 9. Risks and mitigations
| Risk | Mitigation |
|---|---|
| **Overfitting Celeb-DF:** one face-swap method, 41 training identities, 9 test identities, one quality regime | method-balanced batches (Celeb-DF at most 1/3 of fakes); cross-dataset runs (DFF-only -> Celeb-DF and the reverse); leave-one-generator-out; identity-disjoint splits; early stopping; label all Celeb-DF results as within-dataset |
| **Overfitting the image-quality distribution:** bins are defined by one measure | add HF-noise and a compression-blockiness measure to the per-bin report and to G2/G3; quality-targeted *paired* degradation instead of a single degradation family; list what the gate cannot see |
| **Shortcut migration:** the model drops "softness" and picks up another proxy (resolution history, JPEG grid, alignment margin, background) | per-bin and per-source tables for each new candidate; cross-dataset tests; single-stream baselines to expose stream-specific shortcuts |
| **Validation optimism / test reuse** | gate and thresholds frozen first; single test evaluation per frozen design with an access log; V contains no test identities |
| **Aux heads or stream dropout lower fusion accuracy** | ablation matrix isolates each; B0 and C-rand controls |
| **Reweighting inflates variance** (repeat cap 4, effective sample size) | ESS reported; design rejected below 50% of N; 3 seeds for finalists |
| **Epoch cap** (models 2 and 4 were still improving at the cap) | keep the 15-epoch cap but record the validation curve; if the best epoch is the last, the run is flagged "possibly under-trained" and extended once with the same pre-declared rule |
| **Data licensing / provenance** (Celeb-DF is a third-party mirror, licence "unknown") | used only in private notebooks, no export of frames; state this in every report; prefer official releases for any new data |
| **Compute budget and environment drift** | staged plan with go/no-go; record package versions in every run |

## 10. Explicitly out of scope
No application or checkpoint change; no threshold change (0.5 stays); no inference-time gate; no new data acquisition; no use of private media; no commit or push without your approval of the exact diff.

## 11. What I need from you to proceed
1. Approve, amend or reject this design (especially the gate thresholds in section 6 and the acceptance rules in section 8).
2. Decide whether the stage-0 gate validation on models 1-4 (inference only, about 1 hour) may start first.
3. Say whether to pursue access to a second labelled video dataset in parallel (needed before any claim beyond Celeb-DF).
