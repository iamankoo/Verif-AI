# Research directions after the video false-positive investigation

Status: proposal for review. Production stays Model 1 (`verifai_v2_best.pth`, unchanged). None of the experimental models is recommended for production. Evidence: `docs/FOUR_MODEL_COMPARISON.md` and `docs/results/four_model_comparison/`.

## 1. What the investigation established (and what it did not)
| Finding | Evidence |
|---|---|
| The production model's **SRM stream alone says FAKE for about 97 % of both real and fake Celeb-DF faces**, and its RGB stream alone says REAL for most; the combination is a coin flip (video accuracy 0.483, ROC-AUC 0.466). | stream breakdown, section 5 of the comparison |
| The same SRM stream over-flags soft / degraded faces on DeepFakeFace (SRM-only FPR 0.51 clean, 0.89 under mixed degradation). | same |
| In the training data the **fakes are systematically softer than the reals** (median Laplacian variance 114 vs 228) - a quality shortcut. The unseen generator (text2img) is *sharper* than its reals, which is where recall collapses. | measurements during the investigation |
| Adding Celeb-DF removes the false positives on held-out Celeb-DF video **but costs unseen-generator recall** (FNR 0.36 -> 0.52 / 0.63). Quality augmentation alone (model 2) quietens the model without making it discriminating on face-swap video (FNR 0.82). | comparison, section 6 |
| Random degradation augmentation is therefore **not** the path forward; it trades one error for the other. | all of the above |

Not established: whether anything here helps phone-recorded genuine video in general (two private clips were used only as qualitative cases), and any cross-dataset claim for Celeb-DF-trained models.

## 2. Goal and acceptance criteria (proposed; to be agreed before any run)
Reduce genuine-media false positives **without significantly increasing missed deepfakes**. Concretely, relative to production, on held-out data: unseen-generator FNR increase **<= 0.05 absolute**, internal-test FNR increase **<= 0.03**, and genuine-video FPR **<= 0.10** on a labelled video set that includes phone-like and compressed footage - each with confidence intervals, selected on validation only, never on the test sets or the private videos.

## 3. Directions
**D1. Training data: balanced across source, quality and manipulation method.**
* Quality must stop being a proxy for the label: for every sharpness / resolution / compression stratum there should be comparable amounts of real and fake. Measure the label-vs-quality dependence of any candidate dataset *before* training (a one-hour, no-GPU check).
* More manipulation families than today's two training generators plus Celeb-DF's single face-swap method. Candidate public sources (each needs its licence / access terms checked first; several require signed agreements, which is why they were not used originally): FaceForensics++, DFDC, and multi-method collections. Third-party Kaggle mirrors have unclear licences (the Celeb-DF copy used here is "unknown"); prefer official releases.
* Genuine phone-like footage with consent for training negatives (never the two private evaluation clips).

**D2. Quality-aware training that does not suppress detection.** Instead of random degradation: (a) quality-stratified sampling weights so each quality bin is label-balanced; (b) *paired* degradation - DeepFakeFace is paired (a real image and its fake counterparts), so the same degradation is applied to both members of a pair and a consistency term asks for equal quality-invariant features; (c) an invariance head (gradient reversal) that predicts the quality bin from the penultimate features and is trained adversarially, so the label head cannot use quality. Success is measured by quality-stratified FPR/FNR, not average accuracy.

**D3. RGB / SRM stream behaviour.** The SRM stream saturates on video frames. Hypotheses to test, one change at a time: (a) auxiliary single-stream heads plus stream dropout so each stream must be informative on its own; (b) per-image normalisation of the noise residual so that overall detail level does not dominate; (c) replacing the three fixed kernels with learned high-pass filters. Use `training/stream_ablation.py` as a **training-time gate**: a run is rejected if either stream alone has FPR > 0.5 on held-out data from a different source.

**D4. Evaluation protocol.** Leave-one-generator-out and leave-one-dataset-out; identity- and source-video-disjoint splits everywhere; video-level evaluation grouped by source video; report operating curves (FPR at fixed FNR, and the reverse), quality-stratified results and bootstrap CIs over identities / videos. Larger held-out sets than 9 identities / 147 videos. The model-selection rule is fixed in writing before the run and uses validation data that contains several generators and quality levels.

**D5. Product-side mitigation (not a model change; needs separate approval).** An explicit INCONCLUSIVE outcome for low-detail faces, with the cut-off taken from the training distribution (the 10th percentile of real training crops) and checked against a validation set - an honest interim safeguard while the model is improved.

## 4. Proposed experiment ladder (cost on one T4)
| # | Experiment | Needs | Cost |
|---|---|---|---|
| E1 | Per-quality-bin FPR/FNR of all current checkpoints and label-vs-quality dependence of every available dataset | existing checkpoints and crops | ~1 h, mostly CPU |
| E2 | D3: auxiliary stream heads + stream dropout on DFF + Celeb-DF; stream-ablation gate | Kaggle | ~45 min per run |
| E3 | D2: quality-stratified sampling + invariance head, no random degradation | Kaggle | ~45 min per run |
| E4 | D1: add further manipulation methods / genuine phone-like data | **access to legitimately obtainable data** (blocked until agreed) | data-dependent |
| E5 | D4: leave-one-dataset-out and leave-one-generator-out, with video-level evaluation | E4 data | ~2-3 h |
Each run: seed fixed, selection rule pre-declared, test sets touched once, results written with exact counts and confusion matrices, and a stream breakdown. **No more random augmentation sweeps.**

## 5. Risks
* Data access and licensing may limit D1; the plan should not depend on unofficial mirrors.
* Within-dataset scores (like Celeb-DF's 0.98) overstate generalisation; only cross-dataset results count.
* The compute budget is shared (30 GPU-hours per week on Kaggle); the ladder above fits within it for E2-E3.
* A model that reduces FPR on genuine video by shifting scores toward REAL is not an improvement; every claim must be paired with the matching FNR.
