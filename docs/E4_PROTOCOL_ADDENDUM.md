# E4: source-balanced variant of E3 (pre-registered addendum)

Written after the E3 results (`docs/E3_RESULTS.md`) and **before any E4 model was trained**. E3A and E3B detected the modern held-out fakes and kept Celeb-DF video accuracy, but their DeepFakeFace detection was as poor as Model 4's, so criterion 2 of the frozen rule failed.

**Hypothesis.** The DeepFakeFace loss comes from source imbalance: DeepFakeFace is about 5% of the training fakes (1,145 of 21.6k) while Celeb-DF and DeepDetect fakes dominate. Repeating the DeepFakeFace training rows should recover DeepFakeFace detection without losing the other gains. (This follows the repeat-cap idea of E2: per-sample repeat factor at most 4.)

**Change relative to E3A, and nothing else:** every DeepFakeFace *training* row (real and fake) is repeated 4 times in the training manifest. Validation and all test splits are untouched; same data, seed, recipe (ImageNet init, quality augmentation 0.5, AdamW 1e-4, batch 32, 12 epochs, patience 4) and selection rule (validation only). The notebook prints the unique and repeated row counts and the effective sample size (sum r)^2 / sum r^2.

**Evaluation and decision.** Identical splits, identical protocol, identical frozen promotion rule as in `docs/E3_PROTOCOL.md` section "Promotion rule" (criteria 1-3, unchanged). Model 4 and v2 are re-evaluated inside the same run as references. One candidate only, so there is no selection step. If the rule is not met, Model 4 stays the default and E4 is reported as a failed attempt; the criteria are not relaxed. The same limits as E3 apply (`modern_test` is not generator-disjoint; its reals are FFHQ only; no AI-video claim).
