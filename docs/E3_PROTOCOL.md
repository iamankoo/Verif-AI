# E3: modern-generator training experiment (pre-registered protocol)

Written **before any E3 model was trained**. It replaces E2 for the "modern AI-generated faces" goal (E2 had no modern generators). Nothing below is changed after results are seen.

## Question
Can adding modern-generator fakes and more real-face sources to the Model 4 recipe (DeepFakeFace + Celeb-DF + quality augmentation) detect modern AI-generated faces **without** losing what Model 4 does well (genuine camera-style video) and without the DeepFakeFace detection loss that Model 4 showed?

## Data (licence review: `docs/DATASET_LICENCE_REVIEW.md`; inspection evidence: `docs/results/data_inspect/`)
| Source | Use | Label | Rights basis |
|---|---|---|---|
| DeepFakeFace (existing) | train / val / `test_internal` / `test_crossgen` (unseen generator) | as before | as before |
| Celeb-DF v2 (existing) | train / val / `cdf_test` (held-out identities, video level) | as before | research use, as before |
| FFHQ (via the "Real vs AI Generated Faces" copy, folder `data_source/ffhq`) | real faces: train 8,000 / val 1,000 / `modern_test` 1,500 (file-disjoint, seed 42) | 0 | upstream CC BY-NC-SA 4.0: non-commercial research with attribution (NVlabs / Flickr photographers) |
| SFHQ (same copy, `fake/sfhq/pt1-4`) | train 3,000 / val 500 / `modern_test` 700 | 1 | upstream CC0 |
| DeepDetect-2025, **fake side only** (`ddata/train/fake`, `ddata/test/fake`) | train 9,000 / val 1,000 (from its train folder) / `modern_test` 2,500 (from its test folder); face-filtered with the production detector | 1 | creator-generated outputs, declared Apache-2.0; its real side has unspecified origin and is **not used** |
Excluded: RealVsFake-162k (CelebA/FFHQ reals contradict its CC0 label; PNG-vs-JPG shortcut), DeepFakeFusion-399K (Celeb-DF, FaceForensics++ and other sources; overlaps our Celeb-DF evaluation), the `faceswap`, `stable_diffusion` and `thispersondoesnotexist` folders of the copy (provenance not established), DeepDetect's real side. No private or personal media is used anywhere.

## Leakage control
File-disjoint seeded splits; 64-bit dHash near-duplicate removal (Hamming <= 3) of validation crops against all training crops (all sources) and of test crops against training and validation crops; counts removed are reported. The face detector is the production one.

## Candidates (same recipe as Model 4: ImageNet init, AdamW 1e-4, batch 32, 12 epochs max, patience 4, selection by validation clean/degraded mean ROC-AUC, seed 42)
* **E3A** quality augmentation 0.5 (the Model 4 setting) on the enlarged data
* **E3B** no quality augmentation
Reference models evaluated with the identical protocol and code: **Model 4** (`cdf_C1_qa`) and **v2**.

## Evaluation (all at the fixed threshold 0.5, no threshold tuning; Wilson / bootstrap intervals; exact counts)
Splits: `val`, `test_internal`, `test_crossgen`, `cdf_test` (frame and video level), `modern_test`; clean and class-blind "mixed" degradation. Per-generator FPR/FNR from `metrics.json -> per_generator`. Candidate selection between E3A and E3B uses **validation only**.
Known limits written in advance: `modern_test` reals are FFHQ only while its fakes are SFHQ and DeepDetect (a real-source / fake-source confound: an FFHQ-vs-generator style shortcut cannot be excluded); DeepDetect has no per-image generator labels, so no generator-disjoint test inside it exists; DeepDetect test and train fakes come from the same generator families (in-distribution, not source-disjoint). The only unseen-generator test remains DeepFakeFace `test_crossgen`. Video evidence is Celeb-DF only; no labelled modern AI-video data was available, so **no AI-video claim can be made**.

## Promotion rule (frozen; the selected E3 model replaces Model 4 as default only if ALL hold)
1. **Genuine media is not degraded:** on `cdf_test` video level, FPR <= Model 4's FPR + 0.03 and accuracy >= 0.95.
2. **No loss on DeepFakeFace:** on `test_internal` and `test_crossgen` (clean), FNR <= v2's FNR + 0.05 (absolute) and FPR <= 0.20.
3. **Modern-generator benefit:** on `modern_test` (clean), balanced accuracy exceeds Model 4's by >= 0.10 absolute, with the bootstrap 95% interval of the difference above 0.
If any criterion fails Model 4 stays the default, the E3 checkpoint is preserved, and the report states what failed. Criteria are never relaxed after seeing results.

## Final sanity check (separate from all of the above)
After the model is frozen, the private demonstration files are scored locally and never uploaded, used for training, threshold choice or selection; no scores or file names are published.
