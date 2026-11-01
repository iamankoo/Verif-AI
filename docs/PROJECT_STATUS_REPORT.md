# Verif-AI status report: what we are doing and what we are not

Written 2026-11-02, about 01:30. Numbers are measured and traceable to `docs/CURRENT_DEFAULT_MODEL.md`, `docs/FOUR_MODEL_COMPARISON.md` and `docs/STAGE0_GATE_VALIDATION.md`.

## 1. Where the project stands
| Area | Status |
|---|---|
| Application (FastAPI backend, React frontend, image and video analysis, Grad-CAM, PDF report, deletion) | Working. Verified in a real browser: 9 end-to-end tests pass; backend 145 passed / 7 skipped (all skips are environment-gated); frontend 41 passed. |
| UI | Redesigned as a clean, light interface with navigation and tabbed results. Scores are shown as "model score, 0-1", never as a percentage. |
| Default model | **Model 4** (DeepFakeFace + Celeb-DF + quality augmentation), a temporary owner decision. Production **v2** is untouched and is the rollback (`VERIFAI_MODEL_PATH=models/verifai_v2_best.pth`). |
| GitHub | Branches `fix/video-false-positive` and `feat/ui-completion` are pushed. `main` is unchanged (`9064c6b`); nothing is merged. |
| New training | **Not started.** Step 1 (inspecting candidate datasets on Kaggle) is running. |

## 2. What the measurements say
* v2 calls genuine camera video FAKE (video accuracy 0.48 on held-out Celeb-DF; informal manual tests on genuine camera clips showed the same false positives).
* Model 4 fixes that (video accuracy 0.98 on Celeb-DF identities it never saw; genuine camera clips in informal manual tests scored REAL). That figure is within-dataset and is not general video reliability.
* Model 4 is worse than v2 on AI-edited and text-to-image faces: the miss rate on unseen text-to-image faces is 56% against 36% for v2.
* **Neither model detects modern AI-generated content reliably.** Informal manual tests with AI-generated portraits and videos produced many REAL verdicts from both models; modern generators are outside both training sets. These manual tests are not an evaluation and no accuracy figure is drawn from them.
* All four candidate models fail the stage-0 quality gate (`docs/STAGE0_GATE_VALIDATION.md`).
* Heat-maps (Grad-CAM) are an explanation drawn after scoring. Scores are identical with them on or off, so they cannot improve detection.

## 3. What we are doing now
1. **Inspecting candidate training data** (Kaggle notebook `verif-ai-data-inspect`, CPU, statistics only, no images in the output). Candidates were chosen for a permissive Kaggle-listed licence and for containing modern generators: RealVsFake 162k (CC0), DeepFakeFusion-399K (MIT), DeepDetect-2025 (Apache 2.0), Real vs AI Generated Faces (MIT).
2. **Looking for shortcuts** before training: whether real and fake differ in file format, resolution, file size or sharpness. A dataset with such a difference teaches the model the shortcut instead of detection. This is exactly how the earlier models failed.
3. **Then**: build a balanced training set from the clean datasets plus the existing DeepFakeFace and Celeb-DF data, train on Kaggle, and evaluate on held-out data that was never used for training or selection. The existing acceptance criteria stay unchanged.

## 4. What we are not doing, and why
| Not doing | Reason |
|---|---|
| Training on private test media | Personal media never goes to Kaggle or any cloud, and training on the files used for demonstration would only memorise them. Such files are used only as a final local sanity check, never for training or selection. |
| Promising 95% accuracy | The data and the new model do not exist yet; no measurement supports the claim. Accuracy will be reported only for held-out, labelled data, with confidence intervals and exact counts. |
| Adding heat-maps as a model input | They carry no new information and would need a different architecture and a retrain. |
| Changing thresholds to "fix" results | That only trades missed fakes for false alarms, and tuning on test data is not valid. |
| Using non-commercial, "Unknown"-licence, or tiny video datasets | The OpenFake set is licence "Unknown"; the two video sets are CC BY-NC-ND and about 67 MB. Excluded for now. |
| Running the E2 design as written | It has no modern generators, so it would not address modern AI-generated faces. It must be revised first. |
| Merging into `main` or replacing the production checkpoint | Needs your explicit approval. |
| Claiming the system detects "all AI images or videos" | It is a face-based detector with the documented limits above. |

## 5. Risks and limits
* **Time.** Training and evaluation will realistically finish after the presentation. For the presentation, the safe plan is real camera video with Model 4 plus an honest statement of the limitation. Showing AI-generated photos would likely produce wrong REAL results.
* **Licence uncertainty.** The Kaggle licence is the uploader's label. The rights to the underlying photos were not verified.
* **Video.** A frame-level model on still-image data may still miss modern AI video (a different artefact type), and no licensed modern AI-video dataset is available. This may need to be reported as unsolved.
* **Success is not guaranteed.** The new model may end up at a different trade-off, not above 95%.

## 6. Decisions I need from you
1. Approve the datasets you are comfortable with, given the licence caveat above (my suggestion: RealVsFake 162k, DeepFakeFusion-399K, DeepDetect-2025; I will check the inspection results before using any of them).
2. Whether AI-video detection should be dropped from the goal, or reported as an open limitation, if no usable video data is found.
3. For tomorrow: demo with Model 4 on real camera video and state the limitation openly.
