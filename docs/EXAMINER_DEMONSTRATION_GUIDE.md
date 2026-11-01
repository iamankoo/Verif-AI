# Examiner Demonstration Guide

Everything you need to present and defend Verif-AI: how it works, a feature checklist, the evaluation results with plain-language explanations, a 10–15 minute speaking script and 55 viva questions with answers. Numbers come from `docs/MODEL_EVALUATION_REPORT.md` and `docs/PROJECT_COMPLETION_REPORT.md` and were re-checked against `docs/results/*.json`. Setup: [QUICK_START.md](QUICK_START.md). Upload steps: [IMAGE_VIDEO_DEMONSTRATION_GUIDE.md](IMAGE_VIDEO_DEMONSTRATION_GUIDE.md).

---
## Part 1. How the project works

### 1.1 What it does and why
Verif-AI takes a photo or a video of a person and answers: *does the face show patterns that a trained model associates with fabricated faces?* It returns REAL, FAKE or INCONCLUSIVE per face and per media file, shows where the model looked (Grad-CAM), and writes a PDF forensic report. **Problem:** synthetic faces (face-swaps, inpainting, text-to-image) are easy to create and hard to spot by eye. **Solution:** a two-stream CNN that looks at the picture *and* at its noise residual, wrapped in a usable web application with honest uncertainty handling.
**Scope (say this early):** faces only; trained and tested on one openly licensed dataset; scores are uncalibrated; a REAL verdict is not proof of authenticity; it is not a general "AI-image detector".

### 1.2 Architecture and technologies
```
Browser (React UI)  ──HTTP/JSON──►  FastAPI backend (backend/verifai/api/app.py)
                                        │
        upload validation (_save_upload) ─► Engine (engine.py)
                                        │
   image:  analyze_image()  /  video: analyze_video() (background thread)
                │  FaceDetector (faces.py, MTCNN)   │ IoUTracker (video)
                ▼                                   ▼
   preprocess.py: crop 224 · JPEG q95 round-trip · ImageNet normalise
                ▼
   model.py: TwoStreamNet  ──► fake-score 0..1 per face
                ▼
   scoring.py: classify() (0.5 + INCONCLUSIVE band), aggregate(), temporal_stats()
   explain.py: Grad-CAM          store.py: result + evidence images
                ▼
   JSON result ─► UI result page      report.py: build_pdf() ─► PDF
```
| Layer | Technology (from `requirements.txt` / `package.json`) |
|---|---|
| Model | PyTorch, torchvision, `efficientnet-pytorch` 0.7.1 |
| Face detection | `facenet-pytorch` MTCNN |
| Image/video I/O | Pillow, OpenCV (headless), NumPy |
| Explainability | `grad-cam` (Grad-CAM) |
| API | FastAPI, Uvicorn, python-multipart, Pydantic |
| PDF | ReportLab |
| Frontend | React 18, TypeScript, Tailwind CSS, Vite 5, Vitest |
| Packaging | Dockerfile (multi-stage: Node build → python:3.10-slim runtime, non-root, healthcheck) |
| Training | Kaggle GPU (Tesla T4) via the Kaggle CLI; `backend/training/` |

### 1.3 The model (`backend/verifai/model.py`)
`TwoStreamNet` has two **EfficientNet-B0** encoders. `spatial_net` sees the normal RGB face. `srm_layer` (`SRMLayer`) is a **fixed, non-trainable** 3×3 high-pass filter bank (3 kernels: a second-order residual filter, an 8-neighbour Laplacian and a 4-neighbour Laplacian) whose output — the *noise residual*, where manipulation traces are easier to see — goes to `noise_net`. Each encoder gives 1280 features after global average pooling; they are concatenated (2560) → `Linear(2560,512)` → ReLU → Dropout(0.5) → `Linear(512,1)` → Sigmoid. The output is the **fake-score**: close to 1 = model thinks fabricated, close to 0 = real. 11,972,954 stored values; label convention real = 0, fake = 1. (The old README's "EfficientNet-B4 / 30 SRM kernels / TensorFlow" did not match the code and was removed.)

### 1.4 Dataset and splits
**DeepFakeFace (DFF)**, openrail licence: real photos from IMDB-WIKI; fakes made by Stable-Diffusion *inpainting*, InsightFace *face-swap* and Stable-Diffusion *text-to-image*. Only a seeded sample was downloaded and balanced (a real image and one fake counterpart per sampled image). Splits are **identity-disjoint**: train 2,315 · validation 471 · internal test 575 · cross-generator test 568 faces. Train/val/internal test contain inpainting + insight fakes; the **cross-generator test uses text-to-image fakes the model never saw, of people it never saw**. Celeb-DF / FaceForensics++ / DFDC were not used (they require signed agreements; unofficial mirrors were rejected). Details: `docs/DATASET_DOCUMENTATION.md`.

### 1.5 Image detection internally
`POST /api/analyze/image` → `_save_upload` (checks extension, size, writes under a random name) → `Engine.run_image` → `analyze_image` (`image_pipeline.py`): `load_image_rgb` (EXIF rotation, RGB, size checks) → `FaceDetector.detect` (MTCNN on a ≤1280 px copy; keeps faces with confidence ≥ 0.90 and side ≥ 32 px, max 10) → crops 224×224 with margin 20 → `preprocess_faces` (**JPEG q95 round-trip** because training crops were stored as JPEG q95, then ImageNet mean/std) → `score_crops` → `TwoStreamNet` → per-face fake-score → `classify()` → **image score = max face score** → Grad-CAM (`explain.py`: last block of the RGB encoder, target = the fake output) → annotated image → `AnalysisStore.save_result` → JSON back to the browser; the upload file is deleted in a `finally`.

### 1.6 Video detection internally
`POST /api/analyze/video` → upload saved → `Engine.submit_video` queues a job on a **single worker thread** and returns **202** with an id. The worker runs `analyze_video` (`video_pipeline.py`): `probe_video` → `plan_sampling` (≈2 fps, evenly spaced over the whole clip, ≤ 64 frames, ≤ 900 s) → detect faces in each sampled frame → `IoUTracker` links the same person across frames (tracks) → score every face crop in batches → per-track **median** (tracks need ≥ 3 face frames) → **video score = highest track score** (`video_score_from_frames`) → `classify()`. It also computes `temporal_stats` (fraction of frames > 0.5, longest run, sign flips) and `suspicious_segments` (consecutive frames > 0.5 → time ranges), and keeps the 6 highest-scoring frames with Grad-CAM as evidence. The browser polls `GET /api/analyses/{id}/status` every second, then loads `GET /api/analyses/{id}`.
**This video logic is a design choice, not a validated method** — no labelled video dataset was available.

### 1.7 Forensic analysis and the PDF
"Forensic analysis" here means *evidence presentation*: per-face scores and boxes, the model-input crop, Grad-CAM heat-map, frame timeline, suspicious time ranges, tracks, file SHA-256, model checkpoint SHA-256, warnings (small faces, sparse sampling, low-resolution video) and an explanation of the decision rule. `report.py: build_pdf()` assembles these with ReportLab (verdict banner, details table, images, tables, timeline drawing, explanation, warnings). The PDF is generated on request by `GET /api/analyses/{id}/report.pdf`.

### 1.8 Frontend ↔ backend and model loading
`frontend/src/api.ts` calls the REST endpoints; in development Vite proxies `/api` to port 8000, in the demo the backend serves the built UI itself (`VERIFAI_FRONTEND_DIR`), so everything is same-origin. At backend start (`lifespan` in `app.py`) `load_model` reads `backend/models/verifai_v2_best.pth` with `torch.load(..., weights_only=True)` (safe loading), builds `TwoStreamNet`, loads the weights with `strict=True`, moves to GPU if available else CPU, computes the checkpoint SHA-256 and stores model info. If loading fails the server starts in *degraded* mode (`/api/health` says so; analysis returns 503).

### 1.9 What happens when a user uploads media (one paragraph)
Select file → browser pre-checks type/size (`validateFile`) → preview → **Analyse** → multipart upload with progress → backend re-validates and saves under a random name → image: analysed immediately; video: queued and processed in the background → face detection → preprocessing → model → decision rule → evidence images + JSON stored → upload deleted → result page → optional PDF download → **Delete results** wipes the stored evidence.

---
## Part 5. Feature checklist (what to show, in a sensible order)

| # | Feature | Where | Action | Examiner should see | What to say |
|---|---|---|---|---|---|
| 1 | Homepage | `http://127.0.0.1:8000/` | open | "See beyond the pixels", upload box, Model information, Recent analyses | everything runs locally; no third-party AI |
| 2 | API health | `/api/health` | open in a new tab | `{"status":"ok","model_loaded":true,"device":"cpu",…}` | liveness + model status |
| 3 | Model information | right column / `/api/model` | read | model name v2.0, threshold 0.5, ±0.15 band, checkpoint hash | traceability of the exact weights |
| 4 | Invalid file rejection | upload box | choose a `.txt` | red message "Unsupported file type…" | validation on client and server (415/413/422) |
| 5 | Image upload | Browse files → Analyse image | pick a face image | preview, then progress | file stays on the laptop copy; deleted after analysis |
| 6 | REAL/FAKE result + score | result page | read | verdict badge, Real/Fake bars, distance from 0.5 | uncalibrated score; INCONCLUSIVE band |
| 7 | Face detection information | "Detected faces", "Per-face results" | point at box, table | coloured box + `#1 REAL 0.03`; detector confidence | MTCNN; max-face rule |
| 8 | Forensic analysis | Grad-CAM crop pair, Explanation, Warnings | point | model input vs heat-map | regions that influenced the RGB stream; not proof |
| 9 | PDF report | **Download forensic report (PDF)** | click, open file | verdict banner, hashes, evidence | reproducible, traceable report |
| 10 | Video upload + async job | Browse files → Analyse video | pick video | progress panel with stage | background job, 202, polling |
| 11 | Video result | result page | scroll | summary, frame timeline, suspicious timestamps, tracks, evidence frames | sampling, tracking, aggregation (unvalidated) |
| 12 | Recent analyses | right column | click an entry | reopens earlier result | stored 72 h |
| 13 | Deletion | **Delete results** | click | returns home, entry gone | privacy; `DELETE` → 204 |
| 14 | API docs (optional) | `/docs` | open | Swagger UI listing all routes | documented REST API |
| 15 | Tests (optional) | terminal | `python -m pytest tests -q`, `npm test` | 108 passed/6 skipped; 36 passed | engineering quality |

**Not implemented / do not claim:** user accounts or login; a database (results are JSON files); live webcam analysis; audio deepfake detection; batch upload of many files; calibrated probabilities; a validated video accuracy; analysis of non-face content; mobile app. A Docker image exists and was verified in GitHub Codespaces, but not on Windows Docker Desktop.

---
## Part 6. Evaluation results (verified against the reports)

**Final model:** `verifai_v2_best.pth` — ImageNet-initialised two-stream EfficientNet-B0 + SRM stream, trained on Kaggle (Tesla T4), best epoch 9, chosen by validation ROC-AUC. Threshold **0.5** (fixed). All numbers are **face-level**, one dataset.

| Split | Purpose | n | Accuracy (95 % CI) | Precision | Recall | F1 | ROC-AUC (95 % CI) | PR-AUC | FPR | FNR |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| Validation | used to select the model (optimistic) | 471 | 0.877 (0.847–0.905) | 0.843 | 0.922 | 0.881 | 0.948 (0.928–0.965) | 0.944 | 16.7 % | 7.8 % |
| **Internal test** | unseen identities, seen generators | 575 | **0.875** (0.848–0.902) | 0.860 | 0.890 | 0.875 | **0.941** (0.921–0.960) | 0.942 | 14.0 % | **11.0 %** |
| **Cross-generator test** | unseen identities **and** unseen generator | 568 | **0.746** (0.716–0.778) | 0.818 | 0.641 | 0.719 | **0.849** (0.818–0.879) | 0.841 | 14.6 % | **35.9 %** |

(Precision/recall/F1 treat *fake* as the positive class. Confusion cells TN/FP/FN/TP: validation 199/40/18/214, internal 251/41/31/252, cross-generator 240/41/103/184.) Per generator on the internal test: real accepted as REAL 85.9 %, inpainting detected 91.5 %, insight 86.6 %; **unseen text2img detected 64.1 %**. With the ±0.15 INCONCLUSIVE band the model answers 96.0 % of internal-test faces at 88.2 % accuracy and 92.3 % of cross-generator faces at 75.6 %.
**Comparison (same data):** the original prototype checkpoint scored accuracy 0.503 / 0.511 / 0.507 and ROC-AUC 0.427 / 0.443 / 0.499 on validation / internal / cross-generator — chance level. Of the three trained runs, validation ROC-AUC was A 0.9235 (fine-tune original), **B 0.9477 (ImageNet init, selected)**, C 0.9284 (original, no augmentation).
**Not measured:** video-level metrics of any kind; other datasets; other generators.

### 6.1 Plain-language definitions
- **Accuracy** – share of all faces classified correctly. Misleading if classes are imbalanced (ours are balanced).
- **Precision** – of the faces we called FAKE, the share that really were fake. High precision = few false alarms.
- **Recall** (sensitivity) – of the really fake faces, the share we caught. Recall 0.890 means 89 % caught.
- **F1-score** – harmonic mean of precision and recall; one number balancing both.
- **ROC-AUC** – probability that a randomly chosen fake face gets a higher score than a randomly chosen real one (0.5 = guessing, 1.0 = perfect). Threshold-independent; measures ranking quality.
- **False-positive rate (FPR)** – real faces wrongly called FAKE ÷ all real faces.
- **False-negative rate (FNR)** – fake faces wrongly called REAL ÷ all fake faces (= 1 − recall).
- **Classification threshold** – the cut-off on the score: above 0.5 → FAKE. Fixed in code (`DECISION_THRESHOLD`), never tuned on test data.
- **Internal test** – held-out data from the same *kinds* of fakes used in training, but different people.
- **Cross-generator test** – held-out data made by a generator the model never saw (text-to-image), also of unseen people. **Why it matters:** real-world fakes come from new generators; high accuracy on known generators can hide failure on new ones. Our drop from 87.5 % to 74.6 % (FNR 11 % → 36 %) shows exactly that gap, and we report it instead of hiding it.
- **Training / validation / testing** – training fits the weights; validation is used during development to pick epochs, runs and settings; the test sets are touched only to report final numbers and never to make choices.
- **Why GPU and CPU results are comparable** – the same weights and the same mathematical operations; hardware changes only floating-point rounding. We checked: the Kaggle GPU evaluation and our CPU re-evaluation agreed to every printed digit on all three splits.
- **Why 99.2 % was removed** – it was a claim in the old README for "Celeb-DF-v2" that nothing in the repository supports: the dataset was never available, the code that produced the training data is missing, and the original checkpoint scores at chance on independent data. Our improved numbers are on a *different* dataset and are not comparable.
- **Why scores are not calibrated probabilities** – a sigmoid output is not guaranteed to match real-world frequencies (a 0.9 does not mean "90 % likely fake"); we did not apply calibration (e.g. temperature scaling) and the training data is not representative of the real world.

### 6.2 How leakage was prevented and the held-out evaluation done
Images are split by **identity** (the numeric prefix of the file name is the person), so no person appears in two splits; a real photo and its fake counterpart share a "pair" and always stay in the same split; exact duplicates (SHA-256) and near duplicates (64-bit difference hash, Hamming ≤ 4) were detected and **8 cross-split duplicate images were removed**; `data_validation.py` aborts if any violation is found (result: 0 identity violations, 0 pair violations). The cross-generator set has a generator absent from training. The test splits were evaluated by `training.evaluate` after training; **model selection used only validation ROC-AUC**. Honest caveat: test numbers for all three runs were printed for transparency, so the selection rule was fixed in advance rather than hidden; B also happened to be best on test. Confidence intervals are grouped bootstrap (1,000 resamples). A file-size shortcut diagnostic was computed, but dataset-specific shortcuts (e.g., real photos vary in size while fakes are 512×512) cannot be fully excluded.

---
## Part 7. Presentation script (≈ 13 minutes)

Before you start: backend running, browser open at `http://127.0.0.1:8000`, demo folder ready, PDF viewer ready, one tab on `/api/health`, one on `docs/MODEL_EVALUATION_REPORT.md` or the plots. Pre-tested files only.

**1. Introduction (1 min)**
"Good morning. I am Aniket Raj, and my major project is *Verif-AI* — a system that analyses faces in images and videos and tells whether they show patterns of synthetic or manipulated media. Fake faces are now easy to generate and difficult to identify by eye, which creates risks for misinformation and fraud. Verif-AI addresses this with a two-stream deep-learning model, an explainable web application, and an automatic forensic PDF report. I will also be clear about what it can and cannot do."

**2. Architecture (2 min)**
"The system has three parts. The *frontend* is a React and TypeScript application where the user uploads a file. The *backend* is a FastAPI service. It validates the upload, detects faces with MTCNN, preprocesses each face the same way the training data was prepared, and runs the model. The *model* has two EfficientNet-B0 networks: one looks at the normal RGB image; the other looks at a noise residual produced by fixed SRM high-pass filters, where manipulation traces are easier to see. Their features are concatenated and a small classifier outputs one score between 0 and 1 — the fake-score. A score above 0.5 is FAKE, below is REAL, and very close to 0.5 is reported as INCONCLUSIVE. For images, the pipeline is: upload, face detection, crop, preprocessing, model, decision, Grad-CAM evidence, result, PDF. For videos, the same model is applied to about two frames per second, faces are tracked over time, and the scores are aggregated."

**3. Live image demonstration (3 min)**
[Open home page.] "This is the application; it runs entirely on this laptop. Here is the model information — version, threshold 0.5 and the checkpoint hash." [Click Browse files; choose the real image.] "I select an image from my laptop; here is the preview. I click *Analyse image*." [Result appears.] "The verdict is __. The Real and Fake scores are raw model outputs, not calibrated probabilities. The box marks the detected face; the table shows the face-detector confidence. Here is the model input on the left and the Grad-CAM heat-map on the right — it shows regions that influenced the RGB stream, not proof of editing." [Download PDF; open it.] "The report records the verdict, scores, file hash, model hash, evidence images and limitations, so the analysis can be traced." [Repeat quickly with the fake image. If the model gets it wrong, say:] "This is a miss. Our false-negative rate is 11 % on known generators and 36 % on an unseen generator, so errors like this are expected, and that is why I report them." [Upload the `.txt` file.] "Invalid files are rejected immediately."

**4. Live video demonstration (3 min)**
[Select the genuine-face video; click Analyse video.] "Videos are processed as background jobs: the server returns a job id at once and the page polls the status. The system samples about two frames per second across the whole video — at most 64 frames — detects faces in each, links them into tracks of the same person, scores every face, takes the median score per track, and uses the highest track as the video score." [Result.] "Here is the verdict, the frame timeline against the 0.5 line, suspicious time ranges, the tracks, and the highest-scoring frames as evidence. The PDF contains the same." "An important limitation: I have *not* validated video accuracy because no labelled video dataset was available to us. In our tests, videos made from real photographs were wrongly flagged FAKE, so I do not claim video accuracy."

**5. Model performance (2 min)**
"We trained on a Kaggle GPU. We compared three configurations and selected the best by validation ROC-AUC only — an ImageNet-initialised model. The data is the DeepFakeFace sample, split by identity so no person appears in two sets, with duplicates removed. On the internal test set the accuracy is 87.5 %, ROC-AUC 0.941, and the false-negative rate 11 %. On the cross-generator test — an unseen generator and unseen people — accuracy drops to 74.6 %, ROC-AUC 0.849, and the false-negative rate rises to 36 %. I consider that gap the most honest finding of the project. The original prototype was at chance level on the same data, and the 99.2 % figure in its README could not be reproduced, so we removed it. We also re-ran the evaluation on CPU and obtained the same numbers as on the GPU."

**6. Conclusion (1 min)**
"In summary, I converted a research notebook into a tested, documented, containerised application with image and video pipelines, explainable evidence and PDF reports, and I evaluated it honestly. The limitations are: one dataset, weaker performance on unseen generators, unvalidated video aggregation, uncalibrated scores, and a face-only scope. Future work includes training on more datasets and generators, a labelled video benchmark, probability calibration, temporal models for video, and a GPU deployment. Thank you; I am happy to take questions."

---
## Part 8. Viva questions and answers (55)

**Deepfakes and the problem**
1. *What is a deepfake?* Media whose content, usually a face, is synthesised or altered by machine learning, e.g. face-swapping, inpainting, or fully generated faces from text prompts.
2. *Why is detection hard?* Generators keep improving and change over time; detectors can learn dataset-specific shortcuts and fail on new generators — we measured this (74.6 % on an unseen generator versus 87.5 % on seen ones).
3. *What exactly does your system claim?* It reports whether a face shows patterns the model associates with fabricated faces. It does not prove authenticity or manipulation, and covers faces only.

**Face detection and preprocessing**
4. *Which face detector and why?* MTCNN from `facenet-pytorch`: it is what the original training pipeline used, so crops match training; weights ship with the package, so it works offline.
5. *What preprocessing is applied?* Crop with margin 20 to 224×224, a JPEG quality-95 round-trip, ImageNet mean/std normalisation (`preprocess.py`).
6. *Why a JPEG round-trip at inference?* Training crops were stored as JPEG q95; without the same step inference differs from training. We verified bit-identity with the training storage.
7. *What if there are several faces or none?* Up to 10 faces are scored separately and the image score is the highest; no usable face → INCONCLUSIVE (not an error).
8. *Why ignore very small faces?* Faces under 32 px (or MTCNN confidence < 0.90) are ignored; under 96 px a warning is shown, because tiny faces carry little signal.

**CNN, EfficientNet-B0, SRM, two-stream**
9. *What is a CNN?* A network using learned convolution filters that detect local patterns, stacked so deeper layers recognise more complex structures.
10. *Why EfficientNet-B0?* Compound scaling of depth, width and resolution gives good accuracy for few parameters; two B0s total ≈ 12 M parameters, practical on CPU.
11. *What are SRM filters?* High-pass filters from image steganalysis that suppress image content and expose pixel-level noise patterns. Ours are three fixed 3×3 kernels (`SRMLayer`), not trainable.
12. *Why two streams?* RGB captures visible artefacts (texture, blending); the residual stream captures statistical noise traces that differ between camera images and generated ones. Features are fused late by concatenation.
13. *Why late fusion?* Each stream can use a pretrained backbone and learn its own features; fusing the 1280-dim vectors is simple and effective.
14. *What does the final layer output?* One sigmoid value = fake-score; 0 = real label, 1 = fake label.

**Transfer learning, training, losses, optimisers**
15. *What is transfer learning and did you use it?* Starting from weights learned elsewhere. The selected run B started from **ImageNet** weights; run A started from the original prototype checkpoint.
16. *Why did ImageNet-init win?* We observed higher validation ROC-AUC (0.948 vs 0.924). The original checkpoint is at chance on this data, so it may have been a poorer starting point, but we did not run an experiment to prove the cause.
17. *Loss function?* `BCEWithLogitsLoss` (sigmoid + binary cross-entropy in one numerically stable step) with `pos_weight` for class imbalance (1.02 here).
18. *Optimiser and settings?* AdamW, learning rate 1e-4, weight decay 1e-4, batch 32, ReduceLROnPlateau (halve on validation-loss plateau), gradient clipping 5, seed 42, up to 15 epochs.
19. *Why AdamW?* Adam's adaptive steps plus decoupled weight decay, a standard regulariser for fine-tuning.
20. *What augmentation?* Horizontal flip, mild colour jitter, Gaussian blur (p = 0.2), random JPEG quality 60–95 (p = 0.5): to make the model robust to compression and not rely on exact pixels.
21. *Is the model overfitting?* Training loss kept falling (0.58 → 0.04 by epoch 13) while validation loss bottomed near epoch 8 (0.36) and then rose slightly, a sign of overfitting. We used early stopping (patience 4) and saved the best epoch by validation ROC-AUC (epoch 9).
22. *How did you reduce overfitting?* Augmentation, dropout 0.5, weight decay, early stopping, best-checkpoint selection on held-out validation data.

**Data, leakage, evaluation**
23. *What is data leakage?* Information from the test set influencing training, e.g. the same person or the same image in both sets, which inflates scores.
24. *How did you prevent it?* Identity-disjoint and pair-disjoint splits, SHA-256 and perceptual-hash duplicate removal (8 removed), automatic abort on violations; zero violations reported.
25. *Why train/validation/test?* Train to fit, validation to choose, test to report an unbiased final estimate. Using test data to choose would make the result optimistic.
26. *Why did you pick the model on validation only?* To keep the test sets unbiased; the rule (highest validation ROC-AUC) was fixed before looking at test numbers.
27. *Which dataset and why?* DeepFakeFace (openrail): openly licensed, has several generators and clear labels. Celeb-DF, FaceForensics++ and DFDC need signed agreements and were not available.
28. *Is one dataset enough?* No — that is a stated limitation; results do not transfer automatically to other datasets or real-world videos.

**Metrics and thresholds**
29. *Explain precision and recall for your task.* Precision: when we say FAKE, how often are we right (0.860 internal). Recall: how many fakes we catch (0.890 internal; 0.641 cross-generator).
30. *What is ROC-AUC and why report it?* Ranking quality independent of threshold; 0.941 internal means a random fake outranks a random real 94 % of the time.
31. *Why is the threshold 0.5?* A neutral default set in advance. Tuning it on test data would be cheating; tuning it on validation could be considered but we kept it fixed and report ROC-AUC separately for threshold-free quality.
32. *What is the INCONCLUSIVE band?* Scores within ±0.15 of 0.5 are reported as INCONCLUSIVE — a safety rule for uncertain cases, not a new threshold; evaluated separately (coverage 96 %, accuracy on answered 88.2 % internal).
33. *Which error is worse here, FP or FN?* Depends on use: a false positive wrongly accuses a genuine person's media; a false negative lets a fake pass. We report both (FPR 14 %, FNR 11 % internal) and do not choose for the user.
34. *What is cross-generator generalisation?* Performance on fakes from a generator not seen in training; ours is 74.6 % accuracy, ROC-AUC 0.849, FNR 36 %.
35. *Are your scores probabilities?* No — uncalibrated sigmoid outputs; 0.9 does not mean 90 % likely to be fake.

**Video**
36. *How are video frames chosen?* About 2 fps, evenly spaced across the whole video, at most 64 frames, first 900 s (`plan_sampling`) — not frame 0 only.
37. *How do you combine frames?* Faces are tracked (IoU), each track's median score is taken, the video score is the highest track score; tracks need ≥ 3 frames.
38. *Is the video result validated?* No. No labelled video dataset was available, so video accuracy is unknown, and false positives occurred on videos built from real photographs.
39. *Why might real photos in a video give FAKE?* The model reacts to resampling/re-encoding artefacts; per-frame FPR is 14–17 %, and taking the maximum over tracks amplifies noise.

**Explainability and reports**
40. *What is Grad-CAM?* It uses gradients of the fake-score with respect to the last convolutional block's feature maps to build a heat-map of influential regions (`explain.py`, RGB stream's last block).
41. *Does Grad-CAM prove manipulation?* No; it shows influence on the model's decision, only for the RGB stream.
42. *What does the PDF contain and why?* Verdict, scores, file and model hashes, evidence images, tables, timeline, explanation, warnings — traceability and reproducibility (`report.py`).

**Engineering, deployment, security**
43. *Why FastAPI?* Fast, typed (Pydantic), automatic OpenAPI docs at `/docs`, easy file uploads.
44. *Why is video asynchronous?* It takes seconds to minutes; the API returns 202 with an id and the UI polls status; one worker thread processes one video at a time.
45. *How do frontend and backend communicate?* REST/JSON over HTTP; multipart for uploads; the UI polls `/status` and fetches `/analyses/{id}`, evidence images and PDF.
46. *How is the model loaded safely?* `torch.load(weights_only=True)`, strict state-dict loading, checkpoint SHA-256 shown in the UI and reports.
47. *How is user data protected?* Uploads deleted right after analysis, random filenames, extension + content validation, path-traversal checks, results expire after 72 h or can be deleted, nothing is sent to external services.
48. *Why Docker and what was tested?* To run anywhere identically; multi-stage build, non-root user, healthcheck. Built and run successfully in GitHub Codespaces (4 min 21 s, 2.44 GB); not tested on Windows Docker Desktop.
49. *Why train on Kaggle?* A free T4 GPU trained all three configurations in about 32 minutes in total versus hours on CPU; results were downloaded, hash-checked and re-evaluated on this laptop.

**Challenging the claims**
50. *"87.5 % is on your own dataset — why trust it?"* Correct to ask: it is on identity-disjoint held-out data from the same dataset family, with confidence interval 84.8–90.2 %. It is not a claim about other data; the cross-generator 74.6 % shows what happens with a new generator.
51. *"Your earlier README said 99.2 %. Were you lying?"* That claim came with the original prototype and could not be reproduced: the checkpoint scores at chance on independent data. We removed it and replaced it with measured, reproducible numbers.
52. *"Did you tune on the test set?"* No: threshold fixed at 0.5; model chosen by validation ROC-AUC; test numbers only reported. Test numbers for all runs were printed, but the rule was fixed beforehand.
53. *"Why should I believe the GPU/CPU numbers?"* They are in `docs/results/` (Kaggle GPU json and local CPU json) and match to every printed digit; the checkpoint hash is identical.
54. *"Can it detect any AI-generated image?"* No. It analyses faces and was validated on one dataset and three generators only.
55. *"What if I upload a very good deepfake video?"* It might be missed or flagged unpredictably: the model was trained on image fakes, video aggregation is unvalidated, and unseen generators have a 36 % miss rate.

---
## Part 9. Before the presentation — checklist
- [ ] Laptop charged; Python venv and `frontend\dist` built **in advance** (never install in front of the examiner); start command tested from a *new* PowerShell window the day before.
- [ ] `Test-Path backend\models\verifai_v3_celebdf_qa.pth` is True (current default, Model 4 - read [CURRENT_DEFAULT_MODEL.md](CURRENT_DEFAULT_MODEL.md) so you can answer questions about its weaknesses honestly); `Test-Path backend\models\verifai_v2_best.pth` is True (rollback); `/api/health` says `model_loaded: true` and `/api/model` shows the expected checkpoint.
- [ ] Demo folder prepared **and pre-tested** (≥ 2 real images, ≥ 2 fake images including one you know the model gets right, one low-res, one no-face, one `.txt`); a genuine-face video of yourself; a manipulated video only if legitimately available; spare copies on a USB drive.
- [ ] A PDF downloaded beforehand as a backup; screenshots in `docs/screenshots/`; plots in `docs/figures/v2_selected/`.
- [ ] Browser zoom ~110 %; close other heavy programs; one video at a time.
- [ ] You can explain every number in Part 6 and every limitation in Part 7.
- [ ] Printed or open: `docs/MODEL_EVALUATION_REPORT.md` and `docs/PROJECT_COMPLETION_REPORT.md`.
- [ ] Fallback if the live demo fails: show the recorded screenshots and the PDF, and run `python backend/tools/e2e_smoke.py --manifest backend/manifests/dff_faces_manifest.csv` only if the dataset crops exist locally (they do on the development laptop; they are not in a fresh clone).
