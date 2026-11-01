# User Guide

Verif-AI analyses **faces** in images and videos and reports whether the face evidence looks real or fabricated, with visual evidence and a downloadable PDF report. It is investigative support, **not proof**.

## 1. Quick start
1. Start the backend and open the UI (see the Installation Guide): `http://localhost:5173` (dev) or `http://localhost:8000` (built UI / Docker).
2. Drag an image or video onto the upload area (or click **Browse files**). Supported: JPG, JPEG, PNG, BMP, WEBP, TIFF; MP4, AVI, MOV, MKV, WEBM, M4V (codec permitting). Default limits: images 15 MB, videos 200 MB.
3. Check the preview and click **Analyse image / Analyse video**.
4. Images finish in a few seconds on CPU; videos run as a background job with a progress bar (typically tens of seconds to a few minutes).
5. Read the result, inspect the evidence, and **Download forensic report (PDF)** if needed.

## 2. Reading the result
| Item | Meaning |
|---|---|
| **REAL** | The model found no patterns it associates with fabricated faces. *Not* proof that the media is authentic. |
| **FAKE** | The model found patterns it associates with fabricated faces. *Not* proof of manipulation, and it does not say who made it. |
| **INCONCLUSIVE** | No usable face, too little face evidence in a video, or a score too close to the decision boundary (within ±0.15 of 0.5). Treat as "unknown". |
| **Fake score / Real score** | Raw neural-network outputs (real = 1 − fake). **Uncalibrated**: 0.97 does not mean "97 % likely fake". |
| **Distance from the 0.5 boundary** | A decisiveness hint (low/moderate/high). It is **not** the probability that the verdict is correct. |
| **Detector confidence** | How sure the face detector is that a face exists, not about authenticity. |

### Images
Each detected face is scored independently; the image score is the **highest** face score (one manipulated face flags the image; this can raise false alarms on group photos). The annotated image shows boxes and verdicts. For each face you see the exact model input and a **Grad-CAM** heat-map.

### Videos
Frames are sampled (≈ 2 per second, at most 64), faces are detected and linked across frames into *tracks*, each frame is scored, and the video score is the **maximum over tracks of the median frame score**. The page shows: summary (duration, frames, faces), a **frame timeline** (red dots exceed 0.5), **suspicious time ranges**, tracks, and the **highest-scoring frames** with heat-maps. Many dots flipping across the line means unstable evidence - be cautious. The aggregation method has not been validated on labelled videos (none were available).

### Grad-CAM heat-maps
Warm colours mark regions that increased the **RGB stream's** fake score. They show where the network looked, **not** where an image was edited, and they do not explain the noise-residual stream. Never present a heat-map as proof.

## 3. What the system can and cannot do
Can: detect faces; score each face with the trained model; flag suspicious time ranges; produce evidence images and a report; run fully offline.
Cannot: guarantee any verdict; detect manipulation outside faces; handle unseen generators reliably (see `docs/MODEL_EVALUATION_REPORT.md` for measured cross-generator results); analyse audio; authenticate provenance; work on tiny, heavily compressed or occluded faces; resist deliberate adversarial edits.
The trained model's training data and measured performance are shown on the **Model information** card and in the evaluation report. Check them before relying on a result for a given kind of media.

## 4. Good practice
* Use the original file, not a screenshot or re-encoded copy (compression changes scores).
* Prefer faces at least ~100 px wide; the report warns about small faces.
* Treat REAL, FAKE and INCONCLUSIVE as leads for human review, corroborated by provenance checks.
* Do not use the output as sole evidence in legal, employment or disciplinary decisions.

## 5. Privacy
Uploads are deleted immediately after analysis. Evidence images and results stay on the server for 72 hours by default (configurable) and can be removed with **Delete results**. Nothing is sent to third-party services. Faces are personal data: only analyse media you are entitled to process.

## 6. Troubleshooting
| Message | Meaning / action |
|---|---|
| "The analysis model is not loaded" | The server could not load a checkpoint; see Installation Guide |
| "Unsupported file type" | Convert to one of the supported formats |
| "File is not a readable image / video" | The file is corrupt or uses an unsupported codec; re-encode (H.264 MP4) |
| "No face detected" | INCONCLUSIVE by design; try a clearer, larger, front-facing frame |
| "Cannot reach the Verif-AI server" | Backend not running or wrong address |
