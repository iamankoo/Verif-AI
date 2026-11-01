# Image and Video Demonstration Guide

How to feed images and videos from your laptop into Verif-AI, what happens, and what you will see. Start the application first ([QUICK_START.md](QUICK_START.md)) and open **http://127.0.0.1:8000**. All screen names and button texts below are taken from the real frontend source (`frontend/src`).

> **Golden rule:** Never promise an outcome for a particular file. The model is wrong on a measurable share of faces (see [MODEL_EVALUATION_REPORT.md](MODEL_EVALUATION_REPORT.md)). **Pre-test every demo file the day before** and decide what you will say if the model misses. In a pre-check on the development laptop, 2 of 5 held-out *fake* images were classified REAL — consistent with the measured false-negative rates (11 % internal, 36 % unseen generator). That is an example, not a statistic.

## 1. The images and videos do NOT have to be inside the repository
The application never reads a fixed media folder. You choose any file from anywhere on your laptop (Desktop, Downloads, USB drive) with the file picker or by dragging it into the dashed box. The browser sends a copy to the local backend; your original is untouched, the copy is deleted right after analysis.

## 2. Supported files and limits (from `backend/verifai/config.py`, `image_pipeline.py`, `video_pipeline.py`)

| | Image | Video |
|---|---|---|
| Formats | `.jpg .jpeg .png .bmp .webp .tif .tiff` | `.mp4 .avi .mov .mkv .webm .m4v` |
| Max file size | **15 MB** | **200 MB** |
| Other limits | shortest side ≥ 64 px; ≤ 60,000,000 pixels; up to 10 faces analysed per image | analysed length ≤ 900 s (longer videos: only the first 900 s); **at most 64 frames sampled** at ~2 frames per second |
| Face rules | a face is used only if MTCNN confidence ≥ 0.90 and its shorter side ≥ 32 px; faces under 96 px get a low-resolution warning | same, per sampled frame |
| Mode | synchronous: the result appears when the upload returns | **asynchronous background job**, one video at a time (others queue) |
Extension is checked (case-insensitive); the file content is then actually decoded, so a text file renamed `.jpg` is rejected.

## 3. Image workflow (step by step)
1. **Open** http://127.0.0.1:8000. You see the home page: heading *"See beyond the pixels."*, the upload box, **Model information** (name, version, checkpoint hash, threshold) and **Recent analyses** on the right. If the model were not loaded the upload would be disabled.
2. **Select the image**: click **Browse files** (or drag the file onto the box). The Windows file dialog opens; choose the image. Unsupported types or oversized files are refused immediately with a red message (e.g. `Unsupported file type ".txt"…`, `The file is 20.3 MB; the limit for images is 15 MB.`).
3. A **preview card** shows the picture, file name, "image · size" and the buttons **Analyse image** and **Remove**.
4. Click **Analyse image**. A progress panel shows *Uploading <file>* and then *Analysing…*.
5. What the backend does (`POST /api/analyze/image` → `Engine.run_image` → `analyze_image`): saves the upload under a random name → decodes it (EXIF rotation, RGB) → **MTCNN** finds faces → each face is cropped 224×224 with margin 20 → JPEG q95 round-trip (as in training) → ImageNet normalisation → **two-stream EfficientNet-B0** gives a fake-score 0–1 per face → **Grad-CAM** heat-map of the RGB stream → annotated image → result saved → upload deleted. Typically about half a second on CPU.
6. The **result page** opens (URL like `/#/analysis/<id>`). Top card: the **verdict badge** (REAL / FAKE / INCONCLUSIVE), file name, "image · size · analysis id · seconds", a one-sentence summary, and three buttons: **Download forensic report (PDF)**, **New analysis**, **Delete results**.
7. **Where to read the numbers**
   - *Scores*: bars for **Real score** and **Fake score** (0.000–1.000), plus *Distance from the 0.5 decision boundary* (low / moderate / high). These are raw model outputs, **not calibrated probabilities**, and not "how likely the verdict is right".
   - *Detected faces*: your image with a coloured box per face — **green = REAL, red = FAKE, amber = INCONCLUSIVE** — labelled `#1 REAL 0.00` (face number, verdict, fake-score).
   - *Per-face results*: table with Face, Verdict, Fake, Real and **Detector** (face-detector confidence). Under it, for each face: the **model input crop** (left) and the **Grad-CAM heat-map** (right).
   - *Explanation*: the decision rule in words. *Warnings*: e.g. "Face #1 is small (51px); low-resolution faces are less reliable." *Scope and limitations*: fixed statement at the bottom, including the checkpoint hash.
8. **Interpreting the rule**: the image score = the **highest face fake-score**. Fake-score > 0.5 → FAKE, otherwise REAL; but if the score is within ±0.15 of 0.5 (0.35–0.65) the verdict is **INCONCLUSIVE** (a reporting rule, not a changed threshold). No usable face → INCONCLUSIVE ("could not analyse this image"), which is *not* a statement that the image is real or fake.
9. **Heat-map meaning**: warm colours mark regions that contributed most to the RGB stream's fake-score. It is *not proof* that those pixels were edited and it does not explain the noise-residual (SRM) stream. Say exactly that.
10. **PDF report**: click **Download forensic report (PDF)** → the browser downloads `verifai-report-<8 characters of the analysis id>.pdf` (usually to *Downloads*). Open it to show: verdict banner and scores; *Analysis details* (id, file name/size, **file SHA-256**, resolution, faces analysed, **model checkpoint + hash**, aggregation note); detected-faces image; per-face table; Grad-CAM evidence; *Explanation*; *Warnings*. The PDF is generated on request (`GET /api/analyses/{id}/report.pdf`) from the saved result.
11. **Stored files**: evidence images and the result JSON live in `backend\storage\analyses\<id>\` (images in `artifacts\`) for 72 hours, until you click **Delete results**. Uploaded originals are not kept. `backend\storage` is git-ignored.

## 4. Recommended image test cases
Expected behaviour is what the code is designed to do; **individual verdicts are not predicted**.

| # | Case | How to get it | What the application does | What to explain |
|---|---|---|---|---|
| 1 | Genuine human face | a photo you own, or a *real* held-out image (§6) | one face box; verdict is usually REAL but the measured false-positive rate is 14–17 %, so FAKE/INCONCLUSIVE can happen | scores are uncalibrated; REAL ≠ proof of authenticity |
| 2 | AI-generated / manipulated face | a *fake* held-out image (§6) | usually FAKE for generators seen in training; the unseen generator (text2img) is missed ~36 % of the time | internal vs cross-generator results |
| 3 | Low-resolution image | shrink one image (§6, `make_variants.py`) | if the face is 32–95 px: analysed with a **warning** ("Face #1 is small (51px)…"); < 32 px face: ignored → INCONCLUSIVE; image side < 64 px: rejected (422) | detector rules in `config.py`; reliability drops |
| 4 | Compressed image | save a JPEG at low quality (§6) | analysed normally; heavy compression may change the score. Training used JPEG-quality augmentation 60–95 and a q95 round-trip, so very low qualities are outside what was tested | robustness is not guaranteed |
| 5 | Image without a face | landscape, object, or the gradient made in §6 | verdict **INCONCLUSIVE**, `faces_detected = 0`, explanation "No usable face was detected…"; no error | face-only scope; abstention is a valid outcome |
| 6 | Invalid file type | `notes.txt`, or a PDF | browser refuses immediately (`Unsupported file type ".txt"…`); if sent to the API directly: HTTP 415 `Unsupported image type '.txt'. Allowed: …` | validation on both sides |
| 7 | Text renamed `.jpg` (optional) | rename a text file | HTTP 422 `File is not a readable image (UnidentifiedImageError).` shown as a red banner | content is decoded, not just the extension |

## 5. Video workflow (step by step)
1. Same home page, same upload box. Select a video (`.mp4 .avi .mov .mkv .webm .m4v`, ≤ 200 MB). The preview card shows a small **video player** (muted, with controls) and the buttons **Analyse video** / **Remove**.
2. Click **Analyse video**. The file is uploaded (progress bar), the server answers **HTTP 202** with a job id, and the page jumps to the result URL (`/#/analysis/<id>`), which shows a progress panel *"Analysing <file>…"* with the **stage** (queued → analysing → done) and a progress value. The page asks the server for status **every second** (`GET /api/analyses/{id}/status`) until it is *completed* (then it loads the result) or *failed* (red banner with the reason).
3. What the backend does (`analyze_video`): opens the video (error if not decodable) → `plan_sampling`: about **2 frames per second, evenly spread across the whole video, at most 64 frames** (this is *not* frame 0 only) → MTCNN on each sampled frame → faces are linked into **tracks** across frames by overlap (IoU tracker) → each face crop is scored by the same model → **per-track score = median** of that track's frame scores (tracks need ≥ 3 face frames) → **video score = the highest track score** (if no track is long enough, all observations are pooled) → verdict by the same rule: > 0.5 FAKE, ≤ 0.5 REAL, within ±0.15 INCONCLUSIVE; no usable faces → INCONCLUSIVE. Grad-CAM evidence is produced for the highest-scoring frames.
4. **Where results appear** on the result page: verdict badge; *Scores* and *Explanation*; **Video summary** (frames sampled, frames with a face, face tracks, corrupt frames skipped, duration/fps); **Frame timeline** (score per sampled frame against the 0.5 line, with "x % of frames above 0.5, longest consecutive run, score std, boundary crossings"); **Suspicious timestamps** (consecutive frames above 0.5 grouped into time ranges, or "No analysed frame scored above the 0.5 decision boundary."); **Face tracks** (frames, time range, median score per track); **Highest-scoring frames** (evidence thumbnails with time, frame number and score).
5. **PDF**: same **Download forensic report (PDF)** button. The video PDF adds *Video analysis* (duration, frames sampled, tracks, aggregation "median (not validated on labelled videos)"), the frame-level timeline chart, suspicious timestamps, tracks and highest-scoring-frame evidence.
6. Timing: about 7 s for a 24-frame clip on the development laptop (CPU); a full 64-frame video takes proportionally longer. Do not start a second video while one is running.

## 6. Preparing demonstration media (outside the repository)

**Suggested personal folder (create it anywhere, *not* inside the repository; the application does not need it):**
```
VerifAI-Demo\
  images\  real\   fake\   edge-cases\
  videos\  real\   fake\   edge-cases\
  reports\             <- save your downloaded PDFs here
```
**Legal / privacy rules:** use only media you own, recorded with consent, or that is licensed for this use. Do not upload private or sensitive images of other people. The held-out dataset images below are public research data (DeepFakeFace, openrail licence; real photos from IMDB-WIKI) — keep them on your laptop for the demo and do not republish them.

### 6.1 Held-out test images with known ground truth (best for an examiner)
The project's evaluation split contains images the model **never trained on**, with known labels. On the development laptop they are already at `backend\data\raw\dff\` (`test_internal\real|fake`, `test_crossgen\real|fake`; this folder is git-ignored, so a fresh clone does **not** have it). File names tell the truth: `wiki__…` = real photo, `inpainting__…` / `insight__…` / `text2img__…` = fake (`text2img` is the **unseen-generator** set). Copy a few into your demo folder:
```powershell
$Repo = "C:\path\to\Verif-AI"          # <- change
$Demo = "$HOME\VerifAI-Demo"           # your personal folder, outside the repo
New-Item -ItemType Directory -Force -Path "$Demo\images\real","$Demo\images\fake","$Demo\images\edge-cases","$Demo\videos\real","$Demo\videos\fake","$Demo\videos\edge-cases","$Demo\reports" | Out-Null
$Raw = "$Repo\backend\data\raw\dff"
Get-ChildItem "$Raw\test_internal\real" -Filter *.jpg | Select-Object -First 5 | Copy-Item -Destination "$Demo\images\real"
Get-ChildItem "$Raw\test_internal\fake" -Filter *.jpg | Select-Object -First 5 | Copy-Item -Destination "$Demo\images\fake"
Get-ChildItem "$Raw\test_crossgen\fake" -Filter *.jpg | Select-Object -First 3 | Copy-Item -Destination "$Demo\images\fake"
```
(This exact pattern was run and produced the expected files.) **On a fresh laptop** the raw sample can be re-created with (internet needed, about 200 MB, run from `backend` with the venv active) `python -m training.fetch_deepfakeface --out data/raw/dff --seed 42` — it samples the same seeded, identity-disjoint images from the public dataset; **this re-download was not repeated in this session**, so test it before relying on it. Simplest for a demo: copy your `VerifAI-Demo` folder to a USB drive.

Note: in this dataset the real photos have varied sizes while the fakes are 512×512. The model only sees 224×224 face crops, but this is one reason the dataset-bias limitation in the evaluation report matters.

### 6.2 Edge-case files
```powershell
$Py = "$Repo\backend\.venv\Scripts\python.exe"
Set-Content -Path "$Demo\images\edge-cases\not-an-image.txt" -Value "this is not an image"
Set-Content -Path "$Demo\videos\edge-cases\not-a-video.mp4" -Value "not a video"
@'
import sys
from PIL import Image
src, low, comp = sys.argv[1:4]
im = Image.open(src).convert("RGB")
w, h = im.size
im.resize((max(64, w // 4), max(64, h // 4)), Image.LANCZOS).save(low, quality=95)   # low-resolution copy
im.save(comp, quality=15)                                                            # heavily compressed copy
'@ | Set-Content -Encoding utf8 "$Demo\make_variants.py"
$one = (Get-ChildItem "$Demo\images\fake" -Filter *.jpg | Select-Object -First 1).FullName
& $Py "$Demo\make_variants.py" $one "$Demo\images\edge-cases\lowres.jpg" "$Demo\images\edge-cases\compressed.jpg"
& $Py -c "import numpy as np; from PIL import Image; a=np.tile(np.linspace(0,255,400,dtype=np.uint8),(300,1)); Image.fromarray(np.dstack([a,a[::-1],a])).save(r'$Demo\images\edge-cases\no-face.png')"
```
Result: `lowres.jpg` (a 512 px image shrunk to 128 px, face ≈ 51 px → warning), `compressed.jpg` (JPEG quality 15), `no-face.png` (a colour gradient → INCONCLUSIVE), and the two fake files that must be rejected. The browser itself blocks `.txt`; `not-a-video.mp4` passes the extension check and then fails during analysis with `The file could not be opened as a video (unsupported or corrupt container/codec).` — useful to show the error path.

### 6.3 Videos
There is **no sample video in the repository** and the project has **no labelled video dataset**, so you must supply your own:
- **Genuine face video:** record 10–30 s of yourself (or a consenting friend) with your phone, face clearly visible, good light, saved as MP4. This is the safest "real" video; pre-test it.
- **Manipulated / deepfake video:** only use a clip you are entitled to use (e.g. a face-swap you made yourself with a tool whose licence allows it, with the subjects' consent). Public deepfake benchmarks (Celeb-DF, FaceForensics++, DFDC) require signed agreements and were **not** used. Be honest that the model was trained on *image* fakes (inpainting, face-swap, text-to-image), not on video deepfake methods, so a video made by another method may not be detected.
- **Video made from photographs (functional demo only):** a slideshow of the held-out fake photos exercises decoding → sampling → tracking → aggregation → report. Build one (OpenCV from the project's venv):
  ```powershell
  @'
  import sys, glob, cv2
  src, out = sys.argv[1], sys.argv[2]
  files = sorted(glob.glob(src + "/*.jpg") + glob.glob(src + "/*.png"))[:8]
  assert files, "no images found"
  W = H = 512
  vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
  for f in files:
      im = cv2.resize(cv2.imread(f), (W, H))
      for _ in range(15):          # 1.5 s per photo
          vw.write(im)
  vw.release()
  print("wrote", out, len(files), "photos")
  '@ | Set-Content -Encoding utf8 "$Demo\make_slideshow.py"
  & $Py "$Demo\make_slideshow.py" "$Demo\images\fake" "$Demo\videos\edge-cases\fake-photo-slideshow.mp4"
  ```
  (Tested: it wrote a ~0.4 MB MP4; the application analysed 16 sampled frames and found 2 tracks.) **Do not present this as a deepfake video** — it is stitched still images.

## 7. Video demonstration order and honest limitations
Recommended order: (1) your genuine face video → (2) a manipulated video if you legitimately have one → (3) the invalid file (error path) → (4) optionally the real-photo slideshow to *discuss the limitation*.

**State these limitations yourself, before the examiner finds them:**
- **Video-level accuracy has not been measured or validated** — no labelled video dataset was available. Never quote an accuracy for video. The aggregation rule (median per track, highest track wins) was chosen by design and is explicitly marked "not validated" in the report.
- **False positives were observed on videos made from real photographs:** a slideshow of four *real* photographs was scored **FAKE (0.896)**, and a zoom clip made by resampling and re-encoding one real photograph was scored **FAKE (0.85)** while the original still image scored REAL. Explanation: the model reacts to resampling/re-encoding artefacts, and taking the *highest* track score lets one noisy track decide; the per-frame false-positive rate is 14–17 %.
- Sampling is ~2 frames/s capped at 64 frames, so short manipulated segments in long videos can be missed (the report warns when frames are more than 5 s apart).
- Results are model-based evidence, not proof; scores are uncalibrated.

## 8. Other things you can show in the UI and API
- **Recent analyses** list (right column): click an entry to reopen its result; remove one with **Delete results** on its result page (calls `DELETE /api/analyses/{id}` → 204 and returns you to the home page, where the entry is gone; its evidence images and result file are deleted from `backend\storage`).
- **Model information** card (home page) — rows: Model, Architecture, Parameters (11,972,954), Device, Decision threshold 0.5, Inconclusive band ±0.15, Checkpoint (file name + first 10 hash characters), Grad-CAM available — and `http://127.0.0.1:8000/api/model` (name `verifai-two-stream-v2_B_imagenetInit`, version 2.0, full SHA-256, limits, retention 72 h).
- **Health check** `http://127.0.0.1:8000/api/health` → `{"status":"ok","model_loaded":true,"device":"cpu",…}`.
- **Interactive API page** `http://127.0.0.1:8000/docs` (FastAPI Swagger UI): you can try `/api/analyze/image` from there too.
