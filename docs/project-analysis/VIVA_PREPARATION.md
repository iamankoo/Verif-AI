# VIVA PREPARATION — Verif-AI / Two-Stream Deepfake Analyzer

110 questions with answers grounded in the **actual implementation**. Golden rule for the viva: *state what the code does, state what was measured, and volunteer the limitations before the examiner finds them.* Where an answer relies on outside literature it is marked *(general knowledge)*; verify before citing.

Quick numbers to memorise: EfficientNet-B0 ×2 · 3 SRM-style 3×3 kernels · 1280+1280→2560→512→1 · BCELoss, Adam 1e-4, batch 4, 10 epochs · 15,981 train / 3,996 val frames · val 98.70 % · test sample 1,000 frames (500/500) 99.20 % · confusion [[496,4],[4,496]] · PR-AUC 0.9995 · 11,972,954 stored floats · frame-level, first-frame inference.

---
## A. Problem and scope (Q1–Q12)

**Q1. What problem does your project solve?**
It classifies a face crop from a video frame as real or deepfake (face-swap, Celeb-DF style) using two complementary views: RGB appearance and a high-pass noise residual.

**Q2. Is it an AI-generated-image detector?**
No. It was trained only on Celeb-DF face-swap video frames. It has never seen StyleGAN/diffusion images, so I can't claim it detects them.

**Q3. Is it a video deepfake detector?**
Partially. The model scores single frames. Training sampled ~10 frames per video, but the inference function reads only frame 0; there is no aggregation or temporal modelling. So it is a frame classifier with a first-frame demo, not a video-level detector.

**Q4. Difference between deepfake, face manipulation, AI-generated face, and real?**
Deepfake (face swap): identity replaced in real footage. Face manipulation: umbrella including reenactment and attribute edits. AI-generated face: synthesised from noise/text, no source photo. Real: unmodified capture. Mine targets the first category only.

**Q5. Why does detecting synthetic humans matter?**
Fraud, impersonation, non-consensual content, and disinformation. Detection supports triage and moderation, but gives no guarantee.

**Q6. What is your research contribution?**
Honestly: a working, compact two-stream baseline. The two-stream idea is established (Zhou et al. 2017 *(general knowledge)*). My contribution is the implementation and evaluation pipeline; novelty would require ablations and cross-dataset results I have not run.

**Q7. What does the README claim that your code doesn't do?**
B4 (code: B0), 30 kernels (code: 3), TensorFlow (code: PyTorch), softmax (code: sigmoid), per-video prediction (code: per-frame), separate notebooks (code: one), face alignment (code: detect+crop only).

**Q8. Why do those discrepancies exist?**
I can't reconstruct the history from the repo (no git log), but the README evidently describes an intended or earlier design rather than the final notebook. The checkpoints and code are the ground truth, so I would fix the README rather than defend it.

**Q9. Which deepfake types does the training data contain?**
Celeb-DF fakes: face swaps of 59 celebrities *(general knowledge from the dataset paper)*. One generator family, one compression regime.

**Q10. What would you say is out of scope?**
Audio deepfakes, reenactment, full-image synthesis, localisation, real-time inference, deployment.

**Q11. Who is the intended user?**
Currently a researcher/student running a notebook. There is no end-user interface.

**Q12. How would you define success for this project?**
A video-level AUC on a video- and identity-disjoint test set, plus cross-dataset results. Currently only frame accuracy on self-split data exists.

---
## B. Architecture (Q13–Q30)

**Q13. Describe the architecture in one breath.**
Face crop → (a) EfficientNet-B0 on RGB; (b) fixed SRM-style filter → EfficientNet-B0; global average pool each to 1280; concatenate (2560) → Linear 512 → ReLU → Dropout 0.5 → Linear 1 → Sigmoid.

**Q14. Why two streams?**
Deepfake generators can leave subtle high-frequency traces not obvious in RGB; a residual stream isolates them while the RGB stream captures semantic/visual artefacts.

**Q15. What is the SRM layer exactly?**
A frozen `conv2d` with weight shape (3,3,3,3): three 3×3 kernels, each repeated across the 3 input channels, stride 1, padding 1, divided by 4. All kernels sum to zero → high-pass.

**Q16. Show the kernels.**
K1 = ¼[[-1,2,-1],[2,-4,2],[-1,2,-1]], K2 = ¼[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], K3 = ¼[[0,-1,0],[-1,4,-1],[0,-1,0]].

**Q17. Are these the real SRM filters?**
SRM (Fridrich & Kodovsky) has 30 basic filters, used in 3 commonly-used 5×5 variants. K1 resembles the 3×3 "square" member; K2/K3 are Laplacians. So "SRM-style" is accurate, "SRM bank" is not.

**Q18. Since each kernel is repeated over the three channels, what does that imply?**
Output k = K_k ∗ (R+G+B): the colour channels are summed before filtering, so chroma-specific residuals and inter-channel noise correlations are lost.

**Q19. The SRM layer is applied to ImageNet-normalised data. Is that a problem?**
It's a deviation from the steganalysis convention (raw pixel values, truncation). Normalisation is affine per channel so zero-sum filters ignore the mean shift, but the per-channel std scaling slightly weights channels differently. Not ablated.

**Q20. Why EfficientNet-B0?**
Small (≈4 M backbone params), fast, strong ImageNet transfer. The honest answer: no experiment compared backbones; B0 was practical for a batch-4 GPU budget.

**Q21. How do you know it's B0 and not B4?**
`from_pretrained('efficientnet-b0')`; feature width 1280 (B4 is 1792); `fc.0.weight` is 512×2560; and both streams' stem conv is 32 channels — verified in the checkpoint.

**Q22. Why ImageNet weights for the noise stream?**
They give good initialisation for filters even if input statistics differ; the stream is fully fine-tuned (stem weights diverged from the RGB stream: mean |Δ| 0.0235).

**Q23. Are the streams weight-shared?**
No. Two independent EfficientNet-B0 instances (5,330,613 values each).

**Q24. How is fusion done and why?**
Late concatenation of pooled features — simplest, avoids resolution/alignment issues. Alternatives: attention fusion, bilinear pooling, cross-stream gating.

**Q25. What does `extract_features` return and why use it instead of `forward`?**
The 1280×7×7 feature map before the ImageNet classifier; `forward` would return 1000-class logits. (The unused `_fc` heads still occupy 2.56 M stored values.)

**Q26. Why global average pooling?**
Parameter-free, translation-invariant summary per channel; reduces 7×7 grid to a vector.

**Q27. Why Dropout 0.5 and a 512 hidden layer?**
Regularise a 2560-d concatenated vector against overfitting; sizes were not tuned — unjustified by experiments.

**Q28. Why sigmoid output and not softmax with two logits?**
Binary task; one probability. (README's "softmax" is wrong.)

**Q29. Total parameters?**
11,972,954 stored floating values: 2×5,330,613 + 1,311,745 + 81 (SRM). About 9.4 M are effective after discarding unused `_fc` heads.

**Q30. Is there any temporal component?**
No. No LSTM/3D conv/attention across frames; no temporal parameters exist in the checkpoint.

---
## C. Training (Q31–Q50)

**Q31. Loss function?**
`nn.BCELoss`: −[y log p + (1−y) log(1−p)] on sigmoid outputs.

**Q32. Why not `BCEWithLogitsLoss`?**
It would be more numerically stable (log-sum-exp). My code applies sigmoid inside the model, and logged mini-batch loss hit exactly 0 at epoch 9, showing saturation. I'd change it.

**Q33. Optimiser and hyperparameters?**
Adam, lr 1e-4, default betas, no weight decay, no scheduler, batch 4, 10 epochs.

**Q34. Why batch size 4?**
Two B0 backbones at 224² on a limited GPU. Downside: noisy BatchNorm statistics and gradient noise.

**Q35. Was BatchNorm in train mode? Consequences?**
Yes (`model.train()`), so BN running stats updated with batch-4 statistics.

**Q36. Did you freeze anything?**
Only the SRM kernel (`requires_grad=False`). Backbones fully fine-tuned.

**Q37. Is the SRM kernel in the optimiser?**
It's passed via `model.parameters()` but has no gradient, so Adam skips it. Verified: identical in all 10 checkpoints.

**Q38. Training curve?**
Loss 0.2883 → 0.0214; accuracy 0.8707 → 0.9921 over 10 epochs. Still slowly improving at epoch 10.

**Q39. Is the 0.9921 a validation accuracy?**
No — it's the running **training** accuracy with dropout on.

**Q40. Why does `plot.py` show loss ≈ 0 at epoch 10?**
It's a bug: I typed the last mini-batch loss (0.00012) instead of the epoch mean (0.0214). The data are also hard-coded, not logged.

**Q41. Why did epoch 10 take 21 minutes vs ~10 for others?**
Not recorded; probably resource contention. Unverified.

**Q42. Did you use early stopping or select the best epoch?**
No. All epochs were saved; epoch 10 was used. Selection on validation would be better.

**Q43. Did you use data augmentation?**
No. `albumentations` is in requirements but never imported. This likely hurts robustness.

**Q44. How were labels assigned?**
`enumerate(['Celeb-real','Celeb-fake'])` → real=0, fake=1.

**Q45. How was the split done?**
`random_split` 80/20 over frames → 15,981 / 3,996.

**Q46. What's wrong with that split?**
Frame-level: near-identical frames from one video land in both sets (leakage) → optimistic val accuracy. Should split by video and by identity.

**Q47. Reproducibility?**
`random` and `torch` seeds set (42). NumPy/CUDA seeds not set; hard-coded paths; non-linear cell execution; so only approximately reproducible.

**Q48. Is the dataset balanced?**
The 1,000-frame test sample is (500/500). The training class ratio is not in the repo — unverified.

**Q49. How long did training take?**
≈2 h 05 min total on a CUDA GPU of unrecorded model.

**Q50. Did you check overfitting?**
Only post hoc: train 99.21 %, val 98.70 %. No per-epoch validation curve, so I can't plot the generalisation gap over time.

---
## D. Data and preprocessing (Q51–Q65)

**Q51. Which dataset?**
Celeb-DF (folders Celeb-real/Celeb-fake; README says v2). Official size 590 real + 5,639 fake *(README; matches the paper)*. Only part is used.

**Q52. How many images?**
19,977 face crops in the training pool; plus 1,000 in the held-out sample.

**Q53. How are frames sampled?**
`interval = max(1, total_frames // 10)`; keep frames where `count % interval == 0` while iterating sequentially; targets ~10 per video.

**Q54. Why not random frames?**
Uniform spacing covers the clip and is deterministic.

**Q55. Face detector?**
MTCNN (facenet-pytorch): image_size 224, margin 20, largest face only (`keep_all=False`, `select_largest=True`), thresholds [0.6,0.7,0.7].

**Q56. Do you align faces?**
No. Detection + margin crop only; the banner's "Face Align" is inaccurate.

**Q57. What if no face is detected?**
MTCNN returns None and nothing is saved — but my code still increments `saved`, so some videos yield fewer than 10 crops. That is a bug (B2).

**Q58. What if there are multiple faces?**
Only the largest face is kept.

**Q59. Why JPEG for crops?**
Convenience (MTCNN `save_path`). It adds a lossy step that the inference path doesn't replicate — a train/inference mismatch, especially for a noise stream.

**Q60. Normalisation?**
ImageNet mean/std after `ToTensor` ([0,1]).

**Q61. Why 224×224?**
Standard EfficientNet-B0 input; MTCNN outputs that size directly.

**Q62. What are the data-leakage risks?**
Same-video frames across split; test sample drawn from the same pool without excluding training videos; identity overlap; source/fake pairing.

**Q63. Is the 99.2 % test set independent?**
Not demonstrably. It was randomly drawn (seed 42) from the same raw video folder as training data; there's no exclusion list. I can't prove independence.

**Q64. Why use the YouTube-real subset?**
I don't — only Celeb-real and Celeb-fake.

**Q65. Could the model be learning compression artefacts?**
Possible. Real and fake Celeb-DF videos have different codec histories *(general knowledge)*; a noise-residual stream can latch onto them. Test via compression-robustness and cross-dataset experiments (not done).

---
## E. Evaluation (Q66–Q85)

**Q66. Report your results.**
Val 98.70 % (3,996 frames). Test sample 99.20 % (1,000 frames): confusion [[496,4],[4,496]], precision/recall/F1 0.99 each, PR-AUC 0.9995.

**Q67. FPR and FNR?**
4/500 = 0.8 % each (derived from the confusion matrix).

**Q68. ROC-AUC?**
Not computed. I have PR-AUC only.

**Q69. Which class is "positive" for PR-AUC?**
Fake (label 1), since `precision_recall_curve(labels, probs)` uses default pos_label=1.

**Q70. Is PR-AUC = 0.9995 meaningful?**
On a balanced 1,000-frame set with 8 errors, it mostly restates accuracy. It's also computed by trapezoid integration, which can be slightly optimistic.

**Q71. Confidence interval on 99.2 %?**
Wilson 95 % ≈ [98.4, 99.6] assuming independence. Frames are clustered by video, so the true interval is wider.

**Q72. Why is the test (99.20) above val (98.70)?**
Within noise, but also consistent with the test frames not being harder than val — i.e., possible overlap with training videos.

**Q73. Is 99.2 % a CelebDF-v2 benchmark result?**
No. It's frame accuracy on a self-sampled set, not the official test list or video-level AUC.

**Q74. Is "73–90 % AUC for other models" a fair comparison?**
No. Different metric (AUC vs accuracy), different protocol, and the claim is unsourced.

**Q75. Video-level performance?**
Unknown; not computed.

**Q76. Cross-dataset performance?**
Unknown; never tested. Literature suggests significant drops *(general knowledge)*.

**Q77. Robustness to compression/blur/resize?**
Unknown; never tested.

**Q78. Inference speed?**
From tqdm: ≈33–45 images/s at batch 4 on an unspecified CUDA GPU, including data loading. CPU and memory not measured.

**Q79. Calibration?**
Not evaluated. Sigmoid outputs saturate near 0/1 (e.g., Prob 0.0000 in the README image), so they're likely over-confident.

**Q80. How is "confidence" shown to the user?**
It's p(fake); even for REAL verdicts the printed "Prob" is p(fake).

**Q81. Threshold choice?**
Fixed 0.5. No operating-point analysis; for deployment I'd pick a threshold at a target FPR.

**Q82. What do the 4 FP and 4 FN tell you?**
Nothing without inspecting images. I haven't analysed error cases — a gap.

**Q83. Did you evaluate with the model in eval mode?**
Yes (`model.eval()`, `torch.no_grad()`), so dropout is off and BN uses running stats.

**Q84. Why is the final "Validation Accuracy" 98.70 % while training was 99.21 %?**
Mostly dropout on/off and generalisation gap; modest 0.5 pt difference, but the val split is itself leaky.

**Q85. How would you evaluate properly?**
Video+identity-disjoint split, official test list, video-level AUC/ROC, bootstrap CIs by video, cross-dataset, robustness sweep, multiple seeds.

---
## F. Inference and explainability (Q86–Q95)

**Q86. Walk through inference.**
Open video → read first frame → BGR→RGB → MTCNN (`post_process=False`, 0–255 tensor) → /255 → resize → ImageNet normalise → model → p; label by >0.5 → Grad-CAM → plot.

**Q87. Why `post_process=False` at inference but `True` at extraction?**
At extraction, the saved JPEG is the raw crop anyway (saving happens before post-process); at inference the code wants 0–255 values to rescale itself. Net effect: same pixel scale, but no JPEG at inference.

**Q88. What bug blocks inference on CPU?**
`torch.load(model_path)` without `map_location` → "Attempting to deserialize object on a CUDA device…" (recorded in the notebook). Fix with `map_location` and `weights_only=True`.

**Q89. What is the "logit vs probability" check for?**
Defensive debugging code. It's dead code because the model already outputs a sigmoid.

**Q90. How does Grad-CAM work here?**
Gradients of the fake-score w.r.t. the last block of the RGB stream's activations are averaged per channel to weight the feature maps → ReLU → upsampled.

**Q91. Limitations of your Grad-CAM?**
Only the RGB stream; 7×7 resolution; always explains the "fake" score even for REAL predictions; no quantitative validation.

**Q92. Can you explain the SRM stream?**
Not currently; I'd add CAM on `noise_net` and perturbation tests.

**Q93. What does `image6.3.png` prove?**
Nothing statistically — one example (REAL, Prob 0.0000). It shows the explainability output format. Its heat-map highlights border/background more than the face centre, which merits investigation (possible context reliance), but one image proves nothing.

**Q94. Why does inference read only frame 0?**
It's a prototype for visual demonstration; I did not build video aggregation.

**Q95. How would you extend to videos?**
Sample N frames, detect faces, batch-infer, aggregate logits (mean/trimmed-mean/top-k), report number of valid frames, optionally add a temporal head.

---
## G. Engineering and security (Q96–Q104)

**Q96. Is there a frontend/API?**
No. The `backend/` folder has no server code.

**Q97. What's `pipeline.py`?**
An empty file — a placeholder, not implemented.

**Q98. What dependencies are wrong?**
`pytorch-gradcam` should be `grad-cam` for the `pytorch_grad_cam` import; `seaborn` is missing; `torchaudio`, `albumentations`, `pandas` are unused; nothing is version-pinned.

**Q99. Any tests?**
No automated tests; only evaluation cells.

**Q100. Security risks if deployed?**
Pickle deserialisation via `torch.load`; hostile media parsing; no upload limits; adversarial inputs; privacy of face data; model-extraction by query.

**Q101. Adversarial robustness?**
Not tested. CNN detectors are vulnerable to small perturbations and to re-encoding *(general knowledge)*.

**Q102. Why are checkpoints in the repo?**
For convenience; 460 MB total, `.gitignore` doesn't exclude `*.pth`. Better: Git LFS or release assets.

**Q103. What's in `backend/venv`?**
A stub conda environment directory (no usable interpreter); should be git-ignored/removed.

**Q104. Hardware needs?**
Training used CUDA (model unrecorded). Inference works on CPU after the `map_location` fix; speed unmeasured.

---
## H. Theory and deeper ML (Q105–Q110)

**Q105. Why do high-pass filters help detect manipulation?**
Natural images have sensor/processing noise with characteristic statistics; blending, resampling and generator up-sampling alter local pixel correlations. A zero-sum kernel cancels smooth content and leaves these local statistics.

**Q106. Why might SRM features be brittle?**
Compression, resizing and denoising wash out the fine residuals that the filters isolate; so a detector reliant on them degrades on social-media-processed video *(general knowledge)*.

**Q107. How does compound scaling work in EfficientNet?**
Depth d=α^φ, width w=β^φ, resolution r=γ^φ with α·β²·γ²≈2; B0 is the base (Tan & Le 2019).

**Q108. What is shortcut learning here?**
The model may exploit dataset-specific cues (codec, background, identity, resolution) that correlate with the label but not with manipulation — hence the need for cross-dataset tests.

**Q109. Frame-level vs video-level metrics: why different?**
Frame-level treats each crop as independent; video-level aggregates evidence and is the operational unit. Aggregation usually boosts AUC but also lets correlated frame errors persist.

**Q110. If you had one week, what would you do first?**
(1) Rebuild the split by video/identity and re-evaluate with video-level ROC-AUC; (2) RGB-only vs SRM-only vs fused ablation; (3) compression-robustness curves; (4) fix inference (CPU-safe, multi-frame); (5) correct the README to match the code.

---
### Cheat-sheet of weaknesses to admit confidently
1. README ≠ code (B4/30 kernels/TF/softmax/per-video).
2. 99.2 % = self-sampled frame accuracy, leakage-prone, not the benchmark protocol.
3. No ablation proving the SRM stream helps.
4. No video aggregation; first-frame inference; last recorded run errors on CPU.
5. No cross-dataset, robustness, ROC-AUC, calibration.
6. No frontend/API/DB/tests/deployment.
