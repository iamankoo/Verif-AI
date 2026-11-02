# Verif-AI - Research references and novelty

Student: Aniket Raj (Roll No. 230103030), B.Tech CSE (AI & Data Science), IIIT Manipur.
Faculty advisor / project guide: **Dr. Sunita Warjri**, Assistant Professor, Department of Computer Science and Engineering (CSE), Indian Institute of Information Technology Manipur - sunitaw@iiitmanipur.ac.in - +91 7085262822 - department page: https://iiitmanipur.ac.in/pages/academic/academicCSE.php

**How to read this file.** Part A lists every paper that is cited in the final report (36 entries; each one is cited at least once, checked by `report/scripts/audit_citations.py`). Titles, authors, years, venues, pages and DOIs were checked against Crossref (journals/conferences) or arXiv (preprints) with `report/scripts/verify_references.py`. Part B says what was actually built. Part C compares existing work with Verif-AI. Part D gives answers for your advisor's questions. Statements about other papers describe only what their titles/abstracts state; nothing here claims capabilities that those papers do not claim.

**One-sentence honesty rule:** Verif-AI does **not** propose a new deep-learning architecture. EfficientNet, SRM filtering, MTCNN and Grad-CAM are not ours, and the two-stream RGB + SRM design comes from the earlier Forensic-Net prototype by Lakshya Singh.

---

## Part A - Research papers used

Format: **Title** - authors - year - venue - DOI/link. *What it introduced.* **Where used / why.** (Chapter numbers refer to `Verif-AI_Project_Report.pdf`.)

### A1. Generative models and face manipulation (background)
1. **Generative Adversarial Nets** - I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio - 2014 - NeurIPS 27 - https://arxiv.org/abs/1406.2661. *Adversarial training of a generator and a discriminator.* **Ch. 1, 2.1.1:** origin of GAN-generated faces that detectors must handle.
2. **A Style-Based Generator Architecture for Generative Adversarial Networks** - T. Karras, S. Laine, T. Aila - 2019 - CVPR, pp. 4396-4405 - doi:10.1109/CVPR.2019.00453. *StyleGAN and the FFHQ face dataset.* **Ch. 1, 2.1.1, 2.1.4, 6.1:** StyleGAN-generated faces, and the source of FFHQ (real faces in experiments E3/E4).
3. **Analyzing and Improving the Image Quality of StyleGAN** - T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, T. Aila - 2020 - CVPR, pp. 8107-8116 - doi:10.1109/CVPR42600.2020.00813. *StyleGAN2.* **Ch. 2.1.1:** successor generator, cited with StyleGAN as an example of high-resolution face synthesis.
4. **FaceForensics++: Learning to Detect Manipulated Facial Images** - A. Rossler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Niessner - 2019 - ICCV, pp. 1-11 - doi:10.1109/ICCV.2019.00009. *A large benchmark of manipulated face videos (four manipulation methods) and detector baselines.* **Ch. 1, 2.1.1, 2.1.4, 2.2:** benchmark of the field; not used for training here (access agreement).
5. **Denoising Diffusion Probabilistic Models** - J. Ho, A. Jain, P. Abbeel - 2020 - NeurIPS 33 - https://arxiv.org/abs/2006.11239. *Diffusion models for image synthesis.* **Ch. 1, 2.1.1:** diffusion-based generators.
6. **High-Resolution Image Synthesis with Latent Diffusion Models** - R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer - 2022 - CVPR, pp. 10674-10685 - doi:10.1109/CVPR52688.2022.01042. *Latent diffusion (the basis of Stable Diffusion).* **Ch. 1, 2.1.1:** text-to-image and inpainting generators, which produce the DeepFakeFace fakes.

### A2. Surveys and detectors
7. **Media Forensics and DeepFakes: An Overview** - L. Verdoliva - 2020 - IEEE J. Sel. Topics Signal Process. 14(5):910-932 - doi:10.1109/JSTSP.2020.3002101. *Survey of media-forensics and deepfake-detection methods.* **Ch. 1, 2.1:** field overview.
8. **DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection** - R. Tolosana, R. Vera-Rodriguez, J. Fierrez, A. Morales, J. Ortega-Garcia - 2020 - Information Fusion 64:131-148 - doi:10.1016/j.inffus.2020.06.014. *Survey of face-manipulation types and detectors.* **Ch. 1, 2.1.**
9. **MesoNet: a Compact Facial Video Forgery Detection Network** - D. Afchar, V. Nozick, J. Yamagishi, I. Echizen - 2018 - IEEE WIFS, pp. 1-7 - doi:10.1109/WIFS.2018.8630761. *A small CNN for face-video forgery detection.* **Ch. 2.1.1, Table 2.1, 2.2:** early compact detector.
10. **Xception: Deep Learning with Depthwise Separable Convolutions** - F. Chollet - 2017 - CVPR, pp. 1800-1807 - doi:10.1109/CVPR.2017.195. *Depthwise-separable convolutional backbone.* **Ch. 2.1.1, Table 2.1:** strongest baseline in FaceForensics++.
11. **Deep Residual Learning for Image Recognition** - K. He, X. Zhang, S. Ren, J. Sun - 2016 - CVPR, pp. 770-778 - doi:10.1109/CVPR.2016.90. *Residual networks.* **Table 2.1:** common baseline backbone.
12. **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** - M. Tan, Q. V. Le - 2019 - ICML (PMLR 97) - https://arxiv.org/abs/1905.11946. *Compound scaling of depth, width and resolution; the B0 baseline.* **Ch. 2.1.2, 3, 4.1.3:** the backbone of both streams (about 5.3 M parameters, 1280-d output).
13. **Two-Stream Neural Networks for Tampered Face Detection** - P. Zhou, X. Han, V. I. Morariu, L. S. Davis - 2017 - CVPR Workshops, pp. 1831-1839 - doi:10.1109/CVPRW.2017.229. *Combines a face-classification stream with a low-level (steganalysis-feature) stream.* **Ch. 2.1.1, 2.2:** the idea behind a two-stream face detector.
14. **A Deep Learning Approach to Universal Image Manipulation Detection Using a New Convolutional Layer** - B. Bayar, M. C. Stamm - 2016 - ACM IH&MMSec, pp. 5-10 - doi:10.1145/2909827.2930786. *Constrained convolutional layer that learns manipulation features.* **Ch. 2.1.1:** learnable high-pass-like residual layers, versus our fixed kernels.
15. **On the Detection of Digital Face Manipulation** - H. Dang, F. Liu, J. Stehouwer, X. Liu, A. K. Jain - 2020 - CVPR, pp. 5780-5789 - doi:10.1109/CVPR42600.2020.00582. *Attention maps to localise manipulated regions; a face-manipulation dataset.* **Ch. 2.1.1:** attention-based localisation in the literature.
16. **Rich Models for Steganalysis of Digital Images** - J. Fridrich, J. Kodovsky - 2012 - IEEE TIFS 7(3):868-882 - doi:10.1109/TIFS.2012.2190402. *Spatial Rich Model: many high-pass residual filters and co-occurrence features.* **Ch. 2.1.2, 4.1.3.2:** basis of the three fixed SRM-style kernels (the original uses many more filters).
17. **Generalizing Face Forgery Detection with High-frequency Features** - Y. Luo, Y. Zhang, J. Yan, W. Liu - 2021 - CVPR, pp. 16312-16321 - doi:10.1109/CVPR46437.2021.01605. *High-frequency (SRM-filtered) features combined with RGB features for better generalisation.* **Ch. 2.2:** closest related work to the SRM-stream idea; we do not re-implement it.
18. **Face X-Ray for More General Face Forgery Detection** - L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, B. Guo - 2020 - CVPR, pp. 5000-5009 - doi:10.1109/CVPR42600.2020.00505. *Detects the blending boundary, aiming at unseen manipulations.* **Ch. 2.2:** generalisation to unseen manipulation methods.
19. **Deepfake Video Detection Using Recurrent Neural Networks** - D. Guera, E. J. Delp - 2018 - IEEE AVSS, pp. 1-6 - doi:10.1109/AVSS.2018.8639163. *CNN features per frame + recurrent network for temporal inconsistencies.* **Ch. 2.2:** the temporal-model approach to video that Verif-AI does not use.
20. **Two-Branch Recurrent Network for Isolating Deepfakes in Videos** - I. Masi, A. Killekar, R. M. Mascarenhas, S. P. Gurudatt, W. AbdAlmageed - 2020 - ECCV (LNCS) pp. 667-684 - doi:10.1007/978-3-030-58571-6_39. *Two-branch network with a recurrent model over video frames.* **Ch. 2.2:** another video-level approach.

### A3. Generalisation, evaluation pitfalls, explanation
21. **CNN-Generated Images Are Surprisingly Easy to Spot... for Now** - S.-Y. Wang, O. Wang, R. Zhang, A. Owens, A. A. Efros - 2020 - CVPR, pp. 8692-8701 - doi:10.1109/CVPR42600.2020.00872. *A classifier trained on one GAN with augmentation generalises to other CNN generators, with limits.* **Ch. 1, 2.1.5, 2.2:** why internal and cross-generator results must be reported separately.
22. **Towards Universal Fake Image Detectors that Generalize Across Generative Models** - U. Ojha, Y. Li, Y. J. Lee - 2023 - CVPR, pp. 24480-24489 - doi:10.1109/CVPR52729.2023.02345. *Detection in a frozen pretrained feature space for cross-generator generalisation.* **Ch. 1, 2.1.5, 2.2.**
23. **On the Detection of Synthetic Images Generated by Diffusion Models** - R. Corvi, D. Cozzolino, G. Zingarini, G. Poggi, K. Nagano, L. Verdoliva - 2023 - IEEE ICASSP, pp. 1-5 - doi:10.1109/ICASSP49357.2023.10095167. *Detectors across GAN and diffusion generators.* **Ch. 1, 2.1.5, 2.2.**
24. **Shortcut Learning in Deep Neural Networks** - R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, F. A. Wichmann - 2020 - Nature Machine Intelligence 2:665-673 - doi:10.1038/s42256-020-00257-z. *Networks can solve benchmarks through unintended cues.* **Ch. 1, 2.1.5, 6.1.5:** frames the image-sharpness finding.
25. **On Calibration of Modern Neural Networks** - C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger - 2017 - ICML (PMLR 70) - https://arxiv.org/abs/1706.04599. *Modern networks are often over-confident.* **Ch. 1, 2.1.5:** why scores are not shown as probabilities.
26. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization** - R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra - 2017 - ICCV, pp. 618-626 - doi:10.1109/ICCV.2017.74. *Gradient-weighted class-activation heat-maps.* **Ch. 2.1.2, 3, 4.1.3.4, 5:** the explanation module (RGB stream).

### A4. Face detection, datasets
27. **Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks** - K. Zhang, Z. Zhang, Z. Li, Y. Qiao - 2016 - IEEE Signal Process. Lett. 23(10):1499-1503 - doi:10.1109/LSP.2016.2603342. *MTCNN.* **Ch. 3, 5:** face localisation (via the facenet-pytorch implementation).
28. **Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics** - Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu - 2020 - CVPR, pp. 3204-3213 - doi:10.1109/CVPR42600.2020.00327. *590 real and 5,639 face-swap videos.* **Ch. 2.1.4, 6.1:** training/validation/test data for Models 3, 4, E3, E4 (within-dataset evidence only; research-use terms).
29. **The DeepFake Detection Challenge (DFDC) Dataset** - B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, C. Canton Ferrer - 2020 - arXiv:2006.07397 - https://arxiv.org/abs/2006.07397. *About 100k clips.* **Ch. 2.1.4:** listed, not used (access rules).
30. **Robustness and Generalizability of Deepfake Detection: A Study with Diffusion Models** - H. Song, S. Huang, Y. Dong, W.-W. Tu - 2023 - arXiv:2309.02218 - https://arxiv.org/abs/2309.02218. *The DeepFakeFace dataset.* **Ch. 2.1.4, 6.1:** main training and test data (a sampled subset).
31. **Deep Expectation of Real and Apparent Age from a Single Image Without Facial Landmarks** - R. Rothe, R. Timofte, L. Van Gool - 2018 (IJCV 126(2-4):144-157) - doi:10.1007/s11263-016-0940-3. *The IMDB-WIKI face dataset.* **Ch. 2.1.4, 6.1:** source of the real images in DeepFakeFace.
32. **Synthetic Faces High Quality (SFHQ) dataset** - D. Beniaguev - GitHub (CC0) - https://github.com/SelfishGene/SFHQ-dataset. *StyleGAN-generated faces.* **Ch. 2.1.4, 6.1:** fake faces in E3/E4.

### A5. Software and statistics
33. **PyTorch: An Imperative Style, High-Performance Deep Learning Library** - A. Paszke et al. - 2019 - NeurIPS 32 - https://arxiv.org/abs/1912.01703. **Ch. 3:** deep-learning framework.
34. **Decoupled Weight Decay Regularization** - I. Loshchilov, F. Hutter - 2019 - ICLR - https://arxiv.org/abs/1711.05101. *AdamW.* **Ch. 3, 5:** optimiser.
35. **Probable Inference, the Law of Succession, and Statistical Inference** - E. B. Wilson - 1927 - JASA 22(158):209-212 - doi:10.1080/01621459.1927.10502953. *Wilson score interval.* **Ch. 6:** confidence intervals for error rates.
36. **An Introduction to the Bootstrap** - B. Efron, R. J. Tibshirani - 1993 - Chapman & Hall (book). **Ch. 6:** grouped-bootstrap intervals.

(The DeepDetect-2025 collection and the Celeb-DF third-party Kaggle copy are datasets without a paper; they are described in Appendix C and `docs/DATASET_LICENCE_REVIEW.md`.)

---

## Part B - What we built (plain language, technically accurate)

1. **A working forensic application, not only a model.** A React + TypeScript dashboard and a FastAPI backend: upload an image or video, get per-face REAL/FAKE/INCONCLUSIVE results, a Grad-CAM view (Original / Heat-map / Overlay), a video timeline, face tracks, stored evidence frames, a downloadable PDF forensic report, history and deletion.
2. **The two-stream pipeline, re-implemented as a tested package.** RGB EfficientNet-B0 + a fixed three-kernel SRM layer feeding a second EfficientNet-B0, concatenated (2560) -> 512 -> 1 -> sigmoid; 11,972,954 parameters. *The architecture is inherited from the Forensic-Net prototype (Lakshya Singh).* What is ours: the same preprocessing at training and inference (JPEG q95 round-trip), strict checkpoint loading with SHA-256 recording, an INCONCLUSIVE band, and the video logic below.
3. **Video analysis.** About 2 frames per second (at most 64), IoU face tracking, per-track median, video score = maximum over tracks, suspicious time ranges, six evidence frames with heat-maps, honest "not stored" states. (The original prototype analysed only the first frame.)
4. **Evaluation and experiments (the part with most research value).** Identity-disjoint splits, near-duplicate removal, validation-only selection; four models compared with identical code (v2, Model 2, Model 3, Model 4); internal vs. cross-generator vs. Celeb-DF video vs. `modern_test`; stream ablation; the quality-shortcut diagnosis (E1); a per-quality-bin acceptance gate validated with synthetic controls (stage 0); pre-registered experiments E3/E4 with a frozen promotion rule. The result that matters: the production model was at chance on held-out video; the cause was an image-sharpness shortcut; E4 (the latest model) is strong on Celeb-DF video (0.980) and `modern_test` (0.983) but **failed promotion criterion 2** (DeepFakeFace missed-fake rate 0.208 internal, 0.509 cross-generator).
5. **A model registry.** E4 (default for the local examiner demonstration), Model 4 (alternative) and v2 (rollback, and what the live demo serves); every result, dashboard chip and PDF names the model, role and checkpoint hash.
6. **Reproducible workflow and privacy design.** Kaggle GPU notebooks that export only checkpoints and metrics; dataset licence review; uploads deleted after analysis; atomic job storage; memory-aware "lite" mode for a 512 MB host; 167 backend tests, 66 frontend tests, Playwright end-to-end tests.

## Part C - Existing research vs. Verif-AI

| Existing paper / system | Its contribution | What it does | What Verif-AI implements or extends | What is different in our integrated application |
|---|---|---|---|---|
| Zhou et al. 2017 | two-stream tampered-face detector | image stream + low-level stream | same two-stream idea with fixed SRM residuals (via the prototype) | identity-disjoint training/evaluation, quality-bin gate, deployed workflow |
| Fridrich & Kodovsky 2012 | rich residual models (steganalysis) | many high-pass filters | three fixed kernels as the noise stream | examines when this stream carries a quality shortcut |
| Luo et al. 2021; Li et al. 2020 | generalisation via high-frequency features / blending boundary | detectors evaluated for unseen forgeries | not re-implemented | measures generalisation with an unseen-generator split and reports failures |
| Guera & Delp 2018; Masi et al. 2020 | recurrent video detectors | temporal modelling of frame features | not used | per-face scores, IoU tracks, per-track median, timelines and evidence (aggregation unvalidated) |
| Selvaraju et al. 2017 | Grad-CAM | explanation maps | applied to the RGB stream; three viewing modes; stored for evidence frames | states clearly that it is not proof |
| Wang 2020; Ojha 2023; Corvi 2023 | cross-generator generalisation studies | detectors trained on one generator, tested on others | motivates the internal vs. cross-generator reporting | no claim of universal detection |
| FaceForensics++, Celeb-DF, DeepFakeFace, FFHQ/SFHQ | datasets | training/benchmarking | used under their terms, leakage controlled | licence review, near-duplicate removal |
| Forensic-Net prototype (L. Singh) | the two-stream design, notebook, checkpoint | first-frame, single-dataset detection | preserved and re-implemented | application, video pipeline, evaluation tooling, registry, tests, deployment |

**Kinds of contribution (for your explanation):**
* *Research novelty:* none at architecture level. Empirical findings only: chance-level behaviour on held-out video; the image-sharpness shortcut and its carrier (noise stream); the acceptance gate that fails all four models although two pass the usual screen; E4's trade-off.
* *Engineering novelty:* training-matched preprocessing, strict loading + hash recording, INCONCLUSIVE band, atomic storage, privacy-preserving deletion, memory-aware hosting mode, model registry.
* *System-level integration:* one application joining face detection, two-stream inference, Grad-CAM, tracking, timelines, evidence frames, PDF reports, history/deletion, hosted deployment.
* *Experimental contribution:* a reproducible, leakage-controlled, pre-registered evaluation of four models plus E3/E4 with a frozen promotion rule.

## Part D - Questions from the faculty advisor

**1. What research papers did you use as references?** "The report cites 36 sources. The core ones are EfficientNet (Tan and Le), the Spatial Rich Model (Fridrich and Kodovsky), the two-stream tampered-face detector of Zhou et al., MTCNN (Zhang et al.), Grad-CAM (Selvaraju et al.), and the datasets Celeb-DF, DeepFakeFace, FFHQ and SFHQ. For generalisation and evaluation I used Wang et al., Ojha et al., Corvi et al. and Geirhos et al. on shortcut learning."

**2. What did you take from those papers?** "The backbone (EfficientNet-B0), the idea of high-pass residual filters from steganalysis (SRM), the two-stream idea, the face detector, the Grad-CAM explanation method, the datasets, and the evaluation cautions: generalise across generators, beware shortcuts, scores are not calibrated probabilities."

**3. What did you implement yourself?** "A complete application and evaluation framework: the inference package with training-matched preprocessing and strict checkpoint loading, the image and video pipelines with face tracking and evidence frames, the Grad-CAM views, the PDF report, the FastAPI backend and React dashboard, the model registry, the leakage-controlled training and evaluation tooling, four-model comparison and experiments E1-E4, tests, and a hosted demonstration. The two-stream architecture itself comes from the earlier Forensic-Net prototype by Lakshya Singh. *(Describe your own part truthfully: the repository does not document how the work was divided beyond that acknowledgement.)*"

**4. What is the research gap?** "The papers I reviewed present methods and benchmark numbers. A usable forensic tool also needs per-face evidence, video timelines, honest handling of uncertainty, privacy, and an evaluation that separates unseen identities from unseen generators. The prototype I started from reported very high accuracy that I could not reproduce on independent data, so the model itself had to be re-evaluated."

**5. What is unique about Verif-AI?** "Not a new architecture. What is distinctive is the combination: an inspectable image-and-video forensic application built around a two-stream detector, together with an evaluation that exposes where the detector fails - for example the chance-level result on genuine video, the sharpness shortcut, and the per-quality gate - and a registry that records which model and checkpoint produced every result."

**6. Is Verif-AI a new deep learning architecture?** "No. The architecture (EfficientNet-B0 RGB stream + fixed SRM stream, late fusion) comes from the Forensic-Net prototype, which in turn builds on EfficientNet, the SRM idea and two-stream detectors. My contribution is the system and the experimental method around it."

**7. Why are two streams used?** "They give complementary views: the RGB stream sees visible inconsistencies, the noise stream sees high-frequency traces that generators may leave. I also tested whether the noise stream really helps and found it carries an image-quality shortcut, which is why I report stream-level diagnostics rather than assuming it is beneficial."

**8. Why is SRM used?** "SRM-style high-pass filters remove most image content and keep the noise residual, where manipulation or generator traces can remain (Fridrich and Kodovsky). I use three fixed kernels (the original model uses many more), so the stream's input does not depend on training data."

**9. Why do you use Grad-CAM?** "To give a reviewer something to inspect: which regions of the RGB stream increased the score. It is an explanation of the model, not proof of manipulation, and it says nothing about the noise stream."

**10. How does your video analysis differ from image-only detection?** "It samples about 2 frames per second, links faces into tracks by box overlap, takes the median score per track and the maximum over tracks, marks suspicious time ranges and stores evidence frames with heat-maps. The aggregation rule was chosen in advance and is not validated on labelled videos."

**11. What are the limitations?** "Face-level model; no claim about AI-generated video; scores uncalibrated; E4 failed the promotion rule on DeepFakeFace (missed-fake rate 0.208 internal, 0.509 cross-generator) and is the demonstration default only; Celeb-DF results are within-dataset; `modern_test` is not generator-disjoint; the live demo serves v2 on a free 512 MB host."

**12. What would be the next research direction?** "Generator-disjoint and source-disjoint test sets, shortcut-robust training balanced across quality bins and sources, score calibration, validation of the video aggregation on labelled video, and a better trade-off between DeepFakeFace and Celeb-DF performance."
