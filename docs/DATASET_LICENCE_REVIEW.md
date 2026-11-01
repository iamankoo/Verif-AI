# Dataset and licence review (candidate training data for E3)

Method: (1) the Kaggle dataset descriptions were read; (2) a private Kaggle notebook profiled the files (counts, formats, resolution, file size, sharpness; statistics only, no images exported: `docs/results/data_inspect/`); (3) the *upstream* terms of the named sources were applied. **The licence badge shown on Kaggle is the uploader's claim and was not accepted as proof.** This is an engineering review, not legal advice. Intended use: a non-commercial college project; trained weights are research artefacts and must not be redistributed commercially.

## Decisions
| Dataset (Kaggle copy) | Kaggle badge | What the description and files show | Decision |
|---|---|---|---|
| RealVsFake 162k (`wish096`) | CC0 | Reals are FFHQ (40.5k) and CelebA (40.5k); fakes are StyleGAN (70k), Stable Diffusion (10k), AiGenImage (1k). CelebA is non-commercial research only and FFHQ is CC BY-NC-SA, so a CC0 badge is not credible. **Shortcut found:** about half of the real images are PNG while 99% of the fakes are JPG. | **Excluded** (rights contradictory; format shortcut) |
| DeepFakeFusion-399K (`ajaysonicu`) | MIT | Composite of Celeb-DF v2 frames, FaceForensics++ C32, "Deepfake and Real Images", GRAVEX-200K, a 140K real/fake faces set and a StyleGAN3 set; file names embed the source (e.g. `..._baseline_cdfv2`). FaceForensics++ requires a signed agreement and Celeb-DF is research-only, so MIT cannot apply to the whole. It also contains Celeb-DF frames, which would leak into our own Celeb-DF evaluation. | **Excluded** (mixed rights; overlaps our test data) |
| DeepDetect-2025 (`ayushmandatta1`) | Apache-2.0 | Fakes generated with StyleGAN3, DALL·E 3, Midjourney and Stable Diffusion 3 (modern generators); reals "sourced from publicly available datasets and photographic repositories" (unspecified; a sample file name matches an FFHQ id). Images are 256 px and the description lists non-face categories (animals, nature, artworks). No per-image generator labels were found. | **Fake side only**, face-filtered, as training/validation/`modern_test` fakes. The creator generated those outputs and declares Apache-2.0; residual risk: the generator providers' output terms were not verified. **Reals not used.** |
| Real vs AI Generated Faces (`philosopher0808`) | MIT | `data_source/ffhq` (70k real FFHQ), `fake/sfhq` (SFHQ, 20k), `fake/thispersondoesnotexist` (20k), `fake/stable_diffusion` (1k), `fake/faceswap` (9k); the `dataset/` folder is a processed train/test/validate copy of the same files (do not use both). | **FFHQ reals** (upstream CC BY-NC-SA 4.0: non-commercial research with attribution) and **SFHQ fakes** (upstream CC0). **Excluded:** thispersondoesnotexist, stable_diffusion and faceswap folders (provenance not established); the processed `dataset/` copy (duplicates). |
| DeepFakeFace, Celeb-DF v2 (already used) | - | Unchanged from earlier experiments; Celeb-DF terms are research-use only. | Kept as before |

## Attribution required by the data used
FFHQ: Karras, Laine, Aila (NVIDIA), CC BY-NC-SA 4.0, photographs by many Flickr users under their own licences (see the FFHQ repository). SFHQ: Synthetic Faces High Quality (CC0). Celeb-DF v2: Li et al. (research use). DeepFakeFace: its authors' terms.

## Shortcuts measured (variance of Laplacian = sharpness; median over 60 sampled images per folder)
| Folder | Format | Sharpness (median) | File size (median) |
|---|---|---|---|
| FFHQ (real) | JPG 512 | 284 | 33 KB |
| SFHQ pt1-pt4 (fake) | JPG 512 | 54 - 143 | 21 - 28 KB |
| DeepDetect train real / fake | JPG 256 | 968 / 822 | 28.5 / 27.7 KB |
| DeepDetect **test** real / fake | JPG 256 | **121 / 732** | **8.9 / 27.2 KB** |
| RealVsFake Real / Fake | 50% PNG / 99% JPG | 302 / 423 | 29 / 34 KB |
Findings: SFHQ fakes are much softer than FFHQ reals (the same softness shortcut found in DeepFakeFace), so quality-aware training and evaluation are required; DeepDetect's test-split reals differ strongly from its train-split reals, a distribution shift that makes its real side unsuitable as a test reference (another reason the real side is not used); RealVsFake has a format shortcut. The E3 protocol (`docs/E3_PROTOCOL.md`) therefore uses sharpness-aware augmentation, file-disjoint splits and near-duplicate removal, and reports the real-source/fake-source confound as a limitation.

## What this review cannot establish
Whether every FFHQ photograph's individual Flickr licence permits this use; whether generator providers' terms cover redistribution of outputs; whether any image depicts a person who has not consented to such use. No dataset here is cleared for commercial use.
