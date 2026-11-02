# Verif-AI project report (LaTeX)

**Title:** VERIF-AI: A Deep Learning-Based Framework for Face-Based Deepfake Detection and Forensic Analysis
**Student:** Aniket Raj (Roll 230103030), B.Tech CSE (AI & Data Science), Semester VII, Course CS4202
**Faculty advisor / guide:** Dr. Sunita Warjri, Assistant Professor, Department of CSE, IIIT Manipur - sunitaw@iiitmanipur.ac.in - +91 7085262822 (department page: https://iiitmanipur.ac.in/pages/academic/academicCSE.php); submission: November 2026

## Contents of this folder

| Path | Purpose |
|---|---|
| `Verif-AI_Project_Report.tex` | Main file (front matter, `\input` of chapters, bibliography) |
| `Verif-AI_Project_Report.pdf` | Compiled report |
| `references.bib` | Bibliography (published works only) |
| `chapters/` | One file per chapter + appendices (`ch1_…` – `ch7_…`, `appendix.tex`, `e4_section.tex`) |
| `figures/` | Institute logo and the v2 ROC / confusion-matrix plots (from `docs/figures/v2_selected/`). The block diagrams are TikZ code inside the chapters |
| `screenshots/` | Screenshots of the real running dashboard (dark UI) and rendered pages of a real PDF report |
| `tables/` | LaTeX tables **generated** from the saved experiment records by `scripts/make_tables.py` |
| `listings/` | Unchanged excerpts of the source code shown in Appendix A |
| `data/` | `e4_selection.json` (E4 metrics from the verified Kaggle results zip) |
| `scripts/` | `make_tables.py`, `make_e4_tables.py` (tables), `make_pair_images.py` (equal-aspect copies of paired screenshots, `screenshots/pair_*.png`), `make_zip.py` (packs this folder) |
| `E4_monitoring_log.md` | Chronological log of the E4 Kaggle run checks (append-only) |

## Build

```
latexmk -pdf Verif-AI_Project_Report.tex      # TinyTeX / TeX Live; needs tikz, pgfplots, natbib, listings, tocloft, multirow, algorithms
python report/scripts/make_tables.py          # optional: regenerate tables/ from docs/results/
```

All paths are relative. Packages used are standard CTAN packages.

## Honesty notes (please read)

* **Structure** follows the reference report's section hierarchy; none of its text, results or figures was reused. (The reference file is a different student's project and was not modified.)
* **Results** are copied from the repository's saved result files (v2 evaluation, four-model comparison, E1, stage-0, E3). Nothing was estimated or typed from memory; Table generation is scripted.
* **E4** (source-balanced variant) completed on Kaggle (12 epochs, 134.5 min) and was verified from its results zip; its held-out results are in Table 6.11 and `data/e4_selection.json` (metrics only, copied unchanged from `verifai_e4_results.zip`). It did not meet the frozen promotion rule (criterion 2), so Model 4 remains the default. The E4 checkpoint is not part of this folder.
* **Not claimed:** 95 % (or any fixed) accuracy, accuracy on AI-generated video, calibrated probabilities, or that Grad-CAM proves manipulation. Celeb-DF numbers are within-dataset evidence only. The live demonstration serves the older v2 checkpoint without Grad-CAM and is *not* production-ready.
* **Screenshots** are real captures of the running application using only a public-domain NASA test image and a synthetic clip made from it. No personal faces, private recordings or identifiers appear. No generated mock-ups are used.
* `figures/iiit_manipur_logo.png` was cropped from the cover of the reference report (institute logo).

## Missing material (placeholders / to supply by the student)

* Supervisor and examiner **signatures** and the signing date (left blank on purpose; the certificate carries no designation or approval date).
* Screenshot of a **FAKE verdict** on a real manipulated sample (no suitable public sample was analysed in the UI); the report says so.
* Screenshots of the **deployed (Vercel/Render) site** itself.

## Typography (formatting revision)

Body text uses Century Schoolbook (TeX Gyre Schola) at 10.15 pt on a 13.6 pt leading with 118 pt side margins, matching measurements taken from the reference
internship report; the cover, declaration and certificate pages are placed at positions measured on that report. Needs `tex-gyre` and `adjustbox` (CTAN).

## Length revision

The report is set to exactly 50 pages (A4): chapters 2-7 and the bibliography now continue on the page where the previous part ends (chapter 1 and Appendix A still start new pages), Appendices B-D follow each other, margins/leading were tightened slightly, listings are set smaller and the large screenshots are scaled down. No text, result or table content was changed.

## Final revision (E4 default, novelty, references)
* The report describes **E4 as the default model of the local examiner demonstration** (Model 4 alternative, v2 rollback and live model); it states that E4 failed the frozen promotion rule and is not deployed live.
* New section 2.2 *Research Gap, Project Contributions, and Novelty* (`chapters/ch2_novelty.tex`); `Research_References_and_Novelty.md` lists every cited paper with DOI/link, where it is used, the comparison with Verif-AI and answers for the faculty advisor.
* References: 36 entries, all cited (`scripts/audit_citations.py`), metadata checked against Crossref/arXiv (`scripts/verify_references.py`); DOIs/URLs are printed in the bibliography.
* Every main chapter (1-7), Appendix A and the Bibliography start on a new page; the PDF has exactly 50 pages. Long captions have short forms for the lists of figures/tables (`tables/*.tex` were edited accordingly; regenerate with `scripts/make_tables.py` and `scripts/make_e4_tables.py` and re-add short captions if you rebuild them).
* `handbook/Verif-AI_Project_Viva_Handbook.pdf` is updated (E4 default, novelty, papers, faculty name); it keeps its 100 numbered questions.
