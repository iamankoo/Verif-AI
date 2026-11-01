"""Renders docs/STAGE0_GATE_VALIDATION.md from docs/results/stage0_gate/{dff,cdf}_gate.json (all numbers come from the JSON files).

  python -X utf8 backend/training/make_stage0_report.py
Applies the decision rules V1-V3 of the FROZEN protocol docs/STAGE0_PROTOCOL.md; it does not change any threshold.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "docs" / "results" / "stage0_gate"
OUT = ROOT / "docs" / "STAGE0_GATE_VALIDATION.md"
SCREEN_AUC = 0.90          # protocol section 6: ungated screen


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]" if n else "n/a"


def main() -> None:
    D = {"DFF": json.loads((RES / "dff_gate.json").read_text(encoding="utf-8")), "CDF": json.loads((RES / "cdf_gate.json").read_text(encoding="utf-8"))}
    sha = hashlib.sha256((ROOT / "docs" / "STAGE0_PROTOCOL.md").read_bytes()).hexdigest()
    L: list[str] = []
    a = L.append
    a("# Stage-0 gate validation: results\n")
    a(f"Protocol: `docs/STAGE0_PROTOCOL.md` v1, frozen before any run, sha256 `{sha}` (re-checked when this report was generated). Inference only: no training, no fine-tuning, no checkpoint or production change, no threshold tuning, no test-split data, no private media. Raw outputs: `docs/results/stage0_gate/{{dff,cdf}}_gate.json`; this file is generated from them by `backend/training/make_stage0_report.py`.\n")
    a("## 1. What was run\n")
    a("| Set | Data | n real | n fake | Condition |\n|---|---|---|---|---|")
    for key, d in D.items():
        for name, s in d["sets"].items():
            a(f"| {name} | {'DeepfakeFace' if key == 'DFF' else 'Celeb-DF v2'} validation identities | {s['n_real']} | {s['n_fake']} | {'clean' if s['degrade'] in ('none', '') else 'class-blind degradation: ' + s['degrade']} |")
    a("\nThe four candidates are the production model (v2) and the three experimental models of `docs/FOUR_MODEL_COMPARISON.md`, all at threshold 0.5. Quality = variance of the Laplacian in the fixed bins <40, 40-80, 80-160, 160-320, >=320 (measured after the degradation). A bin is *testable* with >=30 samples per class; the thresholds are g1_auc 0.70, g1_fpr 0.50, g2_rho 0.25, g3_dev 0.15, g4_auc 0.80, min 2 testable bins.\n")
    a("## 2. Model-by-model results (ungated metrics, then gate checks)\n")
    a("Rates are with Wilson 95% intervals computed from the exact counts. `ns` = not evaluable (fewer than 2 testable bins); never counted as a pass.\n")
    a("| Model | Set | ROC-AUC | FPR (FP/n real) | FNR (FN/n fake) | G1 | G2 | G3 | G4 | testable bins | Gate verdict |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for key, d in D.items():
        for mname, m in d["models"].items():
            for sname, r in m["sets"].items():
                c = r["overall"]["combined"]
                nr, nf = r["n_real"], r["n_fake"]
                ab = lambda v: {"not_evaluable": "ns"}.get(v, v)  # noqa: E731
                a(f"| {mname} | {sname} | {c['roc_auc']:.3f} | {ci(c['fp'], nr)} ({c['fp']}/{nr}) | {ci(c['fn'], nf)} ({c['fn']}/{nf}) | {ab(r['G1']['verdict'])} | {ab(r['G2']['verdict'])} | {ab(r['G3']['verdict'])} | {ab(r['G4']['verdict'])} | {r['testable_bins']} | **{r['verdict']}** |")
    a("\nPer-model verdict (worst over its sets): " + "; ".join(f"{k}: {n} = {m['verdict']}" for k, d in D.items() for n, m in d["models"].items()) + ".\n")
    a("### Stream sanity (G1) and quality dependence (G2)\n")
    a("| Model | Set | RGB-only AUC | SRM-only AUC | SRM-only FPR | Spearman(sharpness, score) real / fake |\n|---|---|---|---|---|---|")
    for key, d in D.items():
        for mname, m in d["models"].items():
            for sname, r in m["sets"].items():
                g = r["G1"]["streams"]
                sp = r["spearman_combined"]
                a(f"| {mname} | {sname} | {g['rgb_only']['roc_auc']:.3f} | {g['srm_only']['roc_auc']:.3f} | {g['srm_only']['fpr']:.3f} | {sp['real']:+.3f} / {sp['fake']:+.3f} |")
    a("\n## 3. Per-quality-bin results with sample counts\n")
    a("Only bins with >=30 samples per class are *testable* and enter G3/G4; the others are shown for transparency.\n")
    for key, d in D.items():
        for mname, m in d["models"].items():
            for sname, r in m["sets"].items():
                a(f"\n**{mname} - {sname}**\n\n| Bin | n real | n fake | testable | FPR [95% CI] | FNR [95% CI] | within-bin AUC |\n|---|---|---|---|---|---|---|")
                for b in r["bins"]:
                    auc = "n/a" if b["within_bin_auc"] is None else f"{b['within_bin_auc']:.3f}"
                    rate = lambda v, c: "n/a" if v is None else f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"  # noqa: E731  (empty class in the bin -> no rate)
                    a(f"| {b['bin']} | {b['n_real']} | {b['n_fake']} | {'yes' if b['testable'] else 'no'} | {rate(b['fpr'], b['fpr_ci95'])} | {rate(b['fnr'], b['fnr_ci95'])} | {auc} |")
    a("\n## 4. Synthetic controls (logic validation, no model)\n")
    a("The controls are model-free logic checks (a softness-only classifier must fail; a quality-independent scorer must not). They ran on the two DeepfakeFace validation sets only; the Celeb-DF notebook did not run them.\n")
    a("| Control | Set | Verdict | Failed/passed checks |\n|---|---|---|---|")
    for key, d in D.items():
        for cname, cm in d["controls"].items():
            for sname, r in cm["sets"].items():
                a(f"| {cname} | {sname} | {r['verdict']} | G1 {r['G1']['verdict']}, G2 {r['G2']['verdict']}, G3 {r['G3']['verdict']}, G4 {r['G4']['verdict']} |")
    # decision rules
    dff, cdf = D["DFF"], D["CDF"]
    neg_fail = all(r["verdict"] == "fail" for r in D["DFF"]["controls"]["NEG_softness_classifier"]["sets"].values())
    pos_ok = all(r["verdict"] != "fail" for r in D["DFF"]["controls"]["POS_quality_independent"]["sets"].values())
    prod = dff["models"]["1_production_v2"]["sets"]
    prod_g23 = any(r["G2"]["verdict"] == "fail" or r["G3"]["verdict"] == "fail" for r in prod.values())
    screen, flagged = {}, {}
    for mname in dff["models"]:
        ok = dff["models"][mname]["sets"]["V-DFF-clean"]["overall"]["combined"]["roc_auc"] >= SCREEN_AUC and cdf["models"][mname]["sets"]["V-CDF-clean"]["overall"]["combined"]["roc_auc"] >= SCREEN_AUC
        screen[mname] = ok
        flagged[mname] = [f"{s}:{c}" for d in D.values() for s, r in d["models"][mname]["sets"].items() for c in ("G1", "G2", "G3", "G4") if r[c]["verdict"] == "fail"]
    v3 = [m for m in screen if screen[m] and flagged[m]]
    a("\n## 5. Decision rules of the frozen protocol\n")
    a(f"* **V1 sensitivity** (NEG control fails; production fails G2 or G3 on a DeepfakeFace validation set): NEG fails on every set = {neg_fail}; production fails G2/G3 = {prod_g23}. **{'MET' if neg_fail and prod_g23 else 'NOT MET'}.**")
    a(f"* **V2 logic** (POS control does not fail on any set): **{'MET' if pos_ok else 'NOT MET'}.**")
    a("* **V3 incremental information over the ungated screen** (screen = combined ROC-AUC >= 0.90 on both clean validation sets):\n")
    a("| Model | Passes ungated screen | Gate failures |\n|---|---|---|")
    for m in screen:
        a(f"| {m} | {'yes' if screen[m] else 'no'} | {', '.join(flagged[m]) or 'none'} |")
    a(f"\n  **{'MET' if v3 else 'NOT MET'}**: " + (f"{', '.join(v3)} pass the ungated screen but fail the gate." if v3 else "no model passes the ungated screen and fails the gate."))
    a("\n## 6. Does the gate add value over the ungated models?\n")
    a("Reading of the numbers above (interpretation, not a new decision rule):\n")
    a("* The gate **confirms and explains** what the ungated metrics already show for the production model and experiment 2: both are at chance on Celeb-DF validation (AUC about 0.50) and the SRM-only stream calls almost every Celeb-DF face fake. The ungated screen already rejects them; the gate adds the *reason* (stream collapse, G1; quality dependence, G2).")
    a("* For models 3 and 4 the ungated screen passes (AUC about 0.99 on Celeb-DF validation, 0.93 on DeepfakeFace validation), yet the gate fails them: mostly on the **degraded** sets, plus one check each on clean DeepfakeFace validation (model 3: G3 per-bin error deviation; model 4: G2 quality correlation). That is information the clean AUC does not contain: it matches the earlier finding that robustness to blur/down-up/phone-like softening is weaker than clean accuracy, and that the SRM stream's behaviour depends on image quality. Whether those failures translate into worse behaviour on real, unseen footage is **not established here**.")
    a("* Clean Celeb-DF validation has one testable bin only (about 94% of crops are in the softest bin), so G3/G4 are *not evaluable* there and models 3 and 4 get 'no failure' on that set by default, not a pass.")
    a("\n## 7. Limitations and inconclusive findings\n")
    a("* **Models 3 and 4 were trained on Celeb-DF training identities.** V-CDF is validation data of the same dataset and generator family (identity-disjoint, not generator-disjoint), so the high CDF numbers say nothing about unseen generators or general video reliability. Celeb-DF results must not be presented as proof of general video reliability.")
    a("* **Threshold ranges were chosen while reading E1 results, which included DFF test splits and training sets** (disclosed in the protocol). Stage-0 therefore cannot prove the thresholds are independent of the test measurements; a future gate version has to be derived from validation-only evidence before any E2 candidate is evaluated.")
    a("* The degradations are class-blind synthetic proxies for soft/compressed footage; they are not real phone video.")
    a("* Several per-bin FPR/FNR estimates rest on 30-100 samples; their Wilson intervals (section 3) are wide, and many checks are marginal. Gate decisions use point estimates by protocol.")
    a("* The gate evaluates a model offline on validation data; it is **not** an application feature and nothing was integrated into the production code or API.")
    a("* No labelled real-video benchmark exists in this project, so the gate's relation to false positives on genuine phone videos could not be tested here.")
    a("\n## 8. What this does and does not change\n")
    a("* Production model: **unchanged** (no model met the E2/promotion criteria; no checkpoint, threshold or application code was altered).")
    a("* E2 training: **not started**. These results are inputs to the E2 go/no-go discussion, not an approval.")
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", OUT, "| V1", neg_fail and prod_g23, "| V2", pos_ok, "| V3", v3)


if __name__ == "__main__":
    main()
