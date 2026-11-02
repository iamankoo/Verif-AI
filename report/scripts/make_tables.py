"""Generates LaTeX tables for the report from the saved experiment records (no number is typed by hand).
   python -X utf8 report/scripts/make_tables.py
Inputs : docs/results/e3_modern/{e3_selection.json,e3_data.json} and, when it exists and has been verified, docs/results/e4_modern/e4_selection.json
Output : report/tables/e3_tables.tex (and report/tables/e4_tables.tex when E4 results are present)"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ["e3a_modern_qa0.5", "e3b_modern_noQA", "model4", "v2"]
NAME = {"e3a_modern_qa0.5": "E3A", "e3b_modern_noQA": "E3B", "model4": "Model 4", "v2": "v2", "e4a_dffx4_modern_qa0.5": "E4"}
SPLIT = {"val": "Validation", "test_internal": "DFF internal test", "test_crossgen": "DFF cross-generator test", "cdf_test": "Celeb-DF test (frames)", "modern_test": "modern\\_test"}


def wilson(k, n, z=1.96):
    if n <= 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def rate(p, n):
    lo, hi = wilson(round(p * n), n)
    return f"{p:.3f} [{lo:.2f},{hi:.2f}]"


def flagged(v):
    return v["fake_detection_rate"] if "fake_detection_rate" in v else 1.0 - v["real_acceptance_rate"]


def tex(s):
    return s.replace("_", "\\_")


FIT_OPEN = "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{"


def FIT(t):
    """scale a wide tabular down to the text width (never up)"""
    return t.replace("\\begin{tabular}", FIT_OPEN + "\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)


def build(sel, runs, label_prefix):
    T = {r["run"]: r for r in sel["table"]}

    def counts(run, split):
        pg = T[run][f"{split}_none_per_generator"]
        nr = sum(v["n"] for g, v in pg.items() if g.startswith("real"))
        return nr, sum(v["n"] for v in pg.values()) - nr

    L = []
    # --- frame-level table
    L.append("\\begin{table}[ht]\\centering\\caption{Frame-level results, clean inputs, fixed threshold 0.5 (FPR: real called FAKE; FNR: fake called REAL; 95\\% Wilson intervals). Counts are real / fake crops.}"
             f"\\label{{tab:{label_prefix}-frame}}\\scriptsize\\begin{{tabular}}{{|l|l|c|c|c|c|c|}}\\hline")
    L.append("\\textbf{Split} & \\textbf{Model} & \\textbf{n (R/F)} & \\textbf{Acc} & \\textbf{AUC} & \\textbf{FPR} & \\textbf{FNR} \\\\ \\hline")
    for sp in ("val", "test_internal", "test_crossgen", "cdf_test", "modern_test"):
        for run in runs:
            m = T[run][f"{sp}_none"]
            nr, nf = counts(run, sp)
            L.append(f"{SPLIT[sp] if run == runs[0] else ''} & {NAME[run]} & {nr}/{nf} & {m['accuracy']:.3f} & {m['roc_auc']:.3f} & {rate(m['false_positive_rate'], nr)} & {rate(m['false_negative_rate'], nf)} \\\\")
        L.append("\\hline")
    L.append("\\end{tabular}\\end{table}\n")
    # --- video-level
    L.append(f"\\begin{{table}}[ht]\\centering\\caption{{Video-level results on the Celeb-DF held-out identities (91 real and 56 fake videos), clean inputs.}}\\label{{tab:{label_prefix}-video}}\\small\\begin{{tabular}}{{|l|c|c|c|c|}}\\hline")
    L.append("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{ROC-AUC} & \\textbf{FPR [95\\% CI]} & \\textbf{FNR [95\\% CI]} \\\\ \\hline")
    for run in runs:
        m = T[run]["cdf_test_none_VIDEO"]
        L.append(f"{NAME[run]} & {m['accuracy']:.3f} & {m['roc_auc']:.3f} & {rate(m['false_positive_rate'], 91)} & {rate(m['false_negative_rate'], 56)} \\\\ \\hline")
    L.append("\\end{tabular}\\end{table}\n")
    # --- per generator
    L.append(f"\\begin{{table}}[ht]\\centering\\caption{{Share of crops called FAKE per source, clean inputs (for real sources this is the false-positive rate; for fake sources it is the detection rate).}}\\label{{tab:{label_prefix}-pergen}}\\scriptsize\\begin{{tabular}}{{|l|l|c|{'c|' * len(runs)}}}\\hline")
    L.append("\\textbf{Split} & \\textbf{Source} & \\textbf{n} & " + " & ".join(f"\\textbf{{{NAME[r]}}}" for r in runs) + " \\\\ \\hline")
    for sp in ("modern_test", "test_internal", "test_crossgen", "cdf_test"):
        for g, v in sorted(T["v2"][f"{sp}_none_per_generator"].items()):
            cells = " & ".join(rate(flagged(T[r][f"{sp}_none_per_generator"][g]), v["n"]) for r in runs)
            L.append(f"{tex(sp) if g == sorted(T['v2'][f'{sp}_none_per_generator'])[0] else ''} & {tex(g)} & {v['n']} & {cells} \\\\")
        L.append("\\hline")
    L.append("\\end{tabular}\\end{table}\n")
    # --- robustness
    L.append(f"\\begin{{table}}[ht]\\centering\\caption{{Robustness to a class-blind \\emph{{mixed}} degradation (blur, down-up resampling, JPEG, noise, low light) applied identically to every crop.}}\\label{{tab:{label_prefix}-mixed}}\\small\\begin{{tabular}}{{|l|c|c|c|c|c|}}\\hline")
    L.append("\\textbf{Model} & \\textbf{modern\\_test Acc} & \\textbf{FPR} & \\textbf{FNR} & \\textbf{Celeb-DF video Acc} & \\textbf{video FPR} \\\\ \\hline")
    for run in runs:
        m, v = T[run]["modern_test_mixed"], T[run]["cdf_test_mixed_VIDEO"]
        L.append(f"{NAME[run]} & {m['accuracy']:.3f} & {m['false_positive_rate']:.3f} & {m['false_negative_rate']:.3f} & {v['accuracy']:.3f} & {v['false_positive_rate']:.3f} \\\\ \\hline")
    L.append("\\end{tabular}\\end{table}\n")
    body = "\n".join(L).replace("\\begin{tabular}", FIT_OPEN + "\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")
    return body, T, counts


def promotion(sel, T, counts, cand):
    e, m4, v2 = T[cand], T["model4"], T["v2"]
    nr, nf = counts(cand, "modern_test")
    ba = lambda r: 1 - (r["modern_test_none"]["false_positive_rate"] + r["modern_test_none"]["false_negative_rate"]) / 2

    def se(r):
        m = r["modern_test_none"]
        p, q = m["false_positive_rate"], m["false_negative_rate"]
        return math.sqrt((p * (1 - p) / nr + q * (1 - q) / nf) / 4)

    diff = ba(e) - ba(m4)
    lo = diff - 1.96 * math.sqrt(se(e) ** 2 + se(m4) ** 2)
    c1 = e["cdf_test_none_VIDEO"]["false_positive_rate"] <= m4["cdf_test_none_VIDEO"]["false_positive_rate"] + 0.03 and e["cdf_test_none_VIDEO"]["accuracy"] >= 0.95
    rows = []
    rows.append(("1", "Celeb-DF video FPR $\\le$ Model 4's + 0.03 and accuracy $\\ge0.95$", f"FPR {e['cdf_test_none_VIDEO']['false_positive_rate']:.3f} vs {m4['cdf_test_none_VIDEO']['false_positive_rate']:.3f}; acc {e['cdf_test_none_VIDEO']['accuracy']:.3f}", c1))
    c2 = True
    for sp in ("test_internal", "test_crossgen"):
        fnr, lim, fpr = e[f"{sp}_none"]["false_negative_rate"], v2[f"{sp}_none"]["false_negative_rate"] + 0.05, e[f"{sp}_none"]["false_positive_rate"]
        ok = fnr <= lim and fpr <= 0.20
        c2 &= ok
        rows.append(("2", f"DFF {tex(sp)}: FNR $\\le$ v2's + 0.05 and FPR $\\le0.20$", f"FNR {fnr:.3f} (limit {lim:.3f}); FPR {fpr:.3f}", ok))
    c3 = diff >= 0.10 and lo > 0
    rows.append(("3", "modern\\_test balanced accuracy $\\ge$ Model 4's + 0.10, interval above 0", f"{ba(e):.3f} vs {ba(m4):.3f} (diff {diff:+.3f}, lower bound {lo:+.3f})", c3))
    t = [f"\\begin{{table}}[ht]\\centering\\caption{{The frozen promotion rule applied to {NAME[cand]} (rule written before training).}}\\label{{tab:promo-{cand.split('_')[0]}}}\\small\\begin{{tabular}}{{|c|p{{5.2cm}}|p{{4.2cm}}|c|}}\\hline",
         "\\textbf{\\#} & \\textbf{Criterion} & \\textbf{Value} & \\textbf{Result} \\\\ \\hline"]
    for n, c, v, ok in rows:
        t.append(f"{n} & {c} & {v} & \\textbf{{{'PASS' if ok else 'FAIL'}}} \\\\ \\hline")
    t.append("\\end{tabular}\\end{table}\n")
    return "\n".join(t), (c1 and c2 and c3)


def main():
    sel = json.loads((ROOT / "docs/results/e3_modern/e3_selection.json").read_text(encoding="utf-8"))
    body, T, counts = build(sel, RUNS, "e3")
    pr, promoted = promotion(sel, T, counts, sel["selected"])
    (OUT / "e3_tables.tex").write_text("% generated by report/scripts/make_tables.py from docs/results/e3_modern/e3_selection.json\n" + body + "\n" + pr, encoding="utf-8")
    print("wrote e3_tables.tex | selected", sel["selected"], "| promoted by the frozen rule:", promoted)
    e4 = ROOT / "docs" / "results" / "e4_modern" / "e4_selection.json"
    if e4.is_file():
        s4 = json.loads(e4.read_text(encoding="utf-8"))
        cand = s4["selected"]
        runs4 = [cand, "model4", "v2"]
        b4, T4, c4 = build(s4, runs4, "e4")
        p4, promoted4 = promotion(s4, T4, c4, cand)
        (OUT / "e4_tables.tex").write_text("% generated from docs/results/e4_modern/e4_selection.json\n" + b4 + "\n" + p4, encoding="utf-8")
        print("wrote e4_tables.tex | promoted by the frozen rule:", promoted4)


if __name__ == "__main__":
    main()
