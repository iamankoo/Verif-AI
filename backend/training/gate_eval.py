"""Stage-0 per-bin ACCEPTANCE GATE: an offline model-acceptance test evaluated on VALIDATION data. It is not an inference rule and is not part of the application.

Protocol: docs/STAGE0_PROTOCOL.md version 1 (frozen). Thresholds below are frozen with it; changing them is a new protocol version.
Checks (per model, per validation set): G1 stream sanity, G2 stream-quality independence, G3 per-bin stability, G4 within-bin discrimination.
Inference only: checkpoints are run forward; nothing is trained, tuned or modified; the decision threshold is fixed at 0.5.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from .quality_bins import BIN_NAMES, bin_index, model_by_bin, wilson

PROTOCOL_VERSION = 1
THRESHOLDS = dict(g1_auc=0.70, g1_fpr=0.50, g2_rho=0.25, g3_dev=0.15, g4_auc=0.80, bin_min=30, min_testable_bins=2)   # FROZEN (protocol v1)


# ----------------------------------------------------------------------------------------------- small statistics helpers
def _auc(y, s) -> Optional[float]:
    y = np.asarray(y)
    return None if len(set(y.tolist())) < 2 else float(roc_auc_score(y, s))


def _auc_hm_se(auc: float, n1: int, n0: int) -> float:
    """Hanley-McNeil standard error of an ROC-AUC (used only to flag 'marginal' checks)."""
    q1, q2 = auc / (2 - auc), 2 * auc * auc / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc * auc) + (n0 - 1) * (q2 - auc * auc)) / (n1 * n0))


def _rho_ci(rho: float, n: int):
    if n < 4 or abs(rho) >= 1:
        return (None, None)
    z, se = math.atanh(rho), 1 / math.sqrt(n - 3)
    return (round(math.tanh(z - 1.96 * se), 4), round(math.tanh(z + 1.96 * se), 4))


# ----------------------------------------------------------------------------------------------- the gate on one validation set
def gate_checks(sharp, labels, scores: Dict[str, np.ndarray], th: Dict = THRESHOLDS) -> Dict:
    """scores: {'combined','rgb_only','srm_only'} fake-scores (higher = more FAKE). Returns every number behind each verdict."""
    sharp, labels = np.asarray(sharp, float), np.asarray(labels, int)
    nr, nf = int((labels == 0).sum()), int((labels == 1).sum())
    out: Dict = dict(n_real=nr, n_fake=nf, overall={}, thresholds=dict(th))
    for v, s in scores.items():
        s = np.asarray(s, float)
        pred = s > 0.5
        fp, fn = int((pred & (labels == 0)).sum()), int((~pred & (labels == 1)).sum())
        out["overall"][v] = dict(roc_auc=_auc(labels, s), fpr=fp / nr if nr else None, fnr=fn / nf if nf else None, fp=fp, fn=fn)
    # ---- G1: each stream, with the other neutralised, must carry label evidence and must not call everything FAKE
    g1, g1_fail, g1_marg = {}, False, False
    for v in ("rgb_only", "srm_only"):
        o = out["overall"][v]
        auc, fpr = o["roc_auc"], o["fpr"]
        se = _auc_hm_se(auc, nf, nr) if auc else None
        ci_fpr = wilson(o["fp"], nr)
        ok = auc is not None and auc >= th["g1_auc"] and fpr <= th["g1_fpr"]
        marg = (se is not None and abs(auc - th["g1_auc"]) < 1.96 * se) or (ci_fpr[0] is not None and ci_fpr[0] <= th["g1_fpr"] <= ci_fpr[1])
        g1[v] = dict(roc_auc=auc, roc_auc_se=None if se is None else round(se, 4), fpr=fpr, fpr_ci95=list(ci_fpr), passed=bool(ok), marginal=bool(marg))
        g1_fail |= not ok
        g1_marg |= marg
    out["G1"] = dict(verdict="fail" if g1_fail else "pass", marginal=g1_marg, streams=g1)
    # ---- G2: stream scores must not move with sharpness
    g2, g2_fail, g2_marg = {}, False, False
    for v in ("rgb_only", "srm_only"):
        s = np.asarray(scores[v], float)
        d = {}
        for lab, name in ((0, "real"), (1, "fake")):
            m = labels == lab
            rho = float(spearmanr(sharp[m], s[m])[0]) if m.sum() > 3 else float("nan")
            ci = _rho_ci(rho, int(m.sum())) if not math.isnan(rho) else (None, None)
            ok = (not math.isnan(rho)) and abs(rho) <= th["g2_rho"]
            marg = ci[0] is not None and (ci[0] <= th["g2_rho"] <= ci[1] or ci[0] <= -th["g2_rho"] <= ci[1])
            d[name] = dict(spearman=None if math.isnan(rho) else round(rho, 4), ci95=list(ci), n=int(m.sum()), passed=bool(ok), marginal=bool(marg))
            g2_fail |= not ok
            g2_marg |= marg
        g2[v] = d
    out["G2"] = dict(verdict="fail" if g2_fail else "pass", marginal=g2_marg, streams=g2)
    # ---- G3 / G4: per-bin stability and within-bin discrimination of the combined score (testable bins only)
    mb = model_by_bin(sharp, labels, np.asarray(scores["combined"], float))
    fpr_all, fnr_all = out["overall"]["combined"]["fpr"], out["overall"]["combined"]["fnr"]
    bins, testable = [], 0
    g3_fail = g4_fail = g3_marg = g4_marg = False
    for b in mb["per_bin"]:
        t = b["n_real"] >= th["bin_min"] and b["n_fake"] >= th["bin_min"]
        row = dict(bin=b["bin"], n_real=b["n_real"], n_fake=b["n_fake"], testable=bool(t), fpr=b["fpr"], fpr_ci95=b["fpr_ci95"], fnr=b["fnr"], fnr_ci95=b["fnr_ci95"], within_bin_auc=b["within_bin_auc"])
        if t:
            testable += 1
            dfp, dfn = abs(b["fpr"] - fpr_all), abs(b["fnr"] - fnr_all)
            row.update(dev_fpr=round(dfp, 4), dev_fnr=round(dfn, 4), g3_passed=bool(dfp <= th["g3_dev"] and dfn <= th["g3_dev"]),
                       g4_passed=bool(b["within_bin_auc"] is not None and b["within_bin_auc"] >= th["g4_auc"]))
            lo_p, hi_p, lo_n, hi_n = *b["fpr_ci95"], *b["fnr_ci95"]
            row["g3_marginal"] = bool((lo_p <= fpr_all + th["g3_dev"] <= hi_p) or (lo_p <= fpr_all - th["g3_dev"] <= hi_p) or (lo_n <= fnr_all + th["g3_dev"] <= hi_n) or (lo_n <= fnr_all - th["g3_dev"] <= hi_n))
            g3_fail |= not row["g3_passed"]; g4_fail |= not row["g4_passed"]; g3_marg |= row["g3_marginal"]
        bins.append(row)
    evaluable = testable >= th["min_testable_bins"]
    out["bins"] = bins
    out["testable_bins"] = testable
    out["G3"] = dict(verdict=("not_evaluable" if not evaluable else "fail" if g3_fail else "pass"), marginal=g3_marg)
    out["G4"] = dict(verdict=("not_evaluable" if not evaluable else "fail" if g4_fail else "pass"), marginal=g4_marg)
    out["spearman_combined"] = mb["spearman_sharpness_vs_score"]
    out["verdict"] = "fail" if any(out[g]["verdict"] == "fail" for g in ("G1", "G2", "G3", "G4")) else "no_failure"
    return out


def model_verdict(per_set: Dict[str, Dict]) -> Dict:
    """FAIL if any evaluable check fails on any set; PASS if no check fails, G1/G2 are evaluated on every set and G3/G4 are evaluable on >= 1 set; else INCONCLUSIVE."""
    fails = [(s, g) for s, r in per_set.items() for g in ("G1", "G2", "G3", "G4") if r[g]["verdict"] == "fail"]
    if fails:
        return dict(verdict="FAIL", failed_checks=[f"{s}:{g}" for s, g in fails])
    g34_evaluable = any(r["G3"]["verdict"] != "not_evaluable" for r in per_set.values())
    return dict(verdict="PASS" if g34_evaluable else "INCONCLUSIVE", failed_checks=[],
                note=None if g34_evaluable else "G3/G4 not evaluable on any set (too few testable bins)")


# ----------------------------------------------------------------------------------------------- synthetic controls (logic validation, no model)
def neg_control(sharp) -> Dict[str, np.ndarray]:
    """'Softness classifier': softer -> FAKE. Quality IS the label proxy, so the gate must FAIL it."""
    sharp = np.maximum(np.asarray(sharp, float), 1e-3)
    s = 1.0 / (1.0 + np.exp(-3.0 * (np.log(np.median(sharp)) - np.log(sharp))))
    return dict(combined=s, rgb_only=s, srm_only=s)


def pos_control(labels, seed: int = 0) -> Dict[str, np.ndarray]:
    """Quality-independent scorer: label signal plus noise unrelated to quality. The gate must PASS it."""
    labels = np.asarray(labels, float)
    rng = np.random.default_rng(seed)
    s = np.clip(0.5 + 0.7 * (labels - 0.5) + rng.normal(0, 0.12, len(labels)), 0, 1)
    return dict(combined=s, rgb_only=s, srm_only=s)


# ----------------------------------------------------------------------------------------------- running on a split
def set_measurements(manifest: str, split: str, degrade: str = ""):
    """Sharpness is measured on the crop AS THE MODEL SEES IT (after the deterministic degradation, if any)."""
    import cv2
    from PIL import Image
    from .dataset import read_manifest
    from .degrade import FIXED_NAMES, fixed_degrade
    rows = read_manifest(manifest, [split])
    sharp = []
    for i, r in enumerate(rows):
        im = Image.open(r["crop_path"]).convert("RGB")
        if degrade:
            im = fixed_degrade(im, FIXED_NAMES[i % len(FIXED_NAMES)] if degrade == "mixed" else degrade, seed=i)
        sharp.append(float(cv2.Laplacian(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
    return rows, np.array(sharp), np.array([int(r["label"]) for r in rows])


def run(models: Dict[str, str], manifest: str, sets: Dict[str, Dict], ref_manifest: str = "", n_ref: int = 150, device: str = "auto", with_controls: bool = False) -> Dict:
    """sets: {'V-DFF-clean': {'split': 'val', 'degrade': ''}, 'V-DFF-stress': {'split': 'val', 'degrade': 'mixed'}, ...}"""
    from verifai.model import load_model
    from .dataset import FaceCropDataset
    from .stream_ablation import collect, head, reference_means

    meas = {name: set_measurements(manifest, d["split"], d.get("degrade", "")) for name, d in sets.items()}
    result: Dict = dict(protocol_version=PROTOCOL_VERSION, thresholds=THRESHOLDS, bins=list(BIN_NAMES), sets={n: dict(split=d["split"], degrade=d.get("degrade", "") or "none",
                        n=int(len(m[2])), n_real=int((m[2] == 0).sum()), n_fake=int((m[2] == 1).sum())) for (n, d), m in zip(sets.items(), meas.values())}, models={}, controls={})
    for name, ck in models.items():
        b = load_model(ck, device)
        model = b.model.eval()
        r1, r2 = reference_means(model, ref_manifest or manifest, "train", n_ref, 0)
        r1, r2 = r1.to(b.device), r2.to(b.device)
        per_set = {}
        for sname, d in sets.items():
            rows, sharp, y = meas[sname]
            ds = FaceCropDataset(manifest, [d["split"]], train=False, degrade=d.get("degrade", ""))
            f1, f2 = collect(model, ds)
            f1, f2 = f1.to(b.device), f2.to(b.device)
            sc = dict(combined=head(model, f1, f2), rgb_only=head(model, f1, r2.expand_as(f2)), srm_only=head(model, r1.expand_as(f1), f2))
            per_set[sname] = gate_checks(sharp, y, sc)
        result["models"][name] = dict(checkpoint=b.info.checkpoint_file, sha256=b.info.checkpoint_sha256, sets=per_set, **model_verdict(per_set))
        del b, model
    if with_controls:
        for cname, fn in (("NEG_softness_classifier", lambda sh, y: neg_control(sh)), ("POS_quality_independent", lambda sh, y: pos_control(y))):
            per_set = {sname: gate_checks(meas[sname][1], meas[sname][2], fn(meas[sname][1], meas[sname][2])) for sname in sets}
            result["controls"][cname] = dict(sets=per_set, **model_verdict(per_set))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref-manifest", default="")
    ap.add_argument("--set", action="append", required=True, help="name:split[:degrade]  e.g. V-DFF-clean:val  or  V-DFF-stress:val:mixed")
    ap.add_argument("--model", action="append", required=True, help="name=path.pth")
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = {}
    for s in a.set:
        parts = s.split(":")
        sets[parts[0]] = dict(split=parts[1], degrade=parts[2] if len(parts) > 2 else "")
    res = run(dict(m.split("=", 1) for m in a.model), a.manifest, sets, a.ref_manifest, with_controls=a.controls)
    json.dump(res, open(a.out, "w"), indent=1)
    for n, m in res["models"].items():
        print(f"{n}: {m['verdict']} {m['failed_checks']}")
    for n, m in res["controls"].items():
        print(f"{n}: {m['verdict']} {m['failed_checks']}")


if __name__ == "__main__":
    main()
