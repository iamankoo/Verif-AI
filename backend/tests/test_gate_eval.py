"""Stage-0 gate logic (protocol v1): thresholds are frozen; the synthetic NEG control fails, the POS control passes; 'not evaluable' is never a pass."""
import numpy as np

from test_training_smoke import make_manifest
from training.gate_eval import THRESHOLDS, gate_checks, model_verdict, neg_control, pos_control, run


def _data(n=800, seed=0, proxy=True):
    """Half real / half fake. With proxy=True fakes are much softer (quality is a label proxy); with proxy=False quality is independent of the label."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    if proxy:
        sharp = np.concatenate([np.exp(rng.uniform(np.log(60), np.log(700), n // 2)), np.exp(rng.uniform(np.log(8), np.log(200), n // 2))])
    else:
        sharp = np.exp(rng.uniform(np.log(8), np.log(700), n))
    return sharp, y


def test_thresholds_are_the_frozen_protocol_v1_values():
    assert THRESHOLDS == dict(g1_auc=0.70, g1_fpr=0.50, g2_rho=0.25, g3_dev=0.15, g4_auc=0.80, bin_min=30, min_testable_bins=2)


def test_neg_control_softness_classifier_fails_the_gate():
    sharp, y = _data(proxy=True)
    r = gate_checks(sharp, y, neg_control(sharp))
    assert r["verdict"] == "fail" and (r["G2"]["verdict"] == "fail" or r["G3"]["verdict"] == "fail")
    assert model_verdict({"set": r})["verdict"] == "FAIL"


def test_pos_control_quality_independent_scorer_passes_the_gate():
    sharp, y = _data(proxy=False, n=1600)
    r = gate_checks(sharp, y, pos_control(y))
    assert r["G1"]["verdict"] == "pass" and r["G2"]["verdict"] == "pass"
    assert r["G3"]["verdict"] == "pass" and r["G4"]["verdict"] == "pass", r["bins"]
    assert model_verdict({"set": r})["verdict"] == "PASS"


def test_single_testable_bin_is_not_evaluable_and_never_a_pass():
    rng = np.random.default_rng(1)
    y = np.array([0] * 200 + [1] * 200)
    sharp = rng.uniform(5, 30, 400)                       # everything lands in the softest bin
    s = np.clip(0.5 + 0.7 * (y - 0.5) + rng.normal(0, 0.12, 400), 0, 1)
    r = gate_checks(sharp, y, dict(combined=s, rgb_only=s, srm_only=s))
    assert r["testable_bins"] == 1 and r["G3"]["verdict"] == "not_evaluable" and r["G4"]["verdict"] == "not_evaluable"
    assert model_verdict({"set": r})["verdict"] == "INCONCLUSIVE"


def test_stream_that_calls_everything_fake_fails_g1():
    sharp, y = _data(proxy=False, n=800)
    always_fake = np.full(len(y), 0.97)
    good = pos_control(y)
    r = gate_checks(sharp, y, dict(combined=good["combined"], rgb_only=good["rgb_only"], srm_only=always_fake))
    assert r["G1"]["verdict"] == "fail" and not r["G1"]["streams"]["srm_only"]["passed"]


def test_end_to_end_on_a_tiny_manifest(tmp_path, random_checkpoint):
    m = make_manifest(tmp_path)
    res = run({"rand": str(random_checkpoint)}, str(m), {"V-clean": dict(split="val", degrade=""), "V-stress": dict(split="val", degrade="mixed")}, n_ref=4, device="cpu", with_controls=True)
    assert set(res["models"]["rand"]["sets"]) == {"V-clean", "V-stress"} and res["models"]["rand"]["verdict"] in ("FAIL", "PASS", "INCONCLUSIVE")
    assert set(res["controls"]) == {"NEG_softness_classifier", "POS_quality_independent"}
    assert res["protocol_version"] == 1 and res["sets"]["V-stress"]["degrade"] == "mixed"
