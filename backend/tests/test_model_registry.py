"""Model registry: roles (production default / rollback / candidate), selection API, and the documented checkpoints.

Synthetic-weight tests always run. Tests that load the real Model 4 / v2 / E4 checkpoints are skipped when a file is absent
(Model 4 and E4 are not stored in Git).
"""
import hashlib
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from conftest import StubDetector, write_png
from verifai.api.app import create_app
from verifai.config import BACKEND_DIR, Settings
from verifai.engine import Engine
from verifai.model import TwoStreamNet, load_model, sha256_file
from verifai.registry import KNOWN, ModelRegistry

BOX = [60, 40, 200, 200]
MODELS = BACKEND_DIR / "models"
# SHA-256 values recorded in docs/CURRENT_DEFAULT_MODEL.md and docs/E4_INTEGRATION.md
DOCUMENTED = {
    "verifai_v3_celebdf_qa.pth": "e7e5f3dc28b252d43e2e9134e7f41185c49526cff5d1c03367d14ae3cf07eafb",
    "verifai_v2_best.pth": "3b94775581931210a51bfbb7c3da301ff1d3f38e658fee0ba7649b23fca370d7",
    "verifai_e4_candidate.pth": "0d553de918c120994e0919386d7afb6378cd489cc9ef4abc2aa4850e5cf9cba0",
}


def _save_random(path: Path, seed: int):
    torch.manual_seed(seed)
    m = TwoStreamNet(pretrained=False)
    torch.save({"state_dict": m.state_dict(), "meta": {"name": f"synthetic-{seed}", "version": "t"}}, path)
    return path


@pytest.fixture()
def three_models(tmp_path):
    """Synthetic checkpoints named like the real ones, so the registry assigns Model 4 / v2 / E4 roles."""
    d = tmp_path / "models"
    d.mkdir()
    return (_save_random(d / "verifai_v3_celebdf_qa.pth", 1), _save_random(d / "verifai_v2_best.pth", 2), _save_random(d / "verifai_e4_candidate.pth", 3))


def _client(tmp_path, three, enable_selection=True):
    m4, v2, e4 = three
    s = Settings(storage_dir=tmp_path / "st", device="cpu", model_path=e4, alternative_model_path=m4, e4_model_path=e4, rollback_model_path=v2,
                 enable_model_selection=enable_selection, max_image_mb=1.0, max_video_mb=2.0, video_sample_fps=4.0, video_max_frames=8)
    eng = Engine(load_model(e4, "cpu"), s)
    eng.detector = StubDetector([[BOX]])
    return TestClient(create_app(s, eng)), eng


def test_registry_roles_and_default(three_models, tmp_path):
    m4, v2, e4 = three_models
    s = Settings(storage_dir=tmp_path, model_path=e4, alternative_model_path=m4, e4_model_path=e4, rollback_model_path=v2)
    r = ModelRegistry(s)
    assert r.default_id == "e4"
    assert {k: v.role for k, v in r.entries.items()} == {"e4": "default", "model4": "alternative", "v2": "rollback"}
    assert "not approved for production" in r.entries["e4"].description
    assert r.selectable("e4") and r.selectable("v2") and r.selectable("model4") and not r.selectable("nope")


def test_selection_disabled_exposes_only_the_default(three_models, tmp_path):
    m4, v2, e4 = three_models
    s = Settings(storage_dir=tmp_path, model_path=e4, alternative_model_path=m4, e4_model_path=e4, rollback_model_path=v2, enable_model_selection=False)
    r = ModelRegistry(s)
    assert r.selectable("e4") and not r.selectable("model4") and not r.selectable("v2")
    assert [m["id"] for m in r.describe([])] == ["e4"]


def test_missing_candidate_file_is_not_selectable(three_models, tmp_path):
    m4, v2, e4 = three_models
    e4.unlink()
    r = ModelRegistry(Settings(storage_dir=tmp_path, model_path=m4, alternative_model_path=m4, e4_model_path=e4, rollback_model_path=v2))
    assert r.default_id == "model4" and not r.selectable("e4") and r.selectable("v2")


def test_when_default_is_v2_it_is_the_production_entry(three_models, tmp_path):
    """The hosted free-tier setup serves v2: it is reported as the default of that instance, never as Model 4 or E4 (their files are absent there)."""
    m4, v2, e4 = three_models
    m4.unlink(); e4.unlink()
    r = ModelRegistry(Settings(storage_dir=tmp_path, model_path=v2, alternative_model_path=m4, e4_model_path=e4, rollback_model_path=v2, enable_model_selection=False))
    assert r.default_id == "v2" and r.entries["v2"].role == "default"
    assert [m["id"] for m in r.describe([])] == ["v2"]


def test_models_endpoint_and_default_unchanged(three_models, tmp_path):
    c, eng = _client(tmp_path, three_models)
    with c:
        j = c.get("/api/models").json()
        assert j["default"] == "e4" and j["selection_enabled"] is True
        roles = {m["id"]: m["role"] for m in j["models"]}
        assert roles == {"e4": "default", "model4": "alternative", "v2": "rollback"}
        e4 = [m for m in j["models"] if m["id"] == "e4"][0]
        assert "not approved for production" in e4["description"] and e4["selectable"] and e4["is_default"] and e4["loaded"]
        mm = c.get("/api/model").json()["model"]
        assert mm["registry_id"] == "e4" and mm["role"] == "default" and "e4" in mm["registry_label"].lower()
    eng.shutdown()


def test_image_analysis_labels_the_model_and_switching_never_changes_the_default(three_models, tmp_path):
    c, eng = _client(tmp_path, three_models)
    png = write_png(tmp_path / "a.png").read_bytes()
    with c:
        d = c.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}).json()
        assert d["model"]["registry_id"] == "e4" and d["model"]["role"] == "default"          # an untagged request uses the default (E4)
        m = c.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}, data={"model": "model4"}).json()
        assert m["model"]["registry_id"] == "model4" and m["model"]["role"] == "alternative"
        assert m["model"]["checkpoint_sha256"] != d["model"]["checkpoint_sha256"]
        # the default is still E4 and an untagged request still uses it
        d2 = c.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}).json()
        assert d2["model"]["registry_id"] == "e4"
        assert [m["loaded"] for m in c.get("/api/models").json()["models"] if m["id"] == "model4"] == [True]
        r = c.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}, data={"model": "v2"}).json()
        assert r["model"]["role"] == "rollback" and "rollback" in r["model"]["role_label"].lower()
        loaded = [m["id"] for m in c.get("/api/models").json()["models"] if m["loaded"]]
        assert sorted(loaded) == ["e4", "v2"]           # the default plus at most one optional model stays in memory
    eng.shutdown()


def test_unknown_model_and_disabled_selection_are_rejected_without_leaving_uploads(three_models, tmp_path):
    c, eng = _client(tmp_path, three_models)
    png = write_png(tmp_path / "a.png").read_bytes()
    with c:
        r = c.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}, data={"model": "does-not-exist"})
        assert r.status_code == 422 and r.json()["code"] == "unknown_model"
        r = c.post("/api/analyze/video", files={"file": ("v.mp4", b"x" * 100, "video/mp4")}, data={"model": "does-not-exist"})
        assert r.status_code == 422
        assert list(eng.store.tmp.iterdir()) == []
    eng.shutdown()
    c2, eng2 = _client(tmp_path / "b", three_models, enable_selection=False)
    with c2:
        r = c2.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}, data={"model": "v2"})
        assert r.status_code == 422 and r.json()["code"] == "unknown_model"
        assert c2.post("/api/analyze/image", files={"file": ("a.png", png, "image/png")}).status_code == 200
        assert [m["id"] for m in c2.get("/api/models").json()["models"]] == ["e4"]
    eng2.shutdown()


def test_video_job_records_the_selected_model(three_models, tmp_path):
    """Video jobs and the PDF record the model that was actually selected (here the alternative Model 4, not the default)."""
    import time
    from conftest import write_video
    c, eng = _client(tmp_path, three_models)
    vid = write_video(tmp_path / "v.mp4").read_bytes()
    with c:
        job = c.post("/api/analyze/video", files={"file": ("v.mp4", vid, "video/mp4")}, data={"model": "model4"}).json()
        for _ in range(120):
            st = c.get(f"/api/analyses/{job['analysis_id']}/status").json()
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(0.25)
        assert st["status"] == "completed", st
        res = c.get(f"/api/analyses/{job['analysis_id']}").json()
        assert res["model"]["registry_id"] == "model4" and res["model"]["role"] == "alternative"
        pdf = c.get(f"/api/analyses/{job['analysis_id']}/report.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    eng.shutdown()


# ---------------------------------------------------------------- real checkpoints (skipped when the file is absent)
@pytest.mark.needs_weights
@pytest.mark.parametrize("fname", sorted(DOCUMENTED))
def test_real_checkpoint_hash_and_load(fname):
    p = MODELS / fname
    if not p.is_file():
        pytest.skip(f"{fname} not present")
    assert sha256_file(p) == DOCUMENTED[fname]
    b = load_model(p, "cpu")                                   # strict load: any mismatch raises
    assert b.info.parameters == 11_972_954 and b.info.input_size == 224 and b.info.decision_threshold == 0.5
    assert KNOWN[fname][0] in ("model4", "v2", "e4")


@pytest.mark.needs_weights
def test_real_e4_inference_runs_on_synthetic_input(tmp_path):
    p = MODELS / "verifai_e4_candidate.pth"
    if not p.is_file():
        pytest.skip("E4 candidate checkpoint not present")
    b = load_model(p, "cpu")
    x = torch.randn(3, 3, 224, 224)
    s = b.predict_batch(x)
    assert s.shape == (3,) and ((s >= 0) & (s <= 1)).all()


def test_default_path_prefers_e4_then_model4_then_v2(tmp_path, monkeypatch):
    """A local start-up with no VERIFAI_MODEL_PATH loads E4 automatically; without E4 it falls back to Model 4, then v2."""
    import verifai.config as cfg
    d = tmp_path / "models"
    d.mkdir()
    monkeypatch.setattr(cfg, "BACKEND_DIR", tmp_path)
    assert cfg.default_model_path().name == "verifai_v2_best.pth"                 # nothing present: the (missing) v2 path is returned
    (d / "verifai_v2_best.pth").write_bytes(b"x")
    assert cfg.default_model_path().name == "verifai_v2_best.pth"
    (d / "verifai_v3_celebdf_qa.pth").write_bytes(b"x")
    assert cfg.default_model_path().name == "verifai_v3_celebdf_qa.pth"
    (d / "verifai_e4_candidate.pth").write_bytes(b"x")
    assert cfg.default_model_path().name == "verifai_e4_candidate.pth"


def test_hosted_image_pins_v2_and_hides_selection():
    """The Dockerfile (used by the live Render service) must keep serving v2 with selection off: E4 is not deployed live."""
    df = (BACKEND_DIR.parent / "Dockerfile").read_text(encoding="utf-8")
    assert "VERIFAI_MODEL_PATH=/app/models/verifai_v2_best.pth" in df and "VERIFAI_ENABLE_MODEL_SELECTION=false" in df
