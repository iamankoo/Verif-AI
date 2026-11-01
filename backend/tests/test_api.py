"""API tests (FastAPI TestClient) with a random-weight model and scripted detector."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import StubDetector, write_png, write_video
from verifai import image_pipeline
from verifai.api.app import create_app
from verifai.config import Settings
from verifai.engine import Engine

BOX = [60, 40, 200, 200]


@pytest.fixture()
def client(random_bundle, tmp_path, monkeypatch):
    s = Settings(storage_dir=tmp_path / "st", device="cpu", max_image_mb=1.0, max_video_mb=2.0, video_sample_fps=4.0, video_max_frames=8)
    eng = Engine(random_bundle, s)
    eng.detector = StubDetector([[BOX]])
    with TestClient(create_app(s, eng)) as c:
        c.engine = eng
        yield c
    eng.shutdown()


def png_bytes(tmp_path, **kw):
    return write_png(tmp_path / "up.png", **kw).read_bytes()


def test_health_and_model_info(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_loaded"] and h["device"] == "cpu"
    m = client.get("/api/model").json()
    assert m["model"]["parameters"] == 11_972_954 and m["limits"]["decision_threshold"] == 0.5
    assert ".mp4" in m["limits"]["video_formats"] and ".png" in m["limits"]["image_formats"] and "scope_note" in m


def test_openapi_docs_available(client):
    spec = client.get("/openapi.json").json()
    assert "/api/analyze/image" in spec["paths"] and "/api/analyze/video" in spec["paths"] and client.get("/docs").status_code == 200


def test_image_analysis_end_to_end(client, tmp_path, monkeypatch):
    monkeypatch.setattr(image_pipeline, "score_crops", lambda b, c, bs=16, m=True: np.array([0.93] * len(c)))
    r = client.post("/api/analyze/image", files={"file": ("../../evil name.png", png_bytes(tmp_path), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["classification"] == "FAKE" and d["fake_probability"] == pytest.approx(0.93) and d["real_probability"] == pytest.approx(0.07)
    assert d["media_type"] == "image" and d["faces_detected"] == 1 and len(d["faces"][0]["box"]) == 4
    assert d["filename"] == "evil name.png"                              # client path components are stripped
    assert d["model"]["checkpoint_sha256"] and d["processing_time_s"] >= 0 and "NOT calibrated" in d["score_semantics"]
    aid = d["analysis_id"]
    assert client.get(f"/api/analyses/{aid}").json()["analysis_id"] == aid
    assert client.get(f"/api/analyses/{aid}/status").json()["status"] == "completed"
    art = client.get(f"/api/analyses/{aid}/artifacts/{d['annotated_artifact']}")
    assert art.status_code == 200 and art.headers["content-type"] == "image/jpeg" and art.content[:2] == b"\xff\xd8"
    assert any(s["analysis_id"] == aid for s in client.get("/api/analyses").json())
    pdf = client.get(f"/api/analyses/{aid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-" and pdf.headers["content-type"] == "application/pdf"
    assert client.delete(f"/api/analyses/{aid}").status_code == 204
    assert client.get(f"/api/analyses/{aid}").status_code == 404
    assert not any(client.engine.store.tmp.iterdir())                    # upload deleted


def test_image_no_face_gives_inconclusive_not_error(client, tmp_path):
    client.engine.detector = StubDetector([[]])
    d = client.post("/api/analyze/image", files={"file": ("a.png", png_bytes(tmp_path), "image/png")}).json()
    assert d["classification"] == "INCONCLUSIVE" and d["faces_detected"] == 0


def test_invalid_uploads(client, tmp_path):
    assert client.post("/api/analyze/image", files={"file": ("a.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post("/api/analyze/image", files={"file": ("a.png", b"", "image/png")}).status_code == 422
    r = client.post("/api/analyze/image", files={"file": ("a.png", b"definitely not a png", "image/png")})
    assert r.status_code == 422 and r.json()["code"] == "unreadable_image"
    big = client.post("/api/analyze/image", files={"file": ("a.png", b"\x89PNG" + b"0" * 1_100_000, "image/png")})
    assert big.status_code == 413
    assert client.post("/api/analyze/video", files={"file": ("a.png", b"x", "image/png")}).status_code == 415
    assert client.post("/api/analyze/image").status_code == 422          # missing file field
    assert not any(client.engine.store.tmp.iterdir())                    # rejected uploads leave nothing behind


def test_unknown_ids_and_path_traversal(client):
    assert client.get("/api/analyses/" + "0" * 32).status_code == 404
    assert client.get("/api/analyses/not-a-valid-id").status_code == 404
    assert client.get("/api/analyses/" + "0" * 32 + "/artifacts/x.jpg").status_code == 404
    assert client.get("/api/analyses/" + "0" * 32 + "/report.pdf").status_code == 404


def test_artifact_name_traversal_blocked(client, tmp_path):
    d = client.post("/api/analyze/image", files={"file": ("a.png", png_bytes(tmp_path), "image/png")}).json()
    for bad in ("..%2F..%2Fresult.json", "%2e%2e%2fjob.json", "result.json"):
        assert client.get(f"/api/analyses/{d['analysis_id']}/artifacts/{bad}").status_code == 404


def test_video_job_lifecycle(client, tmp_path, monkeypatch):
    from verifai import video_pipeline
    monkeypatch.setattr(video_pipeline, "score_crops", lambda b, c, bs=16, m=True: np.array([0.9] * len(c)))
    v = write_video(tmp_path / "v.avi").read_bytes()
    r = client.post("/api/analyze/video", files={"file": ("clip.avi", v, "video/x-msvideo")})
    assert r.status_code == 202
    aid = r.json()["analysis_id"]
    for _ in range(120):
        st = client.get(f"/api/analyses/{aid}/status").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert st["status"] == "completed", st
    d = client.get(f"/api/analyses/{aid}").json()
    assert d["media_type"] == "video" and d["classification"] == "FAKE" and d["video"]["frames_sampled"] >= 3
    assert d["video"]["frames"][0]["faces"][0]["track_id"] == 0 and d["video"]["suspicious_segments"]
    assert client.get(f"/api/analyses/{aid}/report.pdf").content[:5] == b"%PDF-"
    assert not any(client.engine.store.tmp.iterdir())


def test_corrupt_video_fails_gracefully(client):
    r = client.post("/api/analyze/video", files={"file": ("clip.mp4", b"\x00garbage" * 500, "video/mp4")})
    aid = r.json()["analysis_id"]
    for _ in range(80):
        st = client.get(f"/api/analyses/{aid}/status").json()
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert st["status"] == "failed" and "video" in st["error"].lower()
    assert client.get(f"/api/analyses/{aid}").status_code == 409


def test_degraded_mode_when_model_missing(tmp_path):
    s = Settings(storage_dir=tmp_path / "st", model_path=tmp_path / "missing.pth", device="cpu")
    with TestClient(create_app(s)) as c:
        h = c.get("/api/health").json()
        assert h["status"] == "degraded" and not h["model_loaded"] and "FileNotFoundError" in h["detail"]
        assert c.post("/api/analyze/image", files={"file": ("a.png", b"x", "image/png")}).status_code == 503


def test_cors_headers(client):
    r = client.options("/api/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert client.get("/api/health").headers["x-content-type-options"] == "nosniff"


def test_public_history_switch_hides_the_global_list_but_not_individual_results(random_bundle, tmp_path):
    """Public deployments set VERIFAI_PUBLIC_HISTORY=false: nobody can enumerate other visitors' analyses, but a result is still reachable by its own id."""
    s = Settings(storage_dir=tmp_path / "st2", device="cpu", max_image_mb=1.0, public_history=False)
    eng = Engine(random_bundle, s)
    eng.detector = StubDetector([[BOX]])
    with TestClient(create_app(s, eng)) as c:
        d = c.post("/api/analyze/image", files={"file": ("a.png", png_bytes(tmp_path), "image/png")}).json()
        assert c.get("/api/analyses").json() == []
        assert c.get(f"/api/analyses/{d['analysis_id']}").json()["analysis_id"] == d["analysis_id"]
        assert c.delete(f"/api/analyses/{d['analysis_id']}").status_code == 204
    eng.shutdown()


def test_public_history_env_switch():
    assert Settings.from_env({}).public_history is True
    assert Settings.from_env({"VERIFAI_PUBLIC_HISTORY": "false"}).public_history is False


def test_batch_size_env_settings_are_memory_knobs():
    s = Settings.from_env({"VERIFAI_VIDEO_BATCH_SIZE": "2", "VERIFAI_INFER_BATCH_SIZE": "3"})
    assert s.video_batch_size == 2 and s.infer_batch_size == 3
    d = Settings.from_env({})
    assert d.video_batch_size == 16 and d.infer_batch_size == 16          # defaults unchanged
