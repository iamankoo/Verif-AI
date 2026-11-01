"""End-to-end smoke test against a REAL running server with REAL model weights.

  python tools/e2e_smoke.py --manifest manifests/dff_faces_manifest.csv --out runs/e2e/e2e_report.json
  python tools/e2e_smoke.py --base-url http://localhost:8000 ...        # test an already running server

Media (all from the openly licensed DeepFakeFace sample, never private files):
  * known-real image  : a real photograph from split `test_internal`
  * known-fake image  : its generated counterpart (inpainting / insight)
  * unseen-generator  : a text2img fake from `test_crossgen`
  * "videos"          : SLIDESHOW videos assembled from distinct photographs (real-only / fake-only). They exercise the
                        video pipeline (decode, sampling, tracking, aggregation, report) but are NOT real deepfake videos
                        and say nothing about video-level accuracy.
This script verifies that the workflow runs end to end and records what the model returned; it does not by itself
establish accuracy (see docs/MODEL_EVALUATION_REPORT.md for evaluation on held-out splits).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parent.parent


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, data=None, headers=None, timeout=300):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def multipart(path: Path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def pick(rows, split, label, generator=None, n=1, skip=0):
    sel = [r for r in rows if r["split"] == split and r["label"] == str(label) and (generator is None or r["generator"] == generator)]
    return sel[skip:skip + n]


def slideshow(paths, out: Path, fps=2, seconds_each=2, size=(640, 640)):
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for p in paths:
        im = cv2.resize(cv2.imread(str(p)), size)
        for _ in range(fps * seconds_each):
            vw.write(im)
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(BACKEND / "manifests/dff_faces_manifest.csv"))
    ap.add_argument("--base-url")
    ap.add_argument("--model", default=None, help="VERIFAI_MODEL_PATH for the spawned server")
    ap.add_argument("--out", default=str(BACKEND / "runs/e2e/e2e_report.json"))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.manifest, encoding="utf-8")))
    work = Path(tempfile.mkdtemp(prefix="verifai_e2e_"))
    proc, base = None, a.base_url
    if not base:
        port = free_port()
        env = dict(os.environ, VERIFAI_STORAGE_DIR=str(work / "storage"), VERIFAI_DEVICE="auto")
        if a.model:
            env["VERIFAI_MODEL_PATH"] = a.model
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "verifai.api.app:app", "--port", str(port), "--log-level", "warning"], cwd=BACKEND, env=env)
        base = f"http://127.0.0.1:{port}"
    report = {"base_url": base, "checks": [], "analyses": []}

    def check(name, ok, detail=""):
        report["checks"].append({"check": name, "passed": bool(ok), "detail": str(detail)[:300]})
        print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
        return ok

    try:
        for _ in range(120):
            try:
                st, body, _ = http("GET", base + "/api/health", timeout=5)
                if st == 200 and json.loads(body).get("model_loaded"):
                    break
            except Exception:
                pass
            time.sleep(1)
        h = json.loads(http("GET", base + "/api/health")[1])
        check("health: model loaded", h["status"] == "ok" and h["model_loaded"], h)
        m = json.loads(http("GET", base + "/api/model")[1])
        report["model"] = m["model"]
        check("model info available", m["model"]["parameters"] > 0, f'{m["model"]["name"]} v{m["model"]["version"]} on {m["model"]["device"]}')

        def run_image(tag, row):
            p = Path(row["raw_path"]) if os.path.isabs(row["raw_path"]) else BACKEND / row["raw_path"]
            body, hd = multipart(p)
            t = time.time()
            st, resp, _ = http("POST", base + "/api/analyze/image", body, hd)
            dt = time.time() - t
            ok = st == 200
            d = json.loads(resp) if ok else {}
            check(f"image [{tag}] analysed", ok, f"HTTP {st}, {dt:.1f}s")
            if ok:
                report["analyses"].append({"tag": tag, "ground_truth": "fake" if row["label"] == "1" else "real", "generator": row["generator"], "media": "image",
                                           "classification": d["classification"], "fake_probability": d["fake_probability"], "faces": d["faces_detected"],
                                           "seconds": round(dt, 2), "id": d["analysis_id"]})
                art = d.get("annotated_artifact")
                if art:
                    check(f"image [{tag}] evidence image served", http("GET", f'{base}/api/analyses/{d["analysis_id"]}/artifacts/{art}')[0] == 200)
                if d["faces"] and d["faces"][0].get("heatmap_artifact"):
                    check(f"image [{tag}] Grad-CAM heat-map served", http("GET", f'{base}/api/analyses/{d["analysis_id"]}/artifacts/{d["faces"][0]["heatmap_artifact"]}')[0] == 200)
                st, pdf, _ = http("GET", f'{base}/api/analyses/{d["analysis_id"]}/report.pdf')
                check(f"image [{tag}] PDF report", st == 200 and pdf[:5] == b"%PDF-" and len(pdf) > 4000, f"{len(pdf)} bytes")
            return d

        for tag, row in (("known real photo", pick(rows, "test_internal", 0)[0]), ("known fake (seen generator)", pick(rows, "test_internal", 1)[0]),
                         ("known fake (UNSEEN generator text2img)", pick(rows, "test_crossgen", 1, "text2img")[0])):
            run_image(tag, row)

        # invalid inputs against the live server
        st, _, _ = http("POST", base + "/api/analyze/image", *multipart(Path(__file__)))
        check("invalid upload (.py) rejected with 415", st == 415, st)
        bad = work / "broken.png"; bad.write_bytes(b"nope")
        st, _, _ = http("POST", base + "/api/analyze/image", *multipart(bad))
        check("corrupt image rejected with 422", st == 422, st)
        blank = work / "blank.png"; cv2.imwrite(str(blank), (cv2.UMat(480, 640, cv2.CV_8UC3, (120, 120, 120))).get())
        st, resp, _ = http("POST", base + "/api/analyze/image", *multipart(blank))
        check("image without a face -> INCONCLUSIVE (not an error)", st == 200 and json.loads(resp)["classification"] == "INCONCLUSIVE", st)

        # slideshow videos
        def run_video(tag, imgs, truth):
            v = work / f"{tag.replace(' ', '_')}.mp4"
            slideshow(imgs, v)
            body, hd = multipart(v)
            t = time.time()
            st, resp, _ = http("POST", base + "/api/analyze/video", body, hd)
            check(f"video [{tag}] accepted (202)", st == 202, st)
            if st != 202:
                return
            aid = json.loads(resp)["analysis_id"]
            seen = []
            for _ in range(600):
                s = json.loads(http("GET", f"{base}/api/analyses/{aid}/status")[1])
                seen.append(s["progress"])
                if s["status"] in ("completed", "failed"):
                    break
                time.sleep(1)
            dt = time.time() - t
            check(f"video [{tag}] completed", s["status"] == "completed", f'{s["status"]} in {dt:.1f}s {s.get("error") or ""}')
            if s["status"] != "completed":
                return
            d = json.loads(http("GET", f"{base}/api/analyses/{aid}")[1])
            vs = d["video"]
            check(f"video [{tag}] analysed many frames (not only frame 0)", vs["frames_sampled"] >= 5 and len(vs["frames"]) >= 5, f'{vs["frames_sampled"]} sampled, {vs["frames_with_faces"]} with faces')
            report["analyses"].append({"tag": tag, "ground_truth": truth, "media": "SLIDESHOW video (functional test only)", "classification": d["classification"],
                                       "fake_probability": d["fake_probability"], "frames_sampled": vs["frames_sampled"], "frames_with_faces": vs["frames_with_faces"],
                                       "tracks": len(vs["tracks"]), "suspicious_segments": len(vs["suspicious_segments"]), "seconds": round(dt, 1), "id": aid,
                                       "frame_scores": [round(f["frame_fake_probability"], 4) for f in vs["frames"]]})
            for e in vs["evidence_frames"][:1]:
                check(f"video [{tag}] evidence frame served", http("GET", f'{base}/api/analyses/{aid}/artifacts/{e["frame_artifact"]}')[0] == 200)
            st, pdf, _ = http("GET", f"{base}/api/analyses/{aid}/report.pdf")
            check(f"video [{tag}] PDF report", st == 200 and pdf[:5] == b"%PDF-" and len(pdf) > 6000, f"{len(pdf)} bytes")
            (work / f"{aid}.pdf").write_bytes(pdf)

        reals = [Path(r["raw_path"]) if os.path.isabs(r["raw_path"]) else BACKEND / r["raw_path"] for r in pick(rows, "test_internal", 0, n=6, skip=1)]
        fakes = [Path(r["raw_path"]) if os.path.isabs(r["raw_path"]) else BACKEND / r["raw_path"] for r in pick(rows, "test_internal", 1, n=6, skip=1)]
        run_video("real photos slideshow", reals, "real photographs (not a real video)")
        run_video("fake faces slideshow", fakes, "generated faces (not a real deepfake video)")
        ids = [x["id"] for x in report["analyses"]]
        d = http("DELETE", f"{base}/api/analyses/{ids[0]}")[0]
        check("delete analysis (204) then 404", d == 204 and http("GET", f"{base}/api/analyses/{ids[0]}")[0] == 404, d)
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(15)
            except Exception:
                proc.kill()
    failed = [c for c in report["checks"] if not c["passed"]]
    report["summary"] = {"checks": len(report["checks"]), "passed": len(report["checks"]) - len(failed), "failed": len(failed)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"]))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
