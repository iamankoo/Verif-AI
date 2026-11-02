"""Regression: the browser polls /status while the video worker saves progress. A reader must never see an empty/partial job file."""
import threading

from verifai.schemas import JobState, JobStatus
from verifai.store import AnalysisStore


def test_concurrent_job_save_and_load_never_sees_a_partial_file(tmp_path):
    store = AnalysisStore(tmp_path / "st")
    aid = "a" * 32
    job = JobState(analysis_id=aid, status=JobStatus.RUNNING, media_type="video", filename="v.mp4", stage="starting", progress=0.0)
    store.save_job(job)
    stop = threading.Event()
    errors = []

    def writer():
        i = 0
        while not stop.is_set():
            job.progress = (i % 100) / 100.0
            store.save_job(job)
            i += 1

    t = threading.Thread(target=writer)
    t.start()
    try:
        for _ in range(3000):
            try:
                store.load_job(aid)
            except Exception as exc:          # any parse error means the reader saw a half-written file
                errors.append(repr(exc))
                break
    finally:
        stop.set()
        t.join()
    assert not errors, errors
    assert not list((tmp_path / "st").rglob("*.tmp")), "temporary files must not be left behind"
