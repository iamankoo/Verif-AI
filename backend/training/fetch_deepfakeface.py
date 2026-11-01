"""Fetch a seeded, identity-disjoint SAMPLE of the DeepFakeFace dataset (openrail licence).

Dataset: https://huggingface.co/datasets/OpenRL/DeepFakeFace  (Song et al., arXiv:2309.02218)
  wiki.zip       real images (subset of IMDB-WIKI)
  inpainting.zip Stable Diffusion inpainting fakes
  insight.zip    InsightFace (face-swap) fakes
  text2img.zip   Stable Diffusion v1.5 text-to-image fakes
Every zip keeps the same relative paths, so one real image and its fake counterparts share a name.

Split design (all splits are IDENTITY-DISJOINT; identity = numeric prefix of the file name):
  train / val / test_internal : generators {inpainting, insight}   (seen generators)
  test_crossgen               : generator  text2img                (UNSEEN generator, unseen identities)
Per sampled image path we keep the real image and ONE fake counterpart => classes are balanced.

Only the sampled members are downloaded (HTTP range requests); nothing else is fetched.

Usage (from backend/):  python -m training.fetch_deepfakeface --out data/raw/dff --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .remote_zip import open_remote_zip

BASE = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
SEEN_GENS = ("inpainting", "insight")
QUOTAS = {"train": 1200, "val": 250, "test_internal": 300, "test_crossgen": 300}   # image paths per split
MAX_PATHS_PER_IDENTITY = 2
_tl = threading.local()


def identity_of(rel: str) -> str:
    return os.path.basename(rel).split("_", 1)[0]


def list_members(zip_name: str, cache_dir: str):
    cache = os.path.join(cache_dir, "zip_index.json")
    if os.path.isfile(cache):
        idx = json.load(open(cache))
        if zip_name in idx:
            return [n for n, _, _ in idx[zip_name]]
    zf = open_remote_zip(BASE + zip_name + ".zip")
    return [i.filename for i in zf.infolist() if not i.is_dir()]


def sample_plan(wiki_members, seed: int):
    """Return rows [(split, rel_path, identity)] with identity-disjoint splits."""
    by_id = defaultdict(list)
    for m in wiki_members:
        rel = m.split("/", 1)[1]
        by_id[identity_of(rel)].append(rel)
    ids = sorted(by_id)
    rng = random.Random(seed)
    rng.shuffle(ids)
    plan, cursor = [], 0
    for split, quota in QUOTAS.items():
        got = 0
        while got < quota and cursor < len(ids):
            ident = ids[cursor]; cursor += 1
            rels = sorted(by_id[ident]); rng.shuffle(rels)
            for rel in rels[:MAX_PATHS_PER_IDENTITY]:
                if got >= quota:
                    break
                plan.append((split, rel, ident)); got += 1
    return plan


def fake_generator(split: str, rel: str) -> str:
    if split == "test_crossgen":
        return "text2img"
    return SEEN_GENS[int(hashlib.md5(rel.encode()).hexdigest(), 16) % 2]


def _zip(zip_name):
    cache = getattr(_tl, "zips", None)
    if cache is None:
        cache = _tl.zips = {}
    if zip_name not in cache:
        cache[zip_name] = open_remote_zip(BASE + zip_name + ".zip", block=96 * 1024)
    return cache[zip_name]


def download(job):
    zip_name, member, dest = job
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest, None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = _zip(zip_name).read(member)
    with open(dest, "wb") as f:
        f.write(data)
    return dest, hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/raw/dff")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--manifest", default="manifests/dff_raw_manifest.csv")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    wiki = list_members("wiki", a.out)
    plan = sample_plan(wiki, a.seed)
    jobs, rows = [], []
    for split, rel, ident in plan:
        gen = fake_generator(split, rel)
        for label, zname in ((0, "wiki"), (1, gen)):
            member = f"{zname}/{rel}"
            dest = os.path.join(a.out, split, "real" if label == 0 else "fake", f"{zname}__{rel.replace('/', '_')}")
            jobs.append((zname, member, dest))
            rows.append(dict(split=split, label=label, generator="real" if label == 0 else gen, identity=ident,
                             pair_id=rel, source_zip=zname + ".zip", member=member, raw_path=dest.replace("\\", "/")))
    print(f"planned {len(plan)} image paths -> {len(jobs)} files; identities per split:",
          {s: len({r['identity'] for r in rows if r['split'] == s}) for s in QUOTAS})
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for dest, sha in ex.map(download, jobs):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    for r in rows:
        r["sha256"] = hashlib.sha256(open(r["raw_path"], "rb").read()).hexdigest()
        r["bytes"] = os.path.getsize(r["raw_path"])
    os.makedirs(os.path.dirname(a.manifest), exist_ok=True)
    with open(a.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("manifest ->", a.manifest, "| files:", len(rows))


if __name__ == "__main__":
    main()
