"""Scan the large dataset (coding_challenge_data, ~471k records / 30 GB) in parallel,
clean + de-duplicate it, and write a deterministic random sample in the same
JSON layout so every other tool can use it via --data-dir.

    python scripts/prepare_large.py ~/Downloads/coding_challenge_data --frac 0.055
    python scripts/prepare_large.py ~/Downloads/coding_challenge_data --frac 0.055 --workers 1   # serial baseline

Sampling is by a hash of doc_number, so (a) it is reproducible, (b) all copies of
a duplicated patent fall on the same side of the sample, and (c) a worker can
drop non-sampled records before the expensive parsing/cleaning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.data import dedupe, find_files, load_file, new_stats  # noqa: E402


def in_sample(doc: str, frac: float) -> bool:
    return int(hashlib.md5(doc.encode()).hexdigest()[:8], 16) < frac * 2**32


def scan(args):
    path, frac = args
    t = time.perf_counter()
    stats = new_stats()
    ps = load_file(path, stats, keep=lambda d: in_sample(d, frac))
    return path, ps, stats, time.perf_counter() - t


def to_record(p) -> dict:
    return {"title": p.title, "doc_number": p.doc_number, "filename": p.filename, "abstract": p.abstract,
            "detailed_description": p.description, "claims": p.claims,
            "classification": p.classification or "N/A"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--frac", type=float, default=0.055)
    ap.add_argument("--out", default="data/patent_data_large_sample")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--per-file", type=int, default=5000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    files = find_files(os.path.expanduser(a.src))
    print(f"{len(files)} files, sampling {a.frac:.1%} with {a.workers} worker(s)")
    t = time.perf_counter()
    stats, raw, per_file = Counter(), [], []
    if a.workers == 1:
        results = map(scan, [(f, a.frac) for f in files])
    else:
        ex = ProcessPoolExecutor(a.workers)
        results = ex.map(scan, [(f, a.frac) for f in files], chunksize=2)
    for path, ps, st, dt in results:
        stats.update(st)
        raw.extend(ps)
        per_file.append(dt)
    scan_s = time.perf_counter() - t

    # duplicates: identical copies vs. differing copies
    by_doc: dict[str, list] = {}
    for p in raw:
        by_doc.setdefault(p.doc_number, []).append(p)
    dups = {d: v for d, v in by_doc.items() if len(v) > 1}
    differing = sum(1 for v in dups.values()
                    if len({json.dumps(to_record(x), sort_keys=True) for x in v}) > 1)
    st = dict(stats)
    patents = dedupe(raw, st)

    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):
        os.remove(os.path.join(a.out, f))
    for i in range(0, len(patents), a.per_file):
        with open(os.path.join(a.out, f"patents_batch_{i // a.per_file:04d}.json"), "w") as fh:
            json.dump([to_record(p) for p in patents[i:i + a.per_file]], fh)
    total_s = time.perf_counter() - t

    cls = Counter(p.cpc[:4] or "(none)" for p in patents)
    report = {
        "files": len(files), "workers": a.workers, "scan_seconds": round(scan_s, 1),
        "total_seconds": round(total_s, 1), "mean_file_seconds": round(sum(per_file) / len(per_file), 2),
        "sampled_records": len(raw), "unique_patents": len(patents),
        "duplicate_doc_numbers": len(dups), "duplicates_with_differing_content": differing,
        "stats": st, "top_cpc4": cls.most_common(10),
        "passages_if_all_paras": sum(1 + len(p.claims) + len(p.description) for p in patents),
        "passages_if_10_paras": sum(1 + len(p.claims) + min(10, len(p.description)) for p in patents),
    }
    print(json.dumps(report, indent=2))
    os.makedirs("results", exist_ok=True)
    with open(f"results/prepare_large_w{a.workers}.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
