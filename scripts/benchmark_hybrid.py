"""Time search with and without hybrid filters, and pre-filter vs naive post-filter.

    python scripts/benchmark_hybrid.py                 # real corpus (~640 patents / 32k passages)
    python scripts/benchmark_hybrid.py --scale 1 10 30 # + synthetic copies of the corpus to see scaling
    python scripts/benchmark_hybrid.py --profile large --data-dir data/patent_data_large_sample --max-paras 10

Query embeddings are computed once up front so the numbers isolate *retrieval*
cost (the ~5-15 ms query-encoding cost is constant across strategies and is
reported separately).  "--scale N" replicates every patent N times (new doc
numbers, same text/metadata, embeddings tiled + tiny noise) so filter
selectivity stays realistic while the matrix grows N-fold.
"""
from __future__ import annotations

import argparse
import copy
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.data import DEFAULT_DATA_DIR, load_patents  # noqa: E402
import datetime as dt  # noqa: E402
from patent_search.engine import Filters, PatentSearchEngine, load_model  # noqa: E402

QUERIES = [
    "spoke connecting the hub to the rim of a bicycle wheel",
    "tire tread pattern that reduces road noise",
    "trailer hitch coupling with a locking mechanism",
    "pressure sensor mounted inside a tire transmitting data wirelessly",
    "lightweight composite wheel made of carbon fiber",
    "rubber composition for a tire sidewall containing silica",
    "hub bearing assembly with a seal",
    "valve stem for inflating a tire",
]

FILTER_SETS = {
    "none": Filters(),
    "cpc=B60B (wheels, ~45%)": Filters(cpc_prefixes=["B60B"]),
    "cpc=B60D (couplings, ~3%)": Filters(cpc_prefixes=["B60D"]),
    "kw=tire": Filters(keywords=["tire"]),
    "kw=wheel AND hub": Filters(keywords=["wheel", "hub"]),
    "cpc=B60C + kw=tread": Filters(cpc_prefixes=["B60C"], keywords=["tread"]),
    "title~'wheel'": Filters(title="wheel"),
}


# general-technology queries/filters for the large (all-CPC) dataset
LARGE_QUERIES = [
    "lithium ion battery with improved thermal management",
    "neural network accelerator for matrix multiplication",
    "wireless device transmitting a beam report to a base station",
    "pharmaceutical composition for treating cancer comprising an antibody",
    "semiconductor device with a gate electrode and a dielectric layer",
    "vehicle brake control system using wheel speed sensors",
    "method for encrypting data stored in cloud storage",
    "catheter with an inflatable balloon for delivering a stent",
]
LARGE_FILTERS = {
    "none": Filters(),
    "cpc=G06F (computing, ~5%)": Filters(cpc_prefixes=["G06F"]),
    "cpc=H01M (batteries, ~1%)": Filters(cpc_prefixes=["H01M"]),
    "cpc=B60 (vehicles, ~2%)": Filters(cpc_prefixes=["B60"]),
    "cpc=H (electricity section)": Filters(cpc_prefixes=["H"]),
    "kw=battery": Filters(keywords=["battery"]),
    "kw=wireless AND network": Filters(keywords=["wireless", "network"]),
    "cpc=G06F + kw=memory": Filters(cpc_prefixes=["G06F"], keywords=["memory"]),
    "title~'vehicle'": Filters(title="vehicle"),
    "year 2015-2017": Filters(date_from=dt.date(2015, 1, 1), date_to=dt.date(2017, 12, 31)),
}


class ScaledEngine(PatentSearchEngine):
    """Engine over N copies of the corpus; reuses base embeddings instead of re-encoding."""

    def __init__(self, base: PatentSearchEngine, n: int):
        self._base_emb = base.emb
        self._n = n
        patents = []
        for c in range(n):
            for p in base.patents:
                q = copy.copy(p)
                q.doc_number = f"{p.doc_number}-{c}"
                patents.append(q)
        super().__init__(patents, base.model_name, model=base.model, max_paras=base.max_paras)

    def _load_or_encode(self, cache_dir):
        rng = np.random.default_rng(0)
        emb = np.tile(self._base_emb, (self._n, 1))
        if self._n > 1:  # break exact ties between copies
            emb[len(self._base_emb):] += rng.normal(0, 0.01, emb[len(self._base_emb):].shape).astype(np.float32)
        return emb


def timeit(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts)


def bench(engine: PatentSearchEngine, reps: int, mode: str, queries=QUERIES, filter_sets=FILTER_SETS):
    qemb = {q: engine.encode([q]) for q in queries}
    real_encode = engine.encode
    engine.encode = lambda texts, **kw: qemb[texts[0]] if texts[0] in qemb else real_encode(texts, **kw)
    rows = []
    for name, f in filter_sets.items():
        cand = engine.candidate_ids(f)
        n_cand = len(engine.patents) if cand is None else len(cand)
        pre = statistics.mean(timeit(lambda q=q: engine.search(q, filters=f, mode=mode, strategy="prefilter"), reps)
                              for q in queries)
        post = statistics.mean(timeit(lambda q=q: engine.search(q, filters=f, mode=mode, strategy="postfilter"), reps)
                               for q in queries)
        # correctness: both strategies must return the same top-10
        same = all([r.doc_number for r in engine.search(q, filters=f, mode=mode, strategy="prefilter")] ==
                   [r.doc_number for r in engine.search(q, filters=f, mode=mode, strategy="postfilter")]
                   for q in queries)
        rows.append((name, n_cand, pre, post, same))
    engine.encode = real_encode
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "rrf"])
    ap.add_argument("--profile", default="small", choices=["small", "large"])
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-paras", type=int, default=None)
    a = ap.parse_args()
    queries, filter_sets = (LARGE_QUERIES, LARGE_FILTERS) if a.profile == "large" else (QUERIES, FILTER_SETS)

    t = time.perf_counter()
    patents = load_patents(a.data_dir)
    load_s = time.perf_counter() - t
    base = PatentSearchEngine(patents, model=load_model(), max_paras=a.max_paras)
    print(f"load+clean {len(patents):,} patents: {load_s:.1f}s; index build: "
          + ", ".join(f"{k} {v:.1f}s" for k, v in base.timings.items()))
    enc = timeit(lambda: base.encode([queries[0]]), 20)
    print(f"query encoding (constant cost, excluded below): {enc:.1f} ms\n")

    for n in a.scale:
        eng = base if n == 1 else ScaledEngine(base, n)
        print(f"### scale x{n}: {len(eng.patents):,} patents, {len(eng.passages):,} passages, mode={a.mode}")
        print(f"| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |")
        print(f"|---|---:|---:|---:|---:|:---:|")
        for name, n_cand, pre, post, same in bench(eng, a.reps, a.mode, queries, filter_sets):
            print(f"| {name} | {n_cand:,} | {pre:.2f} | {post:.2f} | {post / pre:.1f}x | {'yes' if same else 'NO'} |")
        print()


if __name__ == "__main__":
    main()
