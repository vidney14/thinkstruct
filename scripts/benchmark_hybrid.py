"""Time search with and without hybrid filters, and pre-filter vs naive post-filter.

    python scripts/benchmark_hybrid.py                 # real corpus (~640 patents / 32k passages)
    python scripts/benchmark_hybrid.py --scale 1 10 30 # + synthetic copies of the corpus to see scaling

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
from patent_search.data import load_patents  # noqa: E402
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
        super().__init__(patents, base.model_name, model=base.model)

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


def bench(engine: PatentSearchEngine, reps: int, mode: str):
    qemb = {q: engine.encode([q]) for q in QUERIES}
    real_encode = engine.encode
    engine.encode = lambda texts, **kw: qemb[texts[0]] if texts[0] in qemb else real_encode(texts, **kw)
    rows = []
    for name, f in FILTER_SETS.items():
        cand = engine.candidate_ids(f)
        n_cand = len(engine.patents) if cand is None else len(cand)
        pre = statistics.mean(timeit(lambda q=q: engine.search(q, filters=f, mode=mode, strategy="prefilter"), reps)
                              for q in QUERIES)
        post = statistics.mean(timeit(lambda q=q: engine.search(q, filters=f, mode=mode, strategy="postfilter"), reps)
                               for q in QUERIES)
        # correctness: both strategies must return the same top-10
        same = all([r.doc_number for r in engine.search(q, filters=f, mode=mode, strategy="prefilter")] ==
                   [r.doc_number for r in engine.search(q, filters=f, mode=mode, strategy="postfilter")]
                   for q in QUERIES)
        rows.append((name, n_cand, pre, post, same))
    engine.encode = real_encode
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, nargs="+", default=[1])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "rrf"])
    a = ap.parse_args()

    t = time.perf_counter()
    base = PatentSearchEngine(load_patents(), model=load_model())
    print(f"index build (incl. cached embeddings): {time.perf_counter() - t:.1f}s")
    enc = timeit(lambda: base.encode([QUERIES[0]]), 20)
    print(f"query encoding (constant cost, excluded below): {enc:.1f} ms\n")

    for n in a.scale:
        eng = base if n == 1 else ScaledEngine(base, n)
        print(f"### scale x{n}: {len(eng.patents):,} patents, {len(eng.passages):,} passages, mode={a.mode}")
        print(f"| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |")
        print(f"|---|---:|---:|---:|---:|:---:|")
        for name, n_cand, pre, post, same in bench(eng, a.reps, a.mode):
            print(f"| {name} | {n_cand:,} | {pre:.2f} | {post:.2f} | {post / pre:.1f}x | {'yes' if same else 'NO'} |")
        print()


if __name__ == "__main__":
    main()
