"""rank_bm25 (pure-Python loop) vs. SparseBM25 (precomputed sparse matrix) on the same docs.

    python scripts/bench_bm25.py --data-dir data/patent_data_large_sample
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.bm25 import SparseBM25  # noqa: E402
from patent_search.data import DEFAULT_DATA_DIR, load_patents, tokenize  # noqa: E402

QUERIES = ["lithium ion battery thermal management", "wireless beam report base station",
           "antibody composition for treating cancer", "gate electrode dielectric layer semiconductor",
           "vehicle brake control wheel speed sensor"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    a = ap.parse_args()
    ps = load_patents(a.data_dir)
    docs = [f"{p.title} {p.abstract} {' '.join(p.claims)}" for p in ps]
    from rank_bm25 import BM25Okapi
    t = time.perf_counter(); old = BM25Okapi([tokenize(d) for d in docs]); b_old = time.perf_counter() - t
    t = time.perf_counter(); new = SparseBM25(docs); b_new = time.perf_counter() - t
    q_old, q_new, same = [], [], True
    for q in QUERIES:
        t = time.perf_counter(); x = old.get_scores(tokenize(q)); q_old.append(time.perf_counter() - t)
        t = time.perf_counter(); y = new.scores(q); q_new.append(time.perf_counter() - t)
        same &= list(np.argsort(-x)[:10]) == list(np.argsort(-y)[:10])
    print(f"| BM25 over {len(ps):,} patents | build s | query ms (median) |\n|---|---:|---:|")
    print(f"| rank_bm25 | {b_old:.1f} | {1000 * statistics.median(q_old):.1f} |")
    print(f"| SparseBM25 | {b_new:.1f} | {1000 * statistics.median(q_new):.2f} |")
    print(f"identical top-10 on all queries: {same}")


if __name__ == "__main__":
    main()
