"""Part 3: evaluate base model -> mine training triplets -> fine-tune -> re-evaluate.

    python scripts/train_eval.py                     # defaults: MiniLM, 1 epoch
    python scripts/train_eval.py --epochs 2 --out models/minilm-patents

Writes results/eval_results.json and prints a markdown table. The fine-tuned
model can then be used by the search engine:  python -m patent_search.cli --model models/minilm-patents "..."
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from patent_search.data import load_patents  # noqa: E402
from patent_search.engine import DEFAULT_MODEL, load_model  # noqa: E402
from patent_search import training as T  # noqa: E402


def row(name, m):
    a, i, d = m["all"], m["independent"], m["dependent"]
    return (f"| {name} | {a['recall@1']:.3f} | {a['recall@10']:.3f} | {a['mrr@10']:.3f} "
            f"| {i['mrr@10']:.3f} | {d['mrr@10']:.3f} |")


def prow(name, p):
    return f"| {name} | {p['cos_pos']:.3f} | {p['cos_neg']:.3f} | {p['margin']:.3f} | {p['triplet_acc']:.3f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="models/minilm-patents")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(a.seed); np.random.seed(a.seed)
    import torch; torch.manual_seed(a.seed)

    patents = load_patents()
    train_ids, test_ids = T.split_patents(patents, seed=a.seed)
    corpus = T.Corpus(patents)
    test_q = T.make_queries(patents, test_ids, max_dependent=2, seed=a.seed)
    train_q = T.make_queries(patents, train_ids, max_dependent=4, seed=a.seed)
    print(f"{len(patents)} patents ({len(train_ids)} train / {len(test_ids)} test), "
          f"{len(corpus.texts)} claim-free passages, {len(train_q)} train queries, {len(test_q)} test queries")

    base = load_model(a.model)
    t = time.perf_counter()
    base_emb = T.encode(base, corpus.texts)
    print(f"encoded corpus in {time.perf_counter() - t:.1f}s")

    bm25_m, _ = T.bm25_metrics(corpus, test_q)
    base_m, base_ranks = T.retrieval_metrics(base, corpus, test_q, base_emb)
    # held-out triplets for pair metrics are mined with the BASE model and frozen
    test_trip = T.mine_triplets(base, patents, corpus, test_q, base_emb)
    base_p = T.pair_metrics(base, test_trip)

    train_trip = T.mine_triplets(base, patents, corpus, train_q, base_emb)
    print(f"mined {len(train_trip)} training triplets; example:\n  A: {train_trip[0][0][:150]}\n"
          f"  P: {train_trip[0][1][:150]}\n  N: {train_trip[0][2][:150]}")
    del base_emb

    t = time.perf_counter()
    tuned = T.train(load_model(a.model), train_trip, a.out, epochs=a.epochs,
                    batch_size=a.batch_size, lr=a.lr, seed=a.seed)
    train_s = time.perf_counter() - t
    tuned_m, tuned_ranks = T.retrieval_metrics(tuned, corpus, test_q)
    tuned_p = T.pair_metrics(tuned, test_trip)

    improved, worse = int((tuned_ranks < base_ranks).sum()), int((tuned_ranks > base_ranks).sum())
    lines = [
        f"Test set: {len(test_q)} claims from {len(test_ids)} held-out patents, "
        f"searched against all {len(patents)} patents (claims removed from index).",
        "", "| model | R@1 | R@10 | MRR@10 | MRR@10 indep. | MRR@10 dep. |", "|---|---:|---:|---:|---:|---:|",
        row("BM25 (reference)", bm25_m), row(f"{a.model.split('/')[-1]} (base)", base_m),
        row(f"fine-tuned ({a.epochs} ep, {len(train_trip)} triplets)", tuned_m),
        "", "| model | cos(claim, pos) | cos(claim, hard neg) | margin | triplet acc |", "|---|---:|---:|---:|---:|",
        prow("base", base_p), prow("fine-tuned", tuned_p),
        "", f"Per-query: {improved} queries ranked their patent higher after tuning, {worse} lower. "
            f"Training took {train_s:.0f}s.",
    ]
    print("\n" + "\n".join(lines))
    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w") as fh:
        json.dump({"config": vars(a), "n_train_triplets": len(train_trip), "n_test_queries": len(test_q),
                   "train_seconds": train_s, "bm25": bm25_m, "base": base_m, "tuned": tuned_m,
                   "pair_base": base_p, "pair_tuned": tuned_p,
                   "per_query": {"improved": improved, "worse": worse}}, fh, indent=2)
    with open("results/eval_results.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
