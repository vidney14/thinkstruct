"""BM25 as a precomputed sparse matrix.

Same scoring as rank_bm25.BM25Okapi (k1=1.5, b=0.75, epsilon=0.25 floor for
negative idf), but the per-(doc, term) BM25 weight is computed once at build
time into a CSC matrix. A query is then a sum of a few matrix columns
(O(postings of the query terms)) instead of a Python loop over every document
per query term. Tokenisation (CountVectorizer, regex [a-z0-9]+ on lowercase)
matches data.tokenize.

Measured on 25k patents (scripts/bench_bm25.py): same build time (~4.5 s), query
0.15 ms vs 26.7 ms for rank_bm25 (~180x), identical top-10.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .data import tokenize


class SparseBM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        cv = CountVectorizer(lowercase=True, token_pattern=r"[a-z0-9]+", dtype=np.float32)
        tf = cv.fit_transform(docs).tocsr()  # (n_docs, vocab) raw term counts
        self.vocab = cv.vocabulary_
        n = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean() if n else 1.0
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(n - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = epsilon * idf.mean()
        # weight = idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)), applied to the nonzeros only
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        t = tf.data
        tf.data = (idf[tf.indices] * t * (k1 + 1) / (t + k1 * (1 - b + b * dl[rows] / avgdl))).astype(np.float32)
        self.W = tf.tocsc()
        self.n = n

    def scores(self, query: str, cand: np.ndarray | None = None) -> np.ndarray:
        ids, counts = np.unique([self.vocab[t] for t in tokenize(query) if t in self.vocab], return_counts=True)
        if len(ids) == 0:
            s = np.zeros(self.n, dtype=np.float32)
        else:
            s = np.asarray(self.W[:, ids] @ counts.astype(np.float32)).ravel()
        return s if cand is None else s[cand]
