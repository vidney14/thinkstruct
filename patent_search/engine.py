"""Part 1 search engine: dense passage search + BM25 + hybrid metadata filters.

Unit of retrieval = *passage* (title+abstract, each claim, each description
paragraph). Passages of a patent are stored contiguously, so a patent's score
is max() over its passage scores (np.maximum.reduceat), and we can hand back
the best-matching claim/paragraph as the "why it matched" snippet.

Hybrid filters (CPC prefix, keywords in title/abstract, title) are resolved
*before* any vector math using in-memory inverted indexes, so a selective
filter shrinks the matrix multiply instead of adding work after it
("pre-filtering"). The naive "post-filter" path (score everything, then scan
results with string checks) is kept for the benchmark in scripts/benchmark_hybrid.py.
"""
from __future__ import annotations

import bisect
import datetime as dt
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

from .bm25 import SparseBM25
from .data import Patent, load_patents, normalize_cpc, tokenize, DEFAULT_DATA_DIR

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MIN_PARAGRAPH_CHARS = 40
# above this candidate fraction, scoring the full matrix and masking is cheaper than
# gathering the candidate rows (measured with scripts/benchmark_hybrid.py)
GATHER_MAX_FRACTION = 0.10


def trigrams(s: str) -> set[str]:
    s = s.lower()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name_or_path: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name_or_path, device=pick_device())
    m.max_seq_length = 256
    return m


@dataclass
class Result:
    doc_number: str
    title: str
    classification: str
    week: str
    score: float
    match_kind: str  # abstract | claim N | paragraph N
    match_text: str
    matched_keywords: list[str]


@dataclass
class Filters:
    cpc_prefixes: list[str] | None = None        # e.g. ["B60B"] -> classification startswith any
    keywords: list[str] | None = None            # must appear in title/abstract
    keyword_mode: str = "all"                    # all | any
    title: str | None = None                     # substring of title (case-insens.)
    title_exact: bool = False
    date_from: dt.date | None = None             # filing week window
    date_to: dt.date | None = None

    def active(self) -> bool:
        return bool(self.cpc_prefixes or self.keywords or self.title or self.date_from or self.date_to)


class PatentSearchEngine:
    def __init__(self, patents: list[Patent], model_name: str = DEFAULT_MODEL,
                 cache_dir: str = "index_cache", model=None, max_paras: int | None = None):
        """max_paras: index only the first N description paragraphs per patent (None = all).
        At scale this bounds passages/patent (~130 -> ~30 with N=10) and therefore
        embedding time and memory; claims and the abstract are always indexed."""
        self.patents = patents
        self.max_paras = max_paras
        self.timings: dict[str, float] = {}
        self.model_name = model_name
        self.model = model or load_model(model_name)
        self.by_id = {p.doc_number: i for i, p in enumerate(patents)}
        for name, fn in (("passages", self._build_passages), ("metadata_indexes", self._build_metadata_indexes),
                         ("bm25", self._build_bm25)):
            t = time.perf_counter()
            fn()
            self.timings[name] = time.perf_counter() - t
        t = time.perf_counter()
        self.emb = self._load_or_encode(cache_dir)
        self.timings["embeddings"] = time.perf_counter() - t
        log.info("index build timings (s): %s", {k: round(v, 2) for k, v in self.timings.items()})

    # ------------------------------------------------------------------ build
    def _build_passages(self):
        texts, kinds, owner = [], [], []
        starts = np.zeros(len(self.patents), dtype=np.int64)
        for i, p in enumerate(self.patents):
            starts[i] = len(texts)
            head = f"{p.title}. {p.abstract}".strip(". ")
            if head:
                texts.append(head); kinds.append("abstract"); owner.append(i)
            for j, c in enumerate(p.claims):
                texts.append(c); kinds.append(f"claim {j + 1}"); owner.append(i)
            for j, para in enumerate(p.description[: self.max_paras]):
                if len(para) >= MIN_PARAGRAPH_CHARS:
                    texts.append(para); kinds.append(f"paragraph {j + 1}"); owner.append(i)
        self.passages, self.kinds = texts, kinds
        self.owner = np.asarray(owner, dtype=np.int64)
        self.starts = starts
        self.ends = np.append(starts[1:], len(texts))

    def _build_metadata_indexes(self):
        # token -> sorted np.array of patent ids, separately for title and title+abstract
        title_inv, ta_inv = defaultdict(set), defaultdict(set)
        for i, p in enumerate(self.patents):
            tt = tokenize(p.title)
            for t in tt:
                title_inv[t].add(i)
            for t in tt + tokenize(p.abstract):
                ta_inv[t].add(i)
        self.title_inv = {k: np.fromiter(sorted(v), np.int64) for k, v in title_inv.items()}
        # character-trigram index for substring title search (same idea as Postgres pg_trgm)
        tri = defaultdict(set)
        for i, p in enumerate(self.patents):
            for g in trigrams(p.title):
                tri[g].add(i)
        self.title_tri = {k: np.fromiter(sorted(v), np.int64) for k, v in tri.items()}
        self.titles_lower = [p.title.lower() for p in self.patents]
        self.ta_inv = {k: np.fromiter(sorted(v), np.int64) for k, v in ta_inv.items()}
        # sorted (cpc, id) list -> prefix range lookup with bisect, O(log n + hits)
        self.cpc_sorted = sorted((p.cpc, i) for i, p in enumerate(self.patents))
        self.cpc_keys = [c for c, _ in self.cpc_sorted]
        self.weeks = np.array([p.date_ordinal for p in self.patents])

    def _build_bm25(self):
        self.bm25 = SparseBM25([f"{p.title} {p.abstract} {' '.join(p.claims)}" for p in self.patents])

    def _load_or_encode(self, cache_dir: str) -> np.ndarray:
        key = hashlib.md5(json.dumps([self.model_name, len(self.passages),
                                      self.passages[:3], self.passages[-3:]]).encode()).hexdigest()[:12]
        path = os.path.join(cache_dir, f"emb_{key}.npy")
        if os.path.exists(path):
            log.info("Loading cached embeddings %s", path)
            return np.load(path)
        t = time.perf_counter()
        emb = self.encode(self.passages, show_progress_bar=True)
        log.info("Encoded %d passages in %.1fs", len(self.passages), time.perf_counter() - t)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, emb)
        return emb

    def encode(self, texts, **kw) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True,
                                 convert_to_numpy=True, **kw).astype(np.float32)

    # ---------------------------------------------------------------- filters
    def candidate_ids(self, f: Filters) -> np.ndarray | None:
        """Resolve filters to a sorted array of patent ids using the indexes. None = no filter."""
        if not f.active():
            return None
        cand: np.ndarray | None = None

        def inter(a, b):
            return b if a is None else np.intersect1d(a, b, assume_unique=True)

        if f.cpc_prefixes:
            hits = []
            for pre in f.cpc_prefixes:
                pre = normalize_cpc(pre)
                lo = bisect.bisect_left(self.cpc_keys, pre)
                hi = bisect.bisect_left(self.cpc_keys, pre + "￿")
                hits.extend(i for _, i in self.cpc_sorted[lo:hi])
            cand = inter(cand, np.unique(np.asarray(hits, dtype=np.int64)))
        if f.keywords:
            sets = []
            for kw in f.keywords:
                toks = tokenize(kw)
                s = None
                for t in toks:  # multi-word keyword: all its tokens must be present ...
                    s = inter(s, self.ta_inv.get(t, np.empty(0, np.int64)))
                if s is not None and len(toks) > 1:  # ... then verify the phrase on the survivors
                    s = np.array([i for i in s if kw.lower() in
                                  f"{self.patents[i].title} {self.patents[i].abstract}".lower()], np.int64)
                sets.append(s if s is not None else np.empty(0, np.int64))
            if f.keyword_mode == "any":
                kw_ids = np.unique(np.concatenate(sets)) if sets else np.empty(0, np.int64)
            else:
                kw_ids = None
                for s in sets:
                    kw_ids = inter(kw_ids, s)
            cand = inter(cand, kw_ids)
        if f.title:
            q = " ".join(f.title.lower().split())
            pool = None
            for g in sorted(trigrams(q), key=lambda g: len(self.title_tri.get(g, ()))):  # rarest first
                pool = inter(pool, self.title_tri.get(g, np.empty(0, np.int64)))
                if len(pool) == 0:
                    break
            pool = np.arange(len(self.patents)) if pool is None else pool  # query < 3 chars
            ok = [i for i in pool if (self.titles_lower[i] == q if f.title_exact else q in self.titles_lower[i])]
            cand = inter(cand, np.asarray(ok, dtype=np.int64))
        if f.date_from or f.date_to:
            lo = f.date_from.toordinal() if f.date_from else -10**9
            hi = f.date_to.toordinal() if f.date_to else 10**9
            cand = inter(cand, np.nonzero((self.weeks >= lo) & (self.weeks <= hi))[0])
        return cand

    def _passes_naive(self, p: Patent, f: Filters) -> bool:
        """String-scan version of the same filters (used only by the post-filter baseline)."""
        if f.cpc_prefixes and not any(p.cpc.startswith(normalize_cpc(x)) for x in f.cpc_prefixes):
            return False
        if f.keywords:
            text_toks = set(tokenize(f"{p.title} {p.abstract}"))
            text = f"{p.title} {p.abstract}".lower()
            hits = [(kw.lower() in text) if len(tokenize(kw)) > 1 else (tokenize(kw)[0] in text_toks)
                    for kw in f.keywords if tokenize(kw)]
            if not (all(hits) if f.keyword_mode == "all" else any(hits)):
                return False
        if f.title:
            t, q = p.title.lower(), " ".join(f.title.lower().split())
            if not (t == q if f.title_exact else q in t):
                return False
        if f.date_from or f.date_to:
            d = p.date_ordinal
            if d < 0 or (f.date_from and d < f.date_from.toordinal()) or (f.date_to and d > f.date_to.toordinal()):
                return False
        return True

    # ----------------------------------------------------------------- search
    def query_for_patent(self, doc_number: str) -> str:
        p = self.patents[self.by_id[doc_number]]
        ind = p.independent_claims()
        return ind[0] if ind else (p.abstract or p.title)

    def search(self, query: str | None = None, *, patent_id: str | None = None,
               filters: Filters | None = None, top_k: int = 10, mode: str = "dense",
               strategy: str = "prefilter") -> list[Result]:
        """mode: dense | bm25 | rrf (dense+bm25 reciprocal-rank fusion)
        Note: with rrf, pre-filtering fuses ranks *within* the filtered set, while the
        post-filter baseline fuses ranks over the whole corpus, so their orders can differ.
        strategy: prefilter (index-resolved, default) | postfilter (naive baseline)"""
        f = filters or Filters()
        exclude = None
        if patent_id:
            if patent_id not in self.by_id:
                raise KeyError(f"{patent_id} not in index")
            query = self.query_for_patent(patent_id)
            exclude = self.by_id[patent_id]
        if not query:
            raise ValueError("need query or patent_id")

        cand = self.candidate_ids(f) if strategy == "prefilter" else None
        if cand is not None and exclude is not None:
            cand = cand[cand != exclude]
        if cand is not None and len(cand) == 0:
            return []

        if mode == "dense":
            ids, scores, best = self._dense(query, cand)
        elif mode == "bm25":
            ids, scores, best = self._bm25(query, cand)
        elif mode == "rrf":
            ids, scores, best = self._rrf(query, cand)
        else:
            raise ValueError(mode)

        order = np.argsort(-scores, kind="stable")
        out: list[Result] = []
        for o in order:
            i = int(ids[o])
            if i == exclude:
                continue
            p = self.patents[i]
            if strategy == "postfilter" and f.active() and not self._passes_naive(p, f):
                continue
            out.append(self._result(i, float(scores[o]), best(int(o)), f))
            if len(out) >= top_k:
                break
        return out

    def _dense(self, query, cand):
        q = self.encode([query])[0]
        if cand is None or len(cand) > GATHER_MAX_FRACTION * len(self.patents):
            # broad / no filter: one contiguous matmul, then pick out the candidates' scores
            rows = None
            sims = self.emb @ q
            seg_starts, ids = self.starts, np.arange(len(self.patents))
            scores = np.maximum.reduceat(sims, seg_starts)
            if cand is not None:
                scores, seg_starts, ids = scores[cand], seg_starts[cand], cand
            bounds_end = self.ends[ids]
        else:
            # selective filter: gather only the candidate patents' passage rows -> smaller matmul
            mask = np.zeros(len(self.patents), dtype=bool)
            mask[cand] = True
            rows = np.flatnonzero(mask[self.owner])  # contiguous per patent, in cand order
            sims = self.emb[rows] @ q
            lens = self.ends[cand] - self.starts[cand]
            seg_starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            ids = cand
            scores = np.maximum.reduceat(sims, seg_starts)
            bounds_end = seg_starts + lens

        def best(k: int) -> int:  # best passage row of the k-th candidate, computed lazily
            a, b = seg_starts[k], bounds_end[k]
            local = a + int(np.argmax(sims[a:b]))
            return int(local if rows is None else rows[local])
        return ids, scores, best

    def _bm25(self, query, cand):
        scores = self.bm25.scores(query, cand)
        ids = np.arange(len(self.patents)) if cand is None else cand
        starts = self.starts[ids]  # BM25 is patent-level -> show abstract as snippet
        return ids, scores, lambda k: int(starts[k])

    def _rrf(self, query, cand, k=60):
        di, ds, db = self._dense(query, cand)
        bi, bs, _ = self._bm25(query, cand)
        rr = np.zeros(len(self.patents))
        for ids, sc in ((di, ds), (bi, bs)):
            rr[ids[np.argsort(-sc)]] += 1.0 / (k + np.arange(1, len(ids) + 1))
        return di, rr[di], db

    def _result(self, i, score, row, f: Filters) -> Result:
        p = self.patents[i]
        kws = []
        if f.keywords:
            text = f"{p.title} {p.abstract}".lower()
            kws = [k for k in f.keywords if k.lower() in text]
        return Result(p.doc_number, p.title, p.classification,
                      p.week.isoformat() if p.week else str(p.year or ""), round(score, 4),
                      self.kinds[row], self.passages[row], kws)


def build_engine(data_dir: str = DEFAULT_DATA_DIR, model: str = DEFAULT_MODEL,
                 cache_dir: str = "index_cache", max_paras: int | None = None) -> PatentSearchEngine:
    return PatentSearchEngine(load_patents(data_dir), model_name=model, cache_dir=cache_dir, max_paras=max_paras)


def results_to_json(results: list[Result]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
