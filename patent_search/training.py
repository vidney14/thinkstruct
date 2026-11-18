"""Part 3 - evaluation & fine-tuning pipeline for the embedding model.

Task we evaluate / train for ("claim -> source patent", a proxy for claim mapping):
  * Index   : every patent represented ONLY by its title+abstract and description
              paragraphs (claims are removed, otherwise a claim trivially finds itself).
  * Query   : a claim of a held-out patent. Independent claims are used as-is;
              dependent claims have their "according to claim N" reference stripped
              (data.strip_dependency) so they read as standalone feature statements.
  * Target  : the patent the claim came from. Patent score = max over its passages,
              exactly like the Part 1 engine.
  Metrics   : Recall@1/5/10 and MRR@10 (retrieval), plus pair metrics on held-out
              triplets: mean cos(claim, positive), mean cos(claim, hard negative),
              their margin, and triplet accuracy (pos > neg).

Training data (train-split patents only - the split is by patent, so no test claim
or its patent text is used as a positive/anchor during training):
  * anchor   = claim (cleaned as above)
  * positive = the paragraph of the SAME patent that best supports the claim
               (BM25 over that patent's paragraphs; falls back to the abstract)
  * negative = "looks related but isn't": the best passage of the highest-ranked
               OTHER patent (base model) sharing the claim's 4-char CPC class,
               skipping rank-1 to reduce false negatives (true prior art).
  Loss: MultipleNegativesRankingLoss (in-batch negatives + the mined hard negative).

Caveat: "the same patent" is a weak label - a claim can legitimately match another
patent (that IS prior art). Hard negatives may therefore contain false negatives;
skipping the top-ranked candidate is a cheap mitigation.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import numpy as np

from .data import Patent, strip_dependency, tokenize

log = logging.getLogger(__name__)
MIN_PARA = 40
MIN_QUERY_CHARS = 60


@dataclass
class Query:
    text: str
    patent: int        # index into patents
    kind: str          # independent | dependent


def split_patents(patents: list[Patent], test_frac: float = 0.2, seed: int = 13):
    idx = list(range(len(patents)))
    random.Random(seed).shuffle(idx)
    n_test = int(len(idx) * test_frac)
    return sorted(idx[n_test:]), sorted(idx[:n_test])


def make_queries(patents: list[Patent], ids, max_dependent: int = 2, seed: int = 13) -> list[Query]:
    rng = random.Random(seed)
    out = []
    for i in ids:
        p = patents[i]
        ind = [c for c in p.independent_claims() if len(c) >= MIN_QUERY_CHARS]
        if ind:
            out.append(Query(ind[0], i, "independent"))
        dep = [strip_dependency(c) for c in p.dependent_claims()]
        dep = [c for c in dep if len(c) >= MIN_QUERY_CHARS]
        for c in rng.sample(dep, min(max_dependent, len(dep))):
            out.append(Query(c, i, "dependent"))
    return out


class Corpus:
    """Claim-free passage corpus (title+abstract + description paragraphs)."""

    def __init__(self, patents: list[Patent]):
        texts, owner = [], []
        for i, p in enumerate(patents):
            head = f"{p.title}. {p.abstract}".strip(". ")
            paras = [x for x in p.description if len(x) >= MIN_PARA]
            for t in ([head] if head else []) + paras:
                texts.append(t)
                owner.append(i)
        self.texts = texts
        self.owner = np.asarray(owner)
        self.n = len(patents)
        order_ok = np.all(np.diff(self.owner) >= 0)
        assert order_ok
        self.has = np.zeros(self.n, bool)
        self.has[np.unique(self.owner)] = True
        # segment starts for reduceat over patents that have passages
        self.pids = np.unique(self.owner)
        self.starts = np.searchsorted(self.owner, self.pids)

    def patent_scores(self, sims: np.ndarray) -> np.ndarray:
        """sims: (Q, P) passage similarities -> (Q, n_patents) max-pooled, -inf if no passage."""
        out = np.full((sims.shape[0], self.n), -np.inf, dtype=np.float32)
        out[:, self.pids] = np.maximum.reduceat(sims, self.starts, axis=1)
        return out

    def best_passage(self, sims_row: np.ndarray, patent: int) -> int:
        rows = np.flatnonzero(self.owner == patent)
        return int(rows[np.argmax(sims_row[rows])])


def encode(model, texts, bs=64):
    return model.encode(texts, batch_size=bs, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def retrieval_metrics(model, corpus: Corpus, queries: list[Query], corpus_emb=None):
    corpus_emb = encode(model, corpus.texts) if corpus_emb is None else corpus_emb
    q_emb = encode(model, [q.text for q in queries])
    scores = corpus.patent_scores(q_emb @ corpus_emb.T)
    gold = np.array([q.patent for q in queries])
    gold_score = scores[np.arange(len(queries)), gold]
    ranks = 1 + (scores > gold_score[:, None]).sum(1)  # 1 = best
    return _summarise(ranks, queries), ranks


def bm25_metrics(corpus: Corpus, queries: list[Query]):
    from rank_bm25 import BM25Okapi
    docs = [[] for _ in range(corpus.n)]
    for t, o in zip(corpus.texts, corpus.owner):
        docs[o].extend(tokenize(t))
    bm = BM25Okapi(docs)
    ranks = []
    for q in queries:
        s = bm.get_scores(tokenize(q.text))
        ranks.append(1 + int((s > s[q.patent]).sum()))
    return _summarise(np.array(ranks), queries), np.array(ranks)


def _summarise(ranks: np.ndarray, queries: list[Query]) -> dict:
    def block(r):
        return {
            "n": int(len(r)),
            "recall@1": float(np.mean(r <= 1)),
            "recall@5": float(np.mean(r <= 5)),
            "recall@10": float(np.mean(r <= 10)),
            "mrr@10": float(np.mean(np.where(r <= 10, 1.0 / r, 0.0))),
        }
    kinds = np.array([q.kind for q in queries])
    return {"all": block(ranks), **{k: block(ranks[kinds == k]) for k in ("independent", "dependent")}}


def mine_triplets(model, patents: list[Patent], corpus: Corpus, queries: list[Query],
                  corpus_emb=None, skip_top: int = 1, pool: int = 10):
    """Build (anchor, positive, hard_negative) text triplets for `queries`."""
    from rank_bm25 import BM25Okapi
    corpus_emb = encode(model, corpus.texts) if corpus_emb is None else corpus_emb
    q_emb = encode(model, [q.text for q in queries])
    sims = q_emb @ corpus_emb.T
    pscores = corpus.patent_scores(sims)
    bm25_cache: dict[int, tuple] = {}
    triplets = []
    for qi, q in enumerate(queries):
        # positive: BM25-best passage of the same patent (independent of the model being trained)
        rows = np.flatnonzero(corpus.owner == q.patent)
        if len(rows) == 0:
            continue
        if q.patent not in bm25_cache:
            bm25_cache[q.patent] = (rows, BM25Okapi([tokenize(corpus.texts[r]) or ["_"] for r in rows]))
        rows, bm = bm25_cache[q.patent]
        pos = corpus.texts[rows[int(np.argmax(bm.get_scores(tokenize(q.text))))]]
        # hard negative: high-ranked other patent in the same CPC class, skipping the very top
        cls = patents[q.patent].cpc[:4]
        order = [int(j) for j in np.argsort(-pscores[qi])[: pool + 1] if j != q.patent]
        same = [j for j in order if patents[j].cpc[:4] == cls] or order
        neg_pid = same[min(skip_top, len(same) - 1)]
        neg = corpus.texts[corpus.best_passage(sims[qi], neg_pid)]
        triplets.append((q.text, pos, neg))
    return triplets


def pair_metrics(model, triplets) -> dict:
    a = encode(model, [t[0] for t in triplets])
    p = encode(model, [t[1] for t in triplets])
    n = encode(model, [t[2] for t in triplets])
    sp, sn = (a * p).sum(1), (a * n).sum(1)
    return {"n": len(triplets), "cos_pos": float(sp.mean()), "cos_neg": float(sn.mean()),
            "margin": float((sp - sn).mean()), "triplet_acc": float((sp > sn).mean())}


def train(model, triplets, out_dir: str, epochs: int = 1, batch_size: int = 32, lr: float = 2e-5, seed: int = 13):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    ds = Dataset.from_dict({"anchor": [t[0] for t in triplets], "positive": [t[1] for t in triplets],
                            "negative": [t[2] for t in triplets]})
    args = SentenceTransformerTrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_steps=int(0.1 * len(ds) / batch_size * epochs), seed=seed,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # duplicate texts in a batch = false in-batch negatives
        logging_steps=10, save_strategy="no", report_to="none",
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(out_dir)
    return model
