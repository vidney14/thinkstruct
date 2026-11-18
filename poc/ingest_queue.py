"""Part 2 proof-of-concept: the INGEST pipeline from docs/SCALING.md, in miniature.

A SQLite file stands in for Postgres (jobs table + patents metadata table), a
directory of .npy files stands in for the vector store, and local processes
stand in for the GPU embedding workers.  Fast-and-sloppy by design.

    python poc/ingest_queue.py reset
    python poc/ingest_queue.py enqueue data/patent_data_small        # 1 job per weekly file
    python poc/ingest_queue.py work --workers 3 --fail-rate 0.3      # inject failures to see retries/DLQ
    python poc/ingest_queue.py status
    python poc/ingest_queue.py retry-failed                          # move dead-lettered jobs back to pending
    python poc/ingest_queue.py search "tire tread noise" --cpc B60C  # SQL pre-filter -> vectors

What it demonstrates
  * atomic job claiming (UPDATE ... RETURNING) - safe with N concurrent workers
  * lease/visibility timeout: a job whose worker died is re-claimed after LEASE_S
  * retries with max attempts, then status='failed' (dead-letter) with the error kept
  * idempotency: patents upserted by doc_number + content hash; re-running a job
    or ingesting a duplicate application is a no-op for unchanged content
  * status tracking: per-job and global counters, throughput
  * query path: metadata filter in SQL -> candidate ids -> vector shard rows only
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.join(os.path.dirname(__file__), "state")
DB = os.path.join(ROOT, "pipeline.db")
SHARDS = os.path.join(ROOT, "shards")
MAX_ATTEMPTS = 3
LEASE_S = 120
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, source TEXT UNIQUE, status TEXT DEFAULT 'pending',  -- pending|running|done|failed
  attempts INT DEFAULT 0, worker TEXT, lease_until REAL, error TEXT,
  n_patents INT, n_new INT, n_passages INT, enqueued REAL, started REAL, finished REAL);
CREATE TABLE IF NOT EXISTS patents(
  doc_number TEXT PRIMARY KEY, title TEXT, abstract TEXT, cpc TEXT, week TEXT,
  content_hash TEXT, model TEXT, shard TEXT, row_start INT, row_end INT, job_id INT);
CREATE INDEX IF NOT EXISTS patents_cpc ON patents(cpc);           -- prefix search via range scan
CREATE VIRTUAL TABLE IF NOT EXISTS patents_fts USING fts5(doc_number UNINDEXED, title, abstract);
"""


def conn():
    c = sqlite3.connect(DB, timeout=30, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def cmd_reset(_):
    import shutil
    shutil.rmtree(ROOT, ignore_errors=True)
    os.makedirs(SHARDS)
    conn().executescript(SCHEMA)
    print("reset", ROOT)


def cmd_enqueue(a):
    c = conn()
    n = 0
    for f in sorted(glob.glob(os.path.join(a.data_dir, "patents_ipa*.json"))):
        n += c.execute("INSERT OR IGNORE INTO jobs(source, enqueued) VALUES (?, ?)",
                       (os.path.abspath(f), time.time())).rowcount
    print(f"enqueued {n} new jobs")


def claim(c, worker):
    now = time.time()
    # pending jobs, or running jobs whose lease expired (worker crashed)
    return c.execute(
        """UPDATE jobs SET status='running', worker=?, attempts=attempts+1, lease_until=?, started=?
           WHERE id = (SELECT id FROM jobs WHERE status='pending'
                       OR (status='running' AND lease_until < ?) ORDER BY id LIMIT 1)
           RETURNING id, source, attempts""", (worker, now + LEASE_S, now, now)).fetchone()


def process(c, model, job_id, source, fail_rate):
    from patent_search.data import load_patents
    import tempfile
    # load_patents works on a directory; point it at a temp dir holding just this file
    with tempfile.TemporaryDirectory() as d:
        os.symlink(source, os.path.join(d, os.path.basename(source)))
        patents = load_patents(d)
    if random.random() < fail_rate:
        raise RuntimeError("injected failure (simulated GPU OOM / network error)")
    todo, texts, spans = [], [], []
    for p in patents:
        h = hashlib.sha1(json.dumps([p.title, p.abstract, p.claims, p.description]).encode()).hexdigest()
        row = c.execute("SELECT content_hash, model FROM patents WHERE doc_number=?", (p.doc_number,)).fetchone()
        if row and row == (h, MODEL):
            continue  # idempotent: unchanged content, same model -> skip
        start = len(texts)
        texts += [f"{p.title}. {p.abstract}"] + p.claims + [x for x in p.description if len(x) >= 40]
        todo.append((p, h, start, len(texts)))
    shard = None
    if texts:
        emb = model.encode(texts, batch_size=64, normalize_embeddings=True).astype(np.float16)
        shard = f"job{job_id:05d}.npy"
        np.save(os.path.join(SHARDS, shard + ".tmp.npy"), emb)
        os.replace(os.path.join(SHARDS, shard + ".tmp.npy"), os.path.join(SHARDS, shard))  # atomic publish
    c.execute("BEGIN IMMEDIATE")
    for p, h, s, e in todo:
        c.execute("""INSERT INTO patents VALUES (?,?,?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(doc_number) DO UPDATE SET title=excluded.title, abstract=excluded.abstract,
                     cpc=excluded.cpc, week=excluded.week, content_hash=excluded.content_hash, model=excluded.model,
                     shard=excluded.shard, row_start=excluded.row_start, row_end=excluded.row_end, job_id=excluded.job_id""",
                  (p.doc_number, p.title, p.abstract, p.cpc, p.week.isoformat() if p.week else None,
                   h, MODEL, shard, s, e, job_id))
        c.execute("DELETE FROM patents_fts WHERE doc_number=?", (p.doc_number,))
        c.execute("INSERT INTO patents_fts VALUES (?,?,?)", (p.doc_number, p.title, p.abstract))
    c.execute("""UPDATE jobs SET status='done', finished=?, error=NULL, n_patents=?, n_new=?, n_passages=?
                 WHERE id=?""", (time.time(), len(patents), len(todo), len(texts), job_id))
    c.execute("COMMIT")
    return len(patents), len(todo)


def worker_main(name, fail_rate):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL, device="cpu")
    c = conn()
    while True:
        job = claim(c, name)
        if not job:
            print(f"[{name}] queue empty, exiting")
            return
        jid, src, attempts = job
        t = time.time()
        try:
            n, new = process(c, model, jid, src, fail_rate)
            print(f"[{name}] job {jid} done: {n} patents ({new} new/changed) in {time.time() - t:.1f}s")
        except Exception as e:  # retry or dead-letter
            if c.in_transaction:
                c.execute("ROLLBACK")
            status = "failed" if attempts >= MAX_ATTEMPTS else "pending"
            c.execute("UPDATE jobs SET status=?, error=?, worker=NULL WHERE id=?", (status, repr(e)[:300], jid))
            print(f"[{name}] job {jid} attempt {attempts} FAILED -> {status}: {e}")


def cmd_work(a):
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=worker_main, args=(f"w{i}", a.fail_rate)) for i in range(a.workers)]
    t = time.time()
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"all workers finished in {time.time() - t:.1f}s")
    cmd_status(a)


def cmd_status(_):
    c = conn()
    print("\njobs by status:", dict(c.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()))
    n, p = c.execute("SELECT COUNT(*), COALESCE(SUM(row_end-row_start),0) FROM patents").fetchone()
    print(f"patents indexed: {n}   passages/vectors: {p}")
    r = c.execute("SELECT MIN(started), MAX(finished), SUM(n_new) FROM jobs WHERE status='done'").fetchone()
    if r[0]:
        print(f"throughput: {r[2] / max(r[1] - r[0], 1e-9):.1f} new patents/s")
    print(f"\n{'id':>3} {'status':8} {'try':>3} {'worker':6} {'pat':>4} {'new':>4} {'pass':>6}  source / error")
    for row in c.execute("SELECT id,status,attempts,COALESCE(worker,''),n_patents,n_new,n_passages,source,error FROM jobs"):
        i, s, t, w, n, new, ps, src, err = row
        print(f"{i:>3} {s:8} {t:>3} {w:6} {n or '':>4} {new if new is not None else '':>4} {ps or '':>6}  "
              f"{os.path.basename(src)}" + (f"  !! {err}" if err and s != 'done' else ""))


def cmd_retry(_):
    n = conn().execute("UPDATE jobs SET status='pending', attempts=0 WHERE status='failed'").rowcount
    print(f"re-queued {n} failed jobs")


def cmd_search(a):
    from sentence_transformers import SentenceTransformer
    c = conn()
    sql, args = "SELECT p.doc_number, title, cpc, shard, row_start, row_end FROM patents p", []
    where = []
    if a.cpc:  # prefix on an indexed column -> index range scan (Postgres: text_pattern_ops)
        where.append("(" + " OR ".join("cpc >= ? AND cpc < ?" for _ in a.cpc) + ")")
        for x in a.cpc:
            args += [x.upper(), x.upper() + "￿"]
    if a.keywords:
        where.append("p.doc_number IN (SELECT doc_number FROM patents_fts WHERE patents_fts MATCH ?)")
        args.append(" AND ".join(f'"{k}"' for k in a.keywords))
    if where:
        sql += " WHERE " + " AND ".join(where)
    t = time.perf_counter()
    rows = c.execute(sql, args).fetchall()
    t_sql = time.perf_counter() - t
    q = SentenceTransformer(MODEL, device="cpu").encode([a.query], normalize_embeddings=True)[0].astype(np.float32)
    t = time.perf_counter()
    shards, res = {}, []
    for doc, title, cpc, shard, s, e in rows:
        if shard not in shards:
            shards[shard] = np.load(os.path.join(SHARDS, shard), mmap_mode="r")
        res.append((float((shards[shard][s:e].astype(np.float32) @ q).max()), doc, title, cpc))
    res.sort(reverse=True)
    print(f"{len(rows)} candidates from SQL in {t_sql * 1000:.1f} ms, vector scoring {1000 * (time.perf_counter() - t):.1f} ms")
    for sc, doc, title, cpc in res[: a.k]:
        print(f"  {sc:.3f} {doc} {cpc:12} {title}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("reset")
    e = sp.add_parser("enqueue"); e.add_argument("data_dir")
    w = sp.add_parser("work"); w.add_argument("--workers", type=int, default=2); w.add_argument("--fail-rate", type=float, default=0.0)
    sp.add_parser("status"); sp.add_parser("retry-failed")
    s = sp.add_parser("search"); s.add_argument("query"); s.add_argument("--cpc", nargs="+")
    s.add_argument("--keywords", nargs="+"); s.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    if a.cmd != "reset" and not os.path.exists(DB):
        cmd_reset(a)
    {"reset": cmd_reset, "enqueue": cmd_enqueue, "work": cmd_work, "status": cmd_status,
     "retry-failed": cmd_retry, "search": cmd_search}[a.cmd](a)


if __name__ == "__main__":
    main()
