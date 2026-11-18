# Part 2 — Running the search engine on 10M patents

Audience: an engineer (intern) who has read the Part 1 code and needs to build the
production version. It is deliberately simple: two data stores, one queue and
stateless services. The open problems are listed at the end, not hidden.

## 1. What we are building

Same product as Part 1: query by text, claim or patent number, then semantic ranking
with hybrid filters (CPC prefix, keywords in title/abstract, title, filing date). Each
result comes back with the claim or paragraph that matched.

**Sizing**, measured on the large sample (`coding_challenge_data`: 471k records, 457k unique patents across all CPC sections, 30 GB of JSON):

| measured | value |
|---|---|
| records → unique patents | 471,000 → 457,385 (2.9% duplicate copies; ~22% of duplicated doc numbers have *differing* content) |
| missing classification / title / description | 38% / 9% / 1% |
| passages per patent, all paragraphs | ~135 (19 claims + ~111 description paragraphs + abstract) |
| passages per patent, first 10 paragraphs | ~30 (what we index; see `--max-paras`) |
| parse + clean throughput | 471 files in 119 s serial → **28 s with 10 processes** (4.3×), ≈ 17k patents/s/machine |
| embedding throughput, MiniLM | ~600 passages/s on an Apple M-series GPU (MPS); an A10G is ~3–5× that |

| planning quantity | value |
|---|---|
| patents | 10M (+~7k/week new US applications; ×N if we add other offices) |
| passages per patent | ~30 (abstract + claims + first 10 description paragraphs). Indexing *all* paragraphs would be ~135, i.e. 4.5× the vectors, RAM and embedding cost |
| vectors | **~300M** × 384 dims (MiniLM), or 768 if we move to a larger model |
| raw text | ~1 MB/patent compressed XML → ~10 TB raw, ~1 TB as Parquet text |
| query load | assume 10 QPS peak and a p95 target of < 500 ms |

## 2. Components

```
             weekly USPTO bulk XML / other offices
                          │
                    ┌─────▼──────┐   1 job per file
   scheduler ──────►│  Job table │◄──────────── status dashboard / alerts
   (cron)           │ (Postgres) │
                    └─────┬──────┘
         ┌────────────────┼──────────────────┐
   ┌─────▼─────┐   ┌──────▼──────┐   ┌───────▼────────┐
   │ parse +   │──►│ embed       │──►│ index writer   │
   │ normalize │   │ (GPU pool)  │   │ (upsert)       │
   │ (CPU)     │   └─────────────┘   └───┬────────┬───┘
   └─────┬─────┘                         │        │
         ▼                               ▼        ▼
   S3: raw + Parquet           Postgres         Vector DB (Qdrant)
   (source of truth)           metadata + FTS   passage vectors + payload
                                    ▲                  ▲
                                    └──── Search API ──┘  ◄── query-encoder (CPU)
```

| component | choice | why |
|---|---|---|
| Object store | S3 | raw files + normalized Parquet; everything can be rebuilt from here |
| Metadata DB | Postgres (RDS) | patents table (doc_number PK, title, abstract, CPC[], dates, content_hash, model_version); `tsvector` GIN index for keywords, `pg_trgm` GIN for title substring, btree `text_pattern_ops` for CPC prefix; also holds the **jobs** table |
| Vector DB | Qdrant (3–4 nodes) | filtered HNSW. Payload indexes on `cpc_levels` (keyword array), `week` (int), `doc_number`. Int8 scalar quantization in RAM, full vectors on disk |
| Queue | the Postgres `jobs` table (`SELECT … FOR UPDATE SKIP LOCKED`) | no extra infrastructure at this volume (thousands of jobs, not millions/s) |
| Workers | containers on ECS/K8s: CPU parsers, GPU (spot) embedders | autoscaled on queue depth |
| Search API | FastAPI, stateless, 2+ replicas | query encoding on CPU (~5 ms for MiniLM, measured locally) |

**CPC prefix trick:** store every level of each code, e.g. `["B","B60","B60B","B60B1","B60B1/04"]`.
A prefix filter then becomes an exact keyword match, which every vector DB can index.

## 3. Pipelines

### 3a. Ingest (weekly + one-time backfill)
1. **Enqueue**: the scheduler inserts one job per source file (`source` is UNIQUE, so enqueueing twice is a no-op).
2. **Parse/normalize** (CPU): XML → Patent records. Apply the same missing-field policy as Part 1. Write Parquet to S3. Compute `content_hash`.
3. **Diff**: skip patents whose `(content_hash, model_version)` is already indexed. Re-publications and corrections only re-embed what changed.
4. **Chunk + embed** (GPU): abstract, each claim and description chunks. Batches of about 256. Write vectors to S3 first, then upsert.
5. **Upsert**: Postgres row and vector points in one job. Point id = `hash(doc_number, passage_no)`, so retries overwrite instead of duplicating. Upsert the vectors **first** and flip `patents.indexed_model` in Postgres **last**, so a half-written patent is never visible as complete.
6. Mark the job `done` and record counts.

### 3b. Query
1. Resolve the input: text, claim, or a patent number (look up its first independent claim).
2. Encode the query (CPU, ~5 ms).
3. Search Qdrant with the filter as a payload condition (`cpc_levels`, `week` range). For keyword/title filters, first get candidate doc_numbers from Postgres (FTS / trigram). If there are fewer than ~10k, pass them as an id filter. Otherwise let HNSW filter on payload.
4. Group passage hits by patent (max score; ask for `limit = 5×k` passages), then fetch titles and snippets from Postgres.
5. Optional: re-rank (cross-encoder or BM25 fusion), as in Part 1's `rrf` mode.

### 3c. Model upgrade / re-index (blue-green)
A new embedding model (e.g. the Part 3 fine-tuned one) means re-embedding 300M passages. Do it like this:
1. Build a new Qdrant collection `passages_v2` from the Parquet in S3, using the same ingest jobs with `model_version=v2`.
2. Run the Part 3 eval set against both collections.
3. Swap the alias `passages → passages_v2`, and keep v1 for a week so you can roll back.

## 4. Errors, retries, tracking

* **jobs table**: `status (pending|running|done|failed), attempts, worker, lease_until, error, n_patents, n_passages, timings`.
* A worker claims a job atomically. If a worker dies, its job's lease expires and another worker picks the job up.
* After 3 failed attempts the job moves to `failed` (the dead-letter state) with the error saved. Retry is one SQL update after the fix.
* **Per-record errors** (malformed XML, empty patent) go to a `rejects` table with a reason. They do **not** fail the whole file.
* **Nightly reconciliation**: `count(patents where indexed_model=v)` vs. distinct doc_numbers in the vector DB, plus a random sample of 1k points checked against Postgres. Anything that disagrees gets re-enqueued.
* **Dashboards and alerts**:
  - jobs by status, ingest lag (newest file week vs. today), throughput, GPU utilisation;
  - query p50/p95, error rate, zero-result rate;
  - a **nightly quality canary**: the Part 3 Recall@10 on a fixed held-out claim set. A drop means an index or model bug.

`poc/ingest_queue.py` implements the jobs table, atomic claiming, leases, retries and the dead-letter state, idempotent upserts, status reporting and the SQL-prefilter → vector query path, in miniature (SQLite + .npy shards).

## 5. Cost (rough, on-demand AWS us-east-1 list prices; ±50%)

**One-time backfill**

| item | estimate |
|---|---|
| Embedding 300M passages, MiniLM, ~2–3k passages/s per A10G (g5.xlarge ~$1/h, spot ~$0.4/h). For comparison, a laptop at the measured 600/s would need ~140 h | ~30–40 GPU-hours → **~$15–40** |
| …same with a 768-d base-size model (~4–5× slower) | ~150 GPU-h → **~$60–150** |
| Parsing 10 TB of XML on CPU spot | **~$50** |

**Monthly steady state**

| item | sizing | $/month |
|---|---|---|
| Qdrant cluster | 300M × 384 int8 ≈ 115 GB + HNSW graph ≈ 40 GB in RAM → 4 × r6i.2xlarge (64 GB) + 4 × 500 GB gp3 for full vectors | ~$1,500 |
| Postgres | db.r6g.xlarge + 1.5 TB gp3 (text + FTS indexes) | ~$550 |
| S3 | ~11 TB | ~$250 |
| Search API + query encoders | 2–3 × c6i.xlarge | ~$300 |
| Weekly ingest | ~7k patents/week → minutes of GPU | <$10 |
| Monitoring, logs, egress | | ~$150 |
| **Total** | | **≈ $2.7k / month** |

Ways to cut it: put only abstract + claims (~20/patent) in the vector index and keep description chunks in a cheaper disk-based index (≈ −40% RAM). Use binary quantization with rescoring (≈ −75% RAM). Use 3-year reserved instances (≈ −40%).

## 6. Things that will be hard at scale

1. **Filtered ANN.** HNSW recall collapses when a filter is very selective (graph neighbours are filtered out). This is the same thing the Part 1 benchmark shows: below about 10% selectivity (tuned on the 25k-patent sample) it was faster to gather the candidate rows and do exact search than to scan everything. Qdrant does the same switch internally (exact search under a payload-cardinality threshold). For keyword filters we have to do it ourselves: small candidate sets go through an id filter and exact scoring.
2. **Passage → patent aggregation** with ANN. We only see the top N passages, so a patent whose best passage ranks N+1 is lost. Over-fetch (5–10× k) and measure the effect with the eval set.
3. **Two stores stay consistent only eventually.** The write order (vectors first, Postgres flag last) plus nightly reconciliation is our answer. Do not try to do cross-store transactions.
4. **Re-embedding cost and time** whenever the model changes (see 3c). Version every vector with `model_version`.
5. **Duplicates and families.** An application, its grant and its continuations are near-identical. Dedup or group by family, or the results fill up with copies of one invention.
6. **Long documents.** Claims are written in legal language and descriptions run to hundreds of paragraphs. Chunking and the 256-token limit throw context away.
7. **Hot spots.** Backfill writes compete with query traffic. Throttle writers or build new segments offline.
8. **Dirty data, seen in the large sample.**
   - 38% of patents have no classification. A CPC filter therefore silently drops more than a third of the corpus, including possibly the most relevant prior art. The UI should say so, and the missing codes should be back-filled from the USPTO master classification file, or predicted with a classifier.
   - 22% of duplicated doc numbers carry different content. The ingest has to pick a winner deterministically (we keep the most complete copy) and record which source won.
9. **Non-English offices** (EP/CN/JP/KR) need translation or a multilingual model. That is a new model version, which means a full re-index.

## 7. Suggested order of work for the intern

1. Postgres schema + jobs table, and port `poc/ingest_queue.py` to it (`SKIP LOCKED`).
2. Parser for USPTO bulk XML → Parquet, plus the rejects table.
3. Embedding worker container (GPU), on a 100k-patent subset → stand up Qdrant and the search API.
4. Eval canary (Part 3 script) wired into CI and run nightly.
5. Backfill 10M, then the dashboards and reconciliation.
