# 2-Minute Walkthrough Video: https://drive.google.com/file/d/1xdtPDXj5CAfI9_OYhWefv0ceicMdKWpW/view?usp=share_link

# Patent search: hybrid semantic search, scaling plan, and embedding fine-tuning

## Problem statement

A patent examiner or attorney has **a claim** (theirs, or one from a patent they are
examining). They need to find **which existing applications disclose the same thing, and where**.
Two things matter:
- the specific claim or description paragraph that matched, not just the patent;
- narrowing by the usual examiner constraints: CPC class (e.g. `B60B` wheels), keywords in the title or abstract, title, and filing date.

So this project builds:

1. **Part 1 – hybrid passage-level search engine.** The input is a natural-language query, a claim, or a patent number (patent → similar patents). The output is ranked patents, each with the best-matching claim or paragraph as the "why it matched" snippet, plus metadata. The filters are resolved through inverted indexes **before** the vector math. Timings are given with filters on and off.
2. **Part 2 – a design for 10M patents** ([docs/SCALING.md](docs/SCALING.md)), plus a proof of concept of its ingest pipeline: a job queue with retries, dead-lettering, idempotent upserts and status tracking ([poc/ingest_queue.py](poc/ingest_queue.py)).
3. **Part 3 – evaluation & training.** A reproducible pipeline that measures how well the embedding model maps a claim back to the patent that discloses it. It mines positive and hard-negative training pairs from the data, fine-tunes the model, and re-measures.
4. **Bonus – the large dataset** (`coding_challenge_data`, 471k records / 30 GB). I ran the engine, benchmark and proof of concept on a real 25k-patent sample and made the slow parts faster, in the spirit of the "Efficiency and large data" option. See [Bonus: large dataset](#bonus-large-dataset-efficiency).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# data: unzip the provided archive so the JSON files are in data/patent_data_small/
#   (or point elsewhere with: export PATENT_DATA_DIR=/path/to/patent_data_small)
```
Optional, for the large dataset (`coding_challenge_data`, unzipped anywhere):
```bash
python scripts/prepare_large.py ~/Downloads/coding_challenge_data --frac 0.055   # -> data/patent_data_large_sample (25k patents, ~30 s)
```

The first run downloads `sentence-transformers/all-MiniLM-L6-v2` (~90 MB). It then encodes all
32k passages (~60 s on an M-series Mac with MPS) and caches them in `index_cache/`. Later
start-ups take about 4 s.

## Data handling (missing fields)

The decision is documented in [patent_search/data.py](patent_search/data.py):

* A patent is **excluded only** if it has no `doc_number`, or no searchable text at all (title, abstract, claims and description all missing). In the sample data, 0 patents are excluded.
* Any other missing field is **kept and handled separately**. A missing description or abstract simply produces no passages of that type: 119 of the 640 patents have no detailed description and stay searchable through their abstract and claims. A missing classification never matches a CPC filter, but the patent is returned when no CPC filter is set.
* Duplicate `doc_number`s keep the **most complete** copy (most non-empty fields; ties → first seen).
* **Large dataset** (same policy, plus):
  - classification `"N/A"` is treated as missing; codes are upper-cased with spaces removed, and the `/` is kept so `A61B5/097` ≠ `A61B50/97`;
  - corrupted files are logged and skipped, and invalid records are dropped. Nothing was corrupted in practice, but the README of the data warned about it;
  - there is no filing week in the file names, so the date filter falls back to the publication year from the doc number.
* Claims are cleaned (`"claim 1 ,"` → `"claim 1,"`). Each claim is tagged independent or dependent, so a patent-number query uses its first independent claim.
* The filing week is parsed from the file name (`patents_ipa240905` → 2024-09-05) and drives the date filter.

## Part 1 – search engine

**Design** ([patent_search/engine.py](patent_search/engine.py))

* **Unit of retrieval = passage.** A passage is the title+abstract, each claim, or each description paragraph (≥ 40 chars), which gives 32,298 passages for 640 patents. A patent's score is the max over its passages. Passages are stored contiguously per patent, so this is a single `np.maximum.reduceat`, and the argmax passage becomes the snippet.
* **Retrieval modes**:
  - `dense` (MiniLM cosine, exact brute force);
  - `bm25` (title+abstract+claims);
  - `rrf` (reciprocal-rank fusion of dense and BM25, which helps on exact technical terms).
* **Hybrid filters**, all combinable:
  - CPC prefix(es) (`--cpc B60B B60C11`);
  - keywords in title/abstract (`--keywords tread noise`, ALL by default or `--any`, multi-word phrases supported);
  - title substring or exact title (`--title`, `--title-exact`);
  - filing-week range (`--from/--to`).
* **How filters are made cheap:**
  - **CPC**: a sorted `(code, id)` list, so a prefix is two binary searches, `O(log n + hits)`.
  - **Keywords**: a token → sorted posting array index over title+abstract. Postings are intersected; phrases are verified only on the survivors.
  - **Title substring**: a character-trigram index (the pg_trgm idea). The rarest trigram is intersected first, then the substring is verified.
  - **Dates**: a vectorised range mask.
  - **Adaptive scoring.** If the candidate set is under 10% of the corpus, only those patents' passage rows are gathered and scored, a smaller matmul. Otherwise the full contiguous matmul runs and the candidates are picked out afterwards, because gathering 45% of a big matrix costs more than multiplying all of it. The threshold was tuned on the large data (15% → 10%; see below).
* **BM25** is a precomputed sparse matrix ([patent_search/bm25.py](patent_search/bm25.py)), so a query is a sum of a few columns. It gives the same scores as `rank_bm25` and is 178× faster per query on 25k patents.

**Run it**

```bash
python -m patent_search.cli "spoke connecting the hub to the rim of a bicycle wheel"
python -m patent_search.cli "tire tread pattern that reduces road noise" --cpc B60C --keywords tread --mode rrf
python -m patent_search.cli --patent 20240051333 -k 5              # patent -> similar patents
python -m patent_search.cli --title "wheel hub" --cpc B60B           # title search within wheels
python -m patent_search.cli "trailer hitch lock" --from 2024-06-01 --to 2024-12-31 --json
python -m patent_search.cli -i                                        # REPL, model loaded once
python -m patent_search.cli --model models/minilm-patents "..."      # use the Part 3 fine-tuned model
```

### Timing: with and without hybrid filters

`python scripts/benchmark_hybrid.py --scale 1 10 30`

Each figure is the median of 5 runs × 8 queries, in ms of retrieval time. Query encoding (~5 ms) is the same in every case and is excluded.
- **post-filter** is the naive baseline: score every passage, then walk the ranking and string-check each patent against the filter.
- **pre-filter** resolves the filter through the indexes first.
- `--scale N` replicates the corpus N times (same metadata, jittered embeddings) to show growth.
- Both strategies return identical top-10s in every case (checked by the script).

### scale x1: 640 patents, 32,298 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 640 | 0.78 | 0.76 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 298 | 0.81 | 0.82 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 20 | 0.09 | 1.00 | 11.1x | yes |
| kw=tire | 315 | 0.87 | 1.53 | 1.8x | yes |
| kw=wheel AND hub | 100 | 0.77 | 2.54 | 3.3x | yes |
| cpc=B60C + kw=tread | 150 | 0.79 | 1.08 | 1.4x | yes |
| title~'wheel' | 189 | 0.77 | 0.77 | 1.0x | yes |

### scale x10: 6,400 patents, 322,980 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 6,400 | 7.37 | 7.33 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 2,980 | 7.75 | 8.01 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 200 | 1.13 | 9.09 | 8.0x | yes |
| kw=tire | 3,150 | 7.10 | 8.61 | 1.2x | yes |
| kw=wheel AND hub | 1,000 | 7.25 | 11.61 | 1.6x | yes |
| cpc=B60C + kw=tread | 1,500 | 9.40 | 10.02 | 1.1x | yes |
| title~'wheel' | 1,890 | 8.31 | 7.53 | 0.9x | yes |

### scale x30: 19,200 patents, 968,940 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 19,200 | 23.36 | 22.19 | 1.0x | yes |
| cpc=B60B (wheels, ~45%) | 8,940 | 22.65 | 22.39 | 1.0x | yes |
| cpc=B60D (couplings, ~3%) | 600 | 5.08 | 25.24 | 5.0x | yes |
| kw=tire | 9,450 | 21.87 | 25.33 | 1.2x | yes |
| kw=wheel AND hub | 3,000 | 21.39 | 29.86 | 1.4x | yes |
| cpc=B60C + kw=tread | 4,500 | 22.73 | 23.13 | 1.0x | yes |
| title~'wheel' | 5,670 | 22.46 | 22.24 | 1.0x | yes |

### Real data: large-dataset sample, 25,182 patents, 747,834 passages, mode=dense
| filter | candidates | pre-filter ms | post-filter ms | speed-up | same top-10 |
|---|---:|---:|---:|---:|:---:|
| none | 25,182 | 18.01 | 18.00 | 1.0x | yes |
| cpc=G06F (computing, ~5%) | 1,324 | 7.57 | 19.21 | 2.5x | yes |
| cpc=H01M (batteries, ~1%) | 258 | 0.83 | 20.85 | 25.1x | yes |
| cpc=B60 (vehicles, ~2%) | 439 | 2.29 | 19.96 | 8.7x | yes |
| cpc=H (electricity section) | 4,112 | 17.22 | 18.80 | 1.1x | yes |
| kw=battery | 449 | 2.41 | 38.09 | 15.8x | yes |
| kw=wireless AND network | 241 | 0.95 | 65.46 | 68.9x | yes |
| cpc=G06F + kw=memory | 220 | 1.02 | 23.23 | 22.7x | yes |
| title~'vehicle' | 660 | 3.82 | 19.04 | 5.0x | yes |
| year 2015-2017 | 3,594 | 16.65 | 18.16 | 1.1x | yes |

(`python scripts/benchmark_hybrid.py --profile large --data-dir data/patent_data_large_sample --max-paras 10`; general-technology queries and filters, since this data covers every CPC section. Index load from cache: 33 s to parse the patents + 6 s to build the indexes.)

**Commentary**

* **On real data the filters pay off even more:** up to **69×** faster for `wireless AND network`, 25× for the battery class H01M, and 23× for `G06F + memory`. Real CPC classes and keywords are far more selective than the vehicle-only sample, and the naive post-filter gets *slower* than no filter (up to 65 ms), because it has to walk deep into the ranking to find 10 survivors.
* **With no filter**, cost grows linearly with passages: 0.75 → 7.3 → 23 ms for 32k → 323k → 969k passages. That is one matmul. At 10M patents (~300–500M passages) exact search is ~10 s per query, so production needs ANN (HNSW / IVF-PQ). See Part 2.
* **Selective filters make search faster, not slower.** A CPC class covering 3% of patents is 5–11× faster than the naive approach, because the matmul only touches the candidates' rows. The naive post-filter is always slightly *slower* than no filter at all: it pays for the full scoring plus a string check per scanned result, and with a selective filter it has to walk deep into the ranking to find 10 survivors.
* **Broad filters.** Gathering rows is a memory copy that costs more than multiplying the contiguous matrix. My first version always gathered and was 3× *slower* than no filter at x10 scale. The engine now switches strategy by selectivity. I first set the threshold at 15%. On the real 25k-patent data, `year 2015-2017` (14% of patents) was still 0.8× (23.4 vs 18.8 ms; [before](results/benchmark_hybrid_large_threshold15.md)), so I lowered it to 10%, giving 1.1×. Now no filter is slower than the baseline. Vector DBs do the same thing (Qdrant/Weaviate switch from filtered-HNSW to exact search below a cardinality threshold).
* **The benchmark also found a correctness bug.** The title filter originally used the word index, so "wheel" missed "WHEELS". I replaced it with a trigram index, which gives true substring semantics and is still indexed.
* **What else I would do:**
  - store embeddings as float16/int8 (2–4× less memory bandwidth, and brute force is bandwidth-bound);
  - keep a per-CPC-class contiguous layout, so a CPC filter is a slice, not a gather;
  - cache popular filter → candidate-set results as roaring bitmaps;
  - at scale, push the filters into the ANN index as payload filters (Part 2).

## Part 2 – implementation at scale

* **Design document:** [docs/SCALING.md](docs/SCALING.md). It covers:
  - components (S3, Postgres for metadata/FTS/jobs, Qdrant for vectors, GPU workers, a stateless API);
  - the ingest, query and blue-green re-index pipelines;
  - error handling and monitoring, including a nightly quality canary built on the Part 3 eval;
  - a cost breakdown: ≈ $2.7k/month steady state, under $200 one-off for embedding;
  - the hard problems.
* **Proof of concept:** [poc/ingest_queue.py](poc/ingest_queue.py), the ingest pipeline in miniature. SQLite stands in for Postgres and `.npy` shards for the vector DB. It shows:
  - atomic job claiming with leases, so a crashed worker's job is picked up again;
  - retries, then dead-lettering after 3 attempts;
  - idempotent upserts keyed on content hash;
  - a status report;
  - a query path that resolves CPC and keyword filters in SQL (an FTS5 index) before touching vectors.

```bash
python poc/ingest_queue.py reset
python poc/ingest_queue.py enqueue data/patent_data_small
python poc/ingest_queue.py work --workers 3 --fail-rate 0.3   # 30% injected failures
python poc/ingest_queue.py status
python poc/ingest_queue.py retry-failed && python poc/ingest_queue.py work --workers 3
python poc/ingest_queue.py search "tire tread noise" --cpc B60C --keywords tread
```

Output from my run (3 workers, CPU embedding, 30% injected failures):

```
enqueued 64 new jobs
... 35 injected failures, retried automatically ...
[w1] job 23 attempt 3 FAILED -> failed: injected failure (simulated GPU OOM / network error)
[w1] job 24 attempt 3 FAILED -> failed: injected failure (simulated GPU OOM / network error)
jobs by status: {'done': 62, 'failed': 2}        # 2 jobs dead-lettered, error kept
$ retry-failed && work                            -> jobs by status: {'done': 64}
patents indexed: 640   passages/vectors: 32298    # same counts as the Part 1 engine
$ (re-run all 64 jobs)                            -> "10 patents (0 new/changed) in 0.0s" each: idempotent
$ search "tire tread pattern that reduces road noise" --cpc B60C --keywords tread
150 candidates from SQL in 0.5 ms, vector scoring 48.4 ms
  0.740 20250001807 B60C1103FI   TIRE WITH BI-DIRECTIONAL PERFORMANCE
  0.723 20240131866 B60C304FI    PNEUMATIC TIRE
```
(The vector scoring is slow because the POC memory-maps a float16 shard file per job. The production design keeps the vectors in Qdrant.)

**On the large dataset it found a real bug.** I ran it on 4 real 1,000-patent batch files (`enqueue ~/Downloads/coding_challenge_data --max-files 4`, `POC_MAX_PARAS=10`, 2 workers on the GPU):
- A real batch takes 150–185 s to embed, longer than the 120 s lease. The lease expired while the first worker was still busy, so a second worker claimed the same job: **2 of 4 jobs were processed twice** ([log](results/poc_large_run_before_heartbeat.txt)).
- Idempotent upserts kept the data correct (4,000 patents, no duplicates), but GPU time was wasted. On the small data each job took 3 s, so this never showed up there.
- **Fix:**
  - a **heartbeat** thread extends the lease every 30 s while a job runs;
  - **fencing**: a worker may only mark a job done or failed while it still owns it, otherwise its writes are rolled back.
- After the fix every job ran exactly once. Wall time went from 434 s to **327 s**, and throughput from 7.0 to **12.4 patents/s (+77%)** ([log](results/poc_large_run.txt)).

## Part 3 – evaluation & training (enhancement chosen)

**Why this one:** the engine is only as good as its embeddings, and an off-the-shelf
MiniLM has never seen patent claim language. You also cannot improve what you cannot measure.
So I built the measurement first, then the training loop that it measures.

**Task: claim → source patent.** This is a proxy for claim mapping
([patent_search/training.py](patent_search/training.py)).

* **Index:** all 640 patents, represented only by title+abstract and description paragraphs. **Claims are removed**, otherwise a claim would find itself.
* **Queries:** claims of 20% held-out patents. The split is by patent, so no test patent is ever used in training. Independent claims are used as-is. Dependent claims have their `"according to claim N"` reference stripped, so they read as standalone features. They are harder, shorter and more generic, and are reported separately.
* **Metrics:** Recall@1/5/10 and MRR@10 for retrieval. Pair metrics on frozen held-out triplets: mean cos(claim, positive), mean cos(claim, hard negative), the margin between them, and triplet accuracy.
* **Training data** comes from the train-split patents only:
  - *positive* = the paragraph of the same patent that best supports the claim. It is chosen by BM25, so it is independent of the model being trained.
  - *hard negative* = the best passage of a highly ranked **different** patent in the **same CPC class**, which looks related but isn't. The rank-1 candidate is skipped, because the top hit may genuinely be prior art (a false negative).
  - Loss: `MultipleNegativesRankingLoss` (in-batch negatives + the mined hard negative), with a no-duplicates batch sampler.

```bash
python scripts/train_eval.py            # evaluates BM25 + base, mines triplets, fine-tunes, re-evaluates
# -> results/eval_results.md / .json, model saved to models/minilm-patents
```

### Results (seed 13, 1 epoch, Apple M-series / MPS)

Test set: 357 claims from 128 held-out patents, searched against all 640 patents (claims removed from index).

| model | R@1 | R@10 | MRR@10 | MRR@10 indep. | MRR@10 dep. |
|---|---:|---:|---:|---:|---:|
| BM25 (reference) | 0.759 | 0.874 | 0.802 | 0.901 | 0.748 |
| all-MiniLM-L6-v2 (base) | 0.658 | 0.852 | 0.724 | 0.817 | 0.673 |
| fine-tuned (1 ep, 2292 triplets) | 0.810 | 0.910 | 0.844 | 0.947 | 0.787 |

| model | cos(claim, pos) | cos(claim, hard neg) | margin | triplet acc |
|---|---:|---:|---:|---:|
| base | 0.676 | 0.649 | 0.027 | 0.594 |
| fine-tuned | 0.660 | 0.415 | 0.244 | 0.874 |

Per-query: 96 queries ranked their patent higher after tuning, 34 lower. Training took 126s.

**What this shows**
* One epoch on 2.3k mined triplets (~2 min on a laptop) raises **MRR@10 from 0.724 to 0.844 (+12 pts)** and Recall@1 from 0.658 to 0.810. The gain holds for both independent and dependent claims, and 96 queries improved against 34 that got worse.
* The pair metrics show *why*. Before tuning, the base model scores a same-CPC hard negative almost as high as the true supporting paragraph (margin 0.027; triplet accuracy 0.59, close to a coin flip). After tuning the margin is 0.244 and the accuracy 0.87. The model learned to separate "same technical area" from "same invention", which is exactly the distinction prior-art search needs.
* **BM25 is a strong baseline here** (MRR 0.802, above the base MiniLM). A claim reuses its own description's exact terminology, so self-retrieval favours lexical matching. That is one reason the engine offers `rrf` fusion. The fine-tuned model is the first to beat BM25. Two caveats:
  - Test positives for the pair metrics were chosen by BM25, which may favour lexical overlap.
  - Real prior art is usually worded differently from the claim, so a dense model's advantage should be larger there than this self-retrieval task can show.

**Limitations / next steps**
* "Same patent" is a weak label. A better eval set would use examiner citations: the USPTO Office Action / PatentsView citation data gives real claim → prior-art pairs.
* The test set is small (357 queries), so differences of 1–2 points are within noise. With more time I would bootstrap confidence intervals and run several seeds.
* Next model steps: try a stronger base (bge-base / a patent-specific model such as PaECTER), use a larger corpus, and add a cross-encoder re-ranker on the top 100.
* **The large dataset.** The pipeline is ready for it: similarity scores are computed in query chunks, so a 10k × 275k matrix never has to fit in memory, and hard negatives fall back to any high-ranked patent when the CPC code is missing. It has been smoke-tested on 300 large-data patents:
  ```bash
  python scripts/train_eval.py --data-dir data/patent_data_large_sample --max-paras 10 \
      --max-test-patents 1000 --max-train-triplets 10000 --out models/minilm-patents-large --tag large
  ```
  I did **not** run it to completion. On a laptop it is about an hour (embedding 275k passages twice plus training), and it is beyond what Part 3 asks for. The reported Part 3 results are on the small dataset.

## Bonus: large dataset efficiency

`coding_challenge_data` has 471k records (30 GB of JSON) across every technology area. I measured each stage, then improved the slow ones:

| stage | before | after | how |
|---|---|---|---|
| scan + clean + dedup all 471k records | 119 s (1 process) | **28 s** (10 processes, 4.3×) | [scripts/prepare_large.py](scripts/prepare_large.py): process pool; records are hash-sampled *before* parsing, so the sample is reproducible and duplicate copies always fall on the same side |
| passages to embed for 25k patents | 3.4M (all paragraphs) | **765k** (−78%) | index abstract + claims + first 10 description paragraphs (`--max-paras 10`) |
| embedding 748k passages | — | 31.6 min (~395 passages/s on Apple MPS) | one-off, then cached in `index_cache/` (load 0.4 s) |
| BM25 query, 25k patents | 26.7 ms (`rank_bm25`) | **0.15 ms** (178×) | sparse-matrix BM25, identical top-10 ([results](results/bench_bm25_large.md)) |
| dense search with a selective filter | 18–65 ms (post-filter) | **0.8–7.6 ms** (up to 69×) | index-resolved pre-filtering + adaptive threshold (table above) |
| ingest jobs on 1,000-patent batches | 7.0 patents/s, jobs ran twice | **12.4 patents/s** | lease heartbeat + fencing (Part 2) |

**What the data looks like** ([results/prepare_large_w10.json](results/prepare_large_w10.json)), which also fed into [docs/SCALING.md](docs/SCALING.md):
- 457,385 unique patents; 2.9% of records are duplicate copies, and about 22% of the duplicated doc numbers have *differing* content;
- **38% have no classification code**, so a CPC filter silently hides a third of the corpus;
- 9% have no title and 1% no description;
- 0 corrupted files;
- about 135 passages per patent if every paragraph is indexed.

## Repository layout

```
patent_search/data.py       loading, cleaning, missing-field policy, claim dependency parsing
patent_search/engine.py     passage index, dense/BM25/RRF, hybrid pre-filtering
patent_search/bm25.py       sparse-matrix BM25
patent_search/cli.py        command-line search
patent_search/training.py   Part 3: eval task, triplet mining, fine-tuning
scripts/benchmark_hybrid.py Part 1 timing, with vs. without filters
scripts/train_eval.py       Part 3 end-to-end run
scripts/prepare_large.py    parallel scan/clean/dedup/sample of the large dataset
scripts/bench_bm25.py       rank_bm25 vs sparse BM25
docs/SCALING.md             Part 2 design doc
poc/ingest_queue.py         Part 2 proof of concept
results/                    saved eval + benchmark outputs
```
