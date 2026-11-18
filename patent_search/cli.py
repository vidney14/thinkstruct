"""Command-line interface for the Part 1 search engine.

Examples
--------
python -m patent_search.cli "wheel spoke made of carbon fibre"
python -m patent_search.cli "tire tread noise reduction" --cpc B60C --keywords tread noise --mode rrf
python -m patent_search.cli --patent 20240051333 -k 5            # prior-art style: patent -> similar patents
python -m patent_search.cli --title "wheel hub" --cpc B60B          # metadata-only style lookup
python -m patent_search.cli -i                                       # interactive REPL (model loaded once)
python -m patent_search.cli --data-dir data/patent_data_large_sample --max-paras 10 "lithium battery cooling" --cpc H01M
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import shlex
import sys
import textwrap
import time

from .data import DEFAULT_DATA_DIR
from .engine import DEFAULT_MODEL, Filters, build_engine, results_to_json


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hybrid semantic patent search")
    ap.add_argument("query", nargs="*", help="natural-language query or a claim")
    ap.add_argument("--patent", help="use a patent (doc number) as the query: finds similar patents")
    ap.add_argument("-k", "--top-k", type=int, default=10)
    ap.add_argument("--mode", choices=["dense", "bm25", "rrf"], default="dense")
    ap.add_argument("--cpc", nargs="+", help="classification prefix(es), e.g. B60B or B60C11")
    ap.add_argument("--keywords", nargs="+", help="keywords that must appear in title/abstract")
    ap.add_argument("--any", action="store_true", help="keywords: match ANY instead of ALL")
    ap.add_argument("--title", help="title substring (case-insensitive)")
    ap.add_argument("--title-exact", action="store_true", help="--title must equal the full title")
    ap.add_argument("--from", dest="date_from", type=dt.date.fromisoformat, help="filing week >= YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", type=dt.date.fromisoformat, help="filing week <= YYYY-MM-DD")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF name or path to a fine-tuned model")
    ap.add_argument("--max-paras", type=int, default=None,
                    help="index only the first N description paragraphs per patent (use 10 for the large data)")
    return ap


def run_query(engine, a) -> None:
    filters = Filters(cpc_prefixes=a.cpc, keywords=a.keywords, keyword_mode="any" if a.any else "all",
                      title=a.title, title_exact=a.title_exact, date_from=a.date_from, date_to=a.date_to)
    query = " ".join(a.query) or None
    if not query and not a.patent:
        # filter-only lookup: rank the filtered set by the title text (or a neutral query)
        query = a.title or " ".join(a.keywords or []) or "vehicle"
    t = time.perf_counter()
    res = engine.search(query, patent_id=a.patent, filters=filters, top_k=a.top_k, mode=a.mode)
    ms = (time.perf_counter() - t) * 1000
    if a.json:
        print(results_to_json(res))
        return
    print(f"\n{len(res)} results in {ms:.1f} ms  (mode={a.mode}, filters={'on' if filters.active() else 'off'})\n")
    for n, r in enumerate(res, 1):
        print(f"{n:>2}. [{r.score:.3f}] {r.doc_number}  {r.title}  ({r.classification}, week {r.week})")
        print(f"    matched {r.match_kind}: " + textwrap.shorten(r.match_text, 220))
        if r.matched_keywords:
            print(f"    keywords: {', '.join(r.matched_keywords)}")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    ap = make_parser()
    a = ap.parse_args(argv)
    engine = build_engine(a.data_dir, a.model, max_paras=a.max_paras)
    if not a.interactive:
        run_query(engine, a)
        return
    print("Interactive mode. Type a query with the same flags, e.g.:  brake disc cooling --cpc B60B -k 5   (Ctrl-D to quit)")
    for line in sys.stdin:
        try:
            run_query(engine, ap.parse_args(shlex.split(line)))
        except SystemExit:
            pass
        except Exception as e:  # keep the REPL alive
            print("error:", e)


if __name__ == "__main__":
    main()
