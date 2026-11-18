"""Loading and cleaning of the patents_ipa{DATE}.json files.

Missing-field policy (documented in README):
  * A patent is EXCLUDED only if it has no doc_number (we cannot identify or
    de-duplicate it) or if it has no searchable text at all (no title, no
    abstract, no claims and no description).
  * Any other missing field is KEPT and handled separately: missing text fields
    become empty strings / lists (they simply contribute no chunks), and a
    missing classification becomes "" which never matches a CPC-prefix filter
    (but the patent is still returned when no CPC filter is set).
  * Duplicated doc_numbers (same application in two files) keep the most
    complete copy (most non-empty fields; ties -> first seen).

Two input layouts are supported (auto-detected):
  * small: data_dir/patents_ipa{YYMMDD}.json - filing week taken from the name.
  * large: data_dir/job_*/patents_batch_*.json (or a prepared sample written by
    scripts/prepare_large.py). No week in the name -> only the publication year
    (first 4 digits of the US doc number) is known; date filters then work at
    year granularity. classification "N/A" is treated as missing; codes such as
    "a61b 5/097" are normalised to "A61B5/097" (upper case, no spaces) for prefix matching. Corrupted files
    are skipped (logged) and invalid records dropped; duplicates keep the most
    complete copy.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import os
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = os.environ.get(
    "PATENT_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data", "patent_data_small"),
)

_CLAIM_NUM = re.compile(r"^\s*\d+\s*\.\s*")
# "The spoke according to claim 1 , wherein" / "as claimed in claim 3" / "of claim 2"
_DEP_REF = re.compile(
    r"\b(according to|as claimed in|as recited in|as defined in|as set forth in|as in|of|in)\s+"
    r"(any one of |any of |one of )?claims?\s+\d+",
    re.I,
)
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Patent:
    doc_number: str
    title: str
    abstract: str
    claims: list[str]
    description: list[str]
    classification: str
    week: dt.date | None
    filename: str = ""
    bibtex: str = ""
    dependent: list[bool] = field(default_factory=list)  # per claim

    @property
    def cpc(self) -> str:
        """Normalised classification for prefix matching (see normalize_cpc)."""
        return normalize_cpc(self.classification)

    @property
    def year(self) -> int | None:
        if self.week:
            return self.week.year
        y = self.doc_number[:4]
        return int(y) if y.isdigit() and 1970 < int(y) < 2100 else None

    @property
    def date_ordinal(self) -> int:
        """Filing week if known, else Jan 1 of the publication year, else -1 (for date filters)."""
        if self.week:
            return self.week.toordinal()
        return dt.date(self.year, 1, 1).toordinal() if self.year else -1

    def completeness(self) -> int:
        return sum(bool(x) for x in (self.title, self.abstract, self.claims, self.description, self.classification))

    def independent_claims(self) -> list[str]:
        return [c for c, d in zip(self.claims, self.dependent) if not d]

    def dependent_claims(self) -> list[str]:
        return [c for c, d in zip(self.claims, self.dependent) if d]


def normalize_cpc(code: str) -> str:
    """Upper-case, no whitespace. The '/' is KEPT: dropping it would make A61B5/097 and
    A61B50/97 identical. Query 'A61B5/' = main group 5 only; 'A61B5' also matches A61B50..."""
    return re.sub(r"\s", "", code or "").upper()


def clean_claim(text: str) -> str:
    text = _CLAIM_NUM.sub("", text or "")
    text = re.sub(r"\s+([,.;:])", r"\1", text)  # "claim 1 ," -> "claim 1,"
    return re.sub(r"\s+", " ", text).strip()


def is_dependent(claim: str) -> bool:
    return bool(_DEP_REF.search(claim[:200]))


def strip_dependency(claim: str) -> str:
    """'The spoke according to claim 2, wherein the convex ...' -> 'The spoke wherein the convex ...'.
    Used to build training/eval queries that are not trivially linked by claim numbers."""
    return re.sub(r"\s+", " ", _DEP_REF.sub("", claim, count=1)).replace(" ,", ",").strip()


def _week_from_filename(path: str) -> dt.date | None:
    m = re.search(r"ipa(\d{6})", os.path.basename(path))
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%y%m%d").date()
    except ValueError:
        return None


def find_files(data_dir: str) -> list[str]:
    for pattern in ("patents_ipa*.json", "job_*/patents_batch_*.json", "patents_batch_*.json", "*.json"):
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))
        if files:
            return files
    return []


def parse_record(r, week: dt.date | None, stats: dict) -> Patent | None:
    """One raw JSON dict -> Patent, or None if it must be excluded (see module docstring)."""
    if not isinstance(r, dict):
        stats["bad_record"] = stats.get("bad_record", 0) + 1
        return None
    doc = str(r.get("doc_number") or "").strip()
    if not doc:
        stats["no_id"] += 1
        return None

    def strs(v):  # tolerate a string or junk where a list of strings is expected
        if isinstance(v, str):
            v = [v]
        return [x for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []

    claims = [clean_claim(c) for c in strs(r.get("claims"))]
    desc = [re.sub(r"\s+", " ", p).strip() for p in strs(r.get("detailed_description"))]
    title = (r.get("title") or "").strip() if isinstance(r.get("title"), str) else ""
    abstract = re.sub(r"\s+", " ", r.get("abstract") or "").strip() if isinstance(r.get("abstract"), str) else ""
    if not (title or abstract or claims or desc):
        stats["no_text"] += 1
        return None
    cls = r.get("classification") if isinstance(r.get("classification"), str) else ""
    cls = "" if cls.strip().upper() in ("N/A", "NA", "NONE") else cls.strip()
    if not cls:
        stats["missing_cls"] += 1
    if not all([title, abstract, claims, desc, cls]):
        stats["missing_fields"] += 1
    # claim 1 never references another claim, even if the regex is fooled
    dep = [i > 0 and is_dependent(c) for i, c in enumerate(claims)]
    return Patent(doc, title, abstract, claims, desc, cls, week,
                  r.get("filename") or "", r.get("bibtex") or "", dep)


def new_stats() -> dict:
    return {"no_id": 0, "no_text": 0, "dup": 0, "missing_cls": 0, "missing_fields": 0,
            "bad_record": 0, "bad_file": 0}


def load_file(path: str, stats: dict, keep=None) -> list[Patent]:
    """Parse one JSON file; a corrupted file is logged and skipped, not fatal.
    keep: optional predicate on doc_number (used for sampling before full parsing)."""
    try:
        with open(path) as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            raise ValueError("top-level JSON is not a list")
    except Exception as e:  # truncated / invalid JSON, wrong encoding, ...
        log.warning("skipping corrupted file %s: %s", path, e)
        stats["bad_file"] += 1
        return []
    week = _week_from_filename(path)
    out = []
    for r in records:
        if keep is not None and isinstance(r, dict) and not keep(str(r.get("doc_number") or "")):
            continue
        p = parse_record(r, week, stats)
        if p is not None:
            out.append(p)
    return out


def dedupe(patents: list[Patent], stats: dict) -> list[Patent]:
    """Keep one record per doc_number: the most complete one (ties -> first seen)."""
    best: dict[str, Patent] = {}
    for p in patents:
        q = best.get(p.doc_number)
        if q is None:
            best[p.doc_number] = p
            continue
        stats["dup"] += 1
        if p.completeness() > q.completeness():
            best[p.doc_number] = p
    return list(best.values())


def load_patents(data_dir: str = DEFAULT_DATA_DIR) -> list[Patent]:
    files = find_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No patent JSON files found in {os.path.abspath(data_dir)}. "
            "Set PATENT_DATA_DIR or pass --data-dir."
        )
    stats = new_stats()
    raw: list[Patent] = []
    for f in files:
        raw.extend(load_file(f, stats))
    out = dedupe(raw, stats)
    log.info("Loaded %d patents from %d files; skipped/flagged: %s", len(out), len(files), stats)
    return out
