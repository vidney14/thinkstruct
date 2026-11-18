"""Loading and cleaning of the patents_ipa{DATE}.json files.

Missing-field policy (documented in README):
  * A patent is EXCLUDED only if it has no doc_number (we cannot identify or
    de-duplicate it) or if it has no searchable text at all (no title, no
    abstract, no claims and no description).
  * Any other missing field is KEPT and handled separately: missing text fields
    become empty strings / lists (they simply contribute no chunks), and a
    missing classification becomes "" which never matches a CPC-prefix filter
    (but the patent is still returned when no CPC filter is set).
  * Duplicated doc_numbers (same application in two weekly files) keep the
    first occurrence.
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
        """Normalised classification, e.g. 'B60B104FI' -> 'B60B'+'104' (kept raw, upper)."""
        return self.classification.upper()

    def independent_claims(self) -> list[str]:
        return [c for c, d in zip(self.claims, self.dependent) if not d]

    def dependent_claims(self) -> list[str]:
        return [c for c, d in zip(self.claims, self.dependent) if d]


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


def load_patents(data_dir: str = DEFAULT_DATA_DIR) -> list[Patent]:
    files = sorted(glob.glob(os.path.join(data_dir, "patents_ipa*.json")))
    if not files:
        raise FileNotFoundError(
            f"No patents_ipa*.json files found in {os.path.abspath(data_dir)}. "
            "Set PATENT_DATA_DIR or pass --data-dir."
        )
    seen: set[str] = set()
    out: list[Patent] = []
    stats = {"no_id": 0, "no_text": 0, "dup": 0, "missing_cls": 0, "missing_fields": 0}
    for f in files:
        week = _week_from_filename(f)
        with open(f) as fh:
            records = json.load(fh)
        for r in records:
            doc = str(r.get("doc_number") or "").strip()
            if not doc:
                stats["no_id"] += 1
                continue
            if doc in seen:
                stats["dup"] += 1
                continue
            claims = [clean_claim(c) for c in (r.get("claims") or []) if c and c.strip()]
            desc = [re.sub(r"\s+", " ", p).strip() for p in (r.get("detailed_description") or []) if p and p.strip()]
            title = (r.get("title") or "").strip()
            abstract = re.sub(r"\s+", " ", r.get("abstract") or "").strip()
            if not (title or abstract or claims or desc):
                stats["no_text"] += 1
                continue
            if not all([title, abstract, claims, desc, r.get("classification")]):
                stats["missing_fields"] += 1
            cls = (r.get("classification") or "").strip()
            if not cls:
                stats["missing_cls"] += 1
            # claim 1 never references another claim, even if the regex is fooled
            dep = [i > 0 and is_dependent(c) for i, c in enumerate(claims)]
            seen.add(doc)
            out.append(
                Patent(doc, title, abstract, claims, desc, cls, week,
                       r.get("filename") or "", r.get("bibtex") or "", dep)
            )
    log.info("Loaded %d patents from %d files; skipped/flagged: %s", len(out), len(files), stats)
    return out
