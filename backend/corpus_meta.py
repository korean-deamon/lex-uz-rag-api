"""
Document-level metadata for citations: doc_id, doc_title, doc_type, url,
edition_date, doc_status.

The vector store holds article-level chunks, which carry the code key and the
lex.uz document number but nothing about the *document* as a whole. The corpus
manifest does, so this module joins the two: `code` -> everything a
`citations[]` entry needs under spec section 4.

Honesty note on two spec-mandated fields:
  edition_date  The corpus is a dated snapshot of the consolidated (in-force)
                text, not a per-article revision history. The only defensible
                edition marker it carries is the snapshot month, so that is what
                is reported -- the same value for every article of a code. Per
                -article amendment dates would need a lex.uz revision scrape.
  doc_status    All 25 documents in the corpus are in force, so this is
                constantly "amalda". A repealed document cannot currently be
                cited because none is indexed; GET /v1/meta says so explicitly
                rather than letting the constant look like a real check.
"""
import csv
from functools import lru_cache

from backend import config

# "Kuchini yo'qotgan" (repealed) is a value this field can legitimately take; it
# is listed here so the vocabulary is visible even though the current snapshot
# only ever produces IN_FORCE.
STATUS_IN_FORCE = "amalda"
STATUS_REPEALED = "kuchini yoʻqotgan"


def _doc_type(filename: str, code_title: str) -> str:
    if filename.startswith("konstitutsiya"):
        return "konstitutsiya"
    if filename.endswith("_qonuni.txt") or "Qonuni" in code_title:
        return "qonun"
    return "kodeks"


def _normalise_snapshot(snapshot: str) -> str:
    """manifest's '2026-07' -> a real date, since the spec types the field as date."""
    snapshot = (snapshot or "").strip()
    if len(snapshot) == 7:  # YYYY-MM
        return f"{snapshot}-01"
    return snapshot or config.LEX_SNAPSHOT_DATE


@lru_cache(maxsize=1)
def doc_meta_by_code() -> dict[str, dict]:
    """code key (e.g. 'soliq_kodeksi') -> document metadata, read once."""
    if not config.CORPUS_MANIFEST.exists():
        return {}

    meta: dict[str, dict] = {}
    with open(config.CORPUS_MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            filename = row["filename"]
            code = filename.removesuffix(".txt")
            lex_doc = (row.get("lex_uz_doc") or "").strip()
            meta[code] = {
                # lex.uz act numbers are written with a leading '-' in URLs; the
                # stable identity is the number itself, so doc_id carries the
                # digits and `url` keeps the exact lex.uz form.
                "doc_id": f"lex:{lex_doc.lstrip('-')}" if lex_doc else "",
                "doc_title": row["code_title"],
                "doc_type": _doc_type(filename, row["code_title"]),
                "url": f"https://lex.uz/uz/docs/{lex_doc}" if lex_doc else "",
                "edition_date": _normalise_snapshot(row.get("snapshot_date", "")),
                "adopted_date": (row.get("adopted") or "").strip(),
                "doc_status": STATUS_IN_FORCE,
                "articles_count": int(row["articles_count"]) if row.get("articles_count") else None,
            }
    return meta


def for_code(code: str | None) -> dict:
    """Metadata for one code, with empty-but-present fields if it is unknown."""
    if code:
        found = doc_meta_by_code().get(code)
        if found:
            return found
    return {
        "doc_id": "",
        "doc_title": "",
        "doc_type": "",
        "url": "",
        "edition_date": config.LEX_SNAPSHOT_DATE,
        "adopted_date": "",
        "doc_status": STATUS_IN_FORCE,
        "articles_count": None,
    }


def indexed_doc_ids() -> set[str]:
    """Every doc_id the index can legitimately cite -- used to catch fabricated refs."""
    return {m["doc_id"] for m in doc_meta_by_code().values() if m["doc_id"]}
