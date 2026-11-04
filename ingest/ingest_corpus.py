"""
Ingest the Uzbek Legal Corpus (25 codes, 7k+ articles, CC0, from
github.com/javohirmat/uzbek-legal-corpus) into Qdrant.

This is the "tree" layer the project needs: every chunk is ONE article from ONE
named code (Konstitutsiya, Soliq kodeksi, Mehnat kodeksi, Jinoyat kodeksi, ...),
carrying:
  - code            machine key, e.g. "soliq_kodeksi"
  - code_title      human title, e.g. "Oʻzbekiston Respublikasining Soliq kodeksi"
  - article_id      e.g. "480.1"
  - article_display e.g. "480¹-modda"
  - section / chapter  where in the code this article sits
  - lex_uz_doc      the lex.uz document id this code was sourced from
  - url             constructed lex.uz link

This lets the chatbot answer "which kodeks, which modda" precisely instead of just
"some chunk of text", and lets retrieval/browsing be filtered by code (the tree).

Run: python ingest/ingest_corpus.py
"""
import json
from pathlib import Path

from ingest.ingest import embed_texts, ensure_collection, qdrant_client, COLLECTION
from qdrant_client.http import models as qmodels
import uuid

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus" / "uzbek-legal-corpus" / "data" / "articles"

BATCH_SIZE = 64  # articles embedded per batch


def iter_records():
    for path in sorted(CORPUS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def record_to_point(rec: dict, vector: list[float]):
    lex_doc = rec.get("lex_uz_doc") or ""
    url = f"https://lex.uz/uz/docs/{lex_doc}" if lex_doc else ""
    article_display = rec.get("article_display", "")
    title = rec.get("title", "")
    header = f"{rec['code_title']}, {article_display}" + (f" ({title})" if title else "")
    text = f"{header}\n{rec.get('text', '')}"

    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"corpus-{rec['code']}-{rec.get('seq')}"))
    return qmodels.PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "source": "uzbek_legal_corpus",
            "code": rec["code"],
            "code_title": rec["code_title"],
            "article_id": rec.get("article_id"),
            "article_display": article_display,
            "section": rec.get("section"),
            "chapter": rec.get("chapter"),
            "lex_uz_doc": lex_doc,
            "url": url,
            "title": header,
            "text": text,
        },
    )


def ingest_corpus():
    ensure_collection()
    client = qdrant_client()

    progress_file = ROOT / "ingest" / "corpus_progress.json"
    done_seen = set()
    if progress_file.exists():
        done_seen = set(json.loads(progress_file.read_text()))
        print(f"[ingest_corpus] resuming, {len(done_seen)} batches already done")

    batch_recs = []
    total = 0
    batch_num = 0
    for rec in iter_records():
        text = rec.get("text", "").strip()
        if not text:
            continue
        batch_recs.append(rec)
        if len(batch_recs) >= BATCH_SIZE:
            batch_num += 1
            key = f"{batch_recs[0]['code']}:{batch_recs[0]['seq']}"
            if key not in done_seen:
                _flush(client, batch_recs)
                done_seen.add(key)
                progress_file.write_text(json.dumps(sorted(done_seen)))
            total += len(batch_recs)
            print(f"[ingest_corpus] {total} articles processed so far...")
            batch_recs = []

    if batch_recs:
        key = f"{batch_recs[0]['code']}:{batch_recs[0]['seq']}"
        if key not in done_seen:
            _flush(client, batch_recs)
        total += len(batch_recs)

    print(f"[ingest_corpus] done. {total} articles upserted total across 25 codes.")
    progress_file.unlink(missing_ok=True)


def _flush(client, recs: list[dict]):
    texts = [f"{r['code_title']}, {r.get('article_display','')}\n{r.get('text','')}" for r in recs]
    vectors = embed_texts(texts)
    points = [record_to_point(r, v) for r, v in zip(recs, vectors)]
    client.upsert(collection_name=COLLECTION, points=points)


if __name__ == "__main__":
    ingest_corpus()
