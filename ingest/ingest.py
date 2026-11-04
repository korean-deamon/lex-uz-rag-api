"""
Chunk + embed + upsert scraped lex.uz documents into Qdrant.

Takes the JSON files produced by scraper/scraper.py (data/raw/*.json) and:
  1. splits each document's text into overlapping chunks
  2. embeds each chunk with a local multilingual sentence-transformer model
     (no API key needed, runs on-device, handles Uzbek/Russian/English fine)
  3. upserts into the Qdrant collection, carrying metadata (doc_id, title, url, scraped_at,
     chunk_index) on every point so answers can always be traced back to a source.

Can be called standalone (`python ingest/ingest.py`) to (re)index everything currently
in data/raw/, or imported and given a specific list of doc dicts (used by the Airflow DAG
right after a scrape, so only the new/changed docs get re-embedded instead of the whole
corpus every day).
"""
import os
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "lex_uz")

# multilingual-e5-base: much stronger retrieval quality than MiniLM for this kind of
# dense legal text where many articles use similar vocabulary -- MiniLM scored every
# candidate ~0.69-0.71 with no real separation and missed the right article outright.
# e5 models are trained specifically for retrieval and expect "query: "/"passage: "
# prefixes on the text being embedded (see embed_texts / embed_query in rag.py).
EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBED_DIM = 768

CHUNK_SIZE = 1200       # characters per chunk
CHUNK_OVERLAP = 200     # characters of overlap between consecutive chunks

_model = None
_qdrant_client = None


def embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def ensure_collection():
    client = qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
        )
        print(f"[ingest] created collection '{COLLECTION}'")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    # e5 models need a "passage: " prefix on indexed text (vs "query: " for searches)
    prefixed = [f"passage: {t}" for t in texts]
    vectors = embed_model().encode(prefixed, normalize_embeddings=True)
    return [v.tolist() for v in vectors]


def upsert_doc(doc: dict):
    """Chunk + embed + upsert a single scraped doc dict (as produced by scraper.py)."""
    chunks = chunk_text(doc["text"])
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc['doc_id']}-{i}")),
                vector=vec,
                payload={
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "scraped_at": doc["scraped_at"],
                    "chunk_index": i,
                    "text": chunk,
                },
            )
        )

    qdrant_client().upsert(collection_name=COLLECTION, points=points)
    return len(points)


def ingest_docs(docs: list[dict]):
    ensure_collection()
    total = 0
    for doc in docs:
        n = upsert_doc(doc)
        total += n
        print(f"[ingest] {doc['doc_id']}: {n} chunks upserted")
    print(f"[ingest] done. {total} chunks upserted across {len(docs)} docs.")


def ingest_all_raw():
    """Re-index everything currently sitting in data/raw/ (full rebuild)."""
    docs = []
    for path in sorted(RAW_DIR.glob("*.json")):
        docs.append(json.loads(path.read_text()))
    ingest_docs(docs)


if __name__ == "__main__":
    ingest_all_raw()
