"""
Container entrypoint: wait for Qdrant, make sure the index exists, then serve.

This is what turns setup into a single `docker compose up`. Without it a
reviewer has to start Qdrant, run the ingest script, notice when it finishes
and only then start uvicorn -- four steps, each of which can be done in the
wrong order, and the wrong order fails in a way that looks like a broken
service (an empty index answers every question with no_answer).

The index build is the slow part, ~85 minutes on CPU, so the work is skipped
whenever Qdrant already holds the articles. That is the normal case: the
collection lives on a named volume, so it survives `docker compose down` and
every later start is immediate.
"""
import os
import subprocess
import sys
import time

EXPECTED_ARTICLES = 7383
WAIT_TIMEOUT_S = 120


def log(message: str) -> None:
    print(f"[entrypoint] {message}", flush=True)


def wait_for_qdrant():
    """Block until Qdrant answers, or give up with a message that says why."""
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    last_error = None

    while time.monotonic() < deadline:
        try:
            client = QdrantClient(host=host, port=port, timeout=10)
            client.get_collections()
            log(f"Qdrant is up at {host}:{port}")
            return client
        except Exception as exc:          # noqa: BLE001 -- any failure means "not yet"
            last_error = exc
            time.sleep(2)

    log(f"Qdrant at {host}:{port} did not answer within {WAIT_TIMEOUT_S}s: {last_error}")
    log("The API cannot serve without it. Is the qdrant service running?")
    sys.exit(1)


def count_articles(client) -> int:
    """
    How many *article-level* points the served collection holds.

    Counted with the same source filter retrieval uses, not as a bare total:
    the collection can also hold raw scraped chunks, which /v1 excludes because
    they carry no article structure. Counting those too would let a half-built
    index pass this check -- 7000 articles plus 400 raw chunks looks complete
    and answers 400 questions' worth of nothing.
    """
    collection = os.getenv("QDRANT_COLLECTION", "lex_uz")
    try:
        from backend.retrieval import _corpus_filter

        return client.count(
            collection_name=collection,
            count_filter=_corpus_filter(),
            exact=True,
        ).count
    except Exception:                     # noqa: BLE001 -- not created yet
        return 0


def build_index():
    """Run the ingest script as a subprocess so its progress keeps streaming."""
    log("Building the index. This takes about 85 minutes on CPU, and only")
    log("happens once -- the collection is kept on a Docker volume. Progress")
    log("is printed every 64 articles; an interrupted build resumes.")
    result = subprocess.run(
        [sys.executable, "ingest/ingest_corpus.py"],
        cwd="/app",
    )
    if result.returncode != 0:
        log(f"Index build failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    client = wait_for_qdrant()
    count = count_articles(client)

    if count >= EXPECTED_ARTICLES:
        log(f"Index ready: {count} articles.")
    elif os.getenv("RAG_AUTO_INGEST", "1") not in ("1", "true", "yes"):
        log(f"Index holds {count} articles and RAG_AUTO_INGEST is off.")
        log("Serving anyway -- expect every question to come back no_answer.")
    else:
        if count:
            log(f"Index holds {count} of {EXPECTED_ARTICLES} articles -- resuming.")
        else:
            log("Index is empty.")
        build_index()
        log(f"Index ready: {count_articles(client)} articles.")

    port = os.getenv("PORT", "8000")
    log(f"Starting the API on port {port}. Health: GET /v1/health")
    os.execvp(
        "uvicorn",
        ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
