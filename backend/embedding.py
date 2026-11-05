"""
The embedding model, loaded once and shared.

It lives here rather than in ingest/ because both halves of the project need
it and neither owns it: `ingest/ingest_corpus.py` embeds articles when building
the index, and `backend/retrieval.py` embeds the question at query time. They
must use the same model and the same prefixes or the vectors are not comparable
and retrieval silently degrades.

Previously retrieval imported it from `ingest.ingest`, which made the live /v1
pipeline depend on the older raw-scrape ingestion script for no reason other
than where the function happened to be written.

multilingual-e5-base was chosen over MiniLM because this corpus is dense legal
text where many articles share vocabulary: MiniLM scored every candidate around
0.69-0.71 with no real separation and missed the right article outright. e5
models are trained for retrieval and expect the "query: " / "passage: " prefixes
applied below -- dropping them costs real accuracy, which is why embedding goes
through these two functions rather than calling the model directly.
"""
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
DIM = 768

_model: SentenceTransformer | None = None


def embed_model() -> SentenceTransformer:
    """The shared model instance. Loads on first use (~1.1 GB, CPU is fine)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed text being *indexed*. e5 wants the "passage: " prefix here."""
    vectors = embed_model().encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    """Embed a *search query*. e5 wants "query: " -- not the passage prefix."""
    vector = embed_model().encode([f"query: {text}"], normalize_embeddings=True)[0]
    return vector.tolist()
