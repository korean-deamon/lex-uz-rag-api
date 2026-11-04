"""
Central settings for the /v1 RAG API.

Everything the test spec needs to be able to compare runs across system changes
(`system` object in every response) is pinned here, so a version bump is one edit
in one file rather than a grep across the pipeline.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --- Qdrant / index ---------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "lex_uz")

# --- versions reported in response.system -----------------------------------
# PIPELINE_VERSION must change whenever retrieval or prompting changes, otherwise
# two runs with different behaviour look identical in the comparison.
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION", "1.0.0")
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_NAME = "hybrid-bm25-dense-v1"
INDEX_VERSION = os.getenv("INDEX_VERSION", "lex-idx-2026-07-corpus-25")
# The Uzbek Legal Corpus snapshot the index was built from (manifest.csv: 2026-07).
LEX_SNAPSHOT_DATE = os.getenv("LEX_SNAPSHOT_DATE", "2026-07-01")

# --- generation -------------------------------------------------------------
# Which provider answers questions. Retrieval is identical for all of them, so
# switching providers changes only the generation half of the pipeline -- which
# is the point: the run's `system.llm_model` records which one produced it, so
# two runs on different providers stay comparable.
#   anthropic  Claude via the official SDK
#   gemini     Google Gemini via the generativelanguage REST API
#   groq       Groq via its OpenAI-compatible REST API
#   offline    deterministic extractive answerer; no credential, no network
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "anthropic")

# Per-provider model id. Each provider names its models differently, so one
# variable per provider beats one shared variable that is wrong for three of
# them. Override whichever you use; `python scripts/list_models.py <provider>`
# prints what your key can actually reach.
ANTHROPIC_MODEL = os.getenv("RAG_ANTHROPIC_MODEL", "claude-haiku-5-5")
GEMINI_MODEL = os.getenv("RAG_GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL = os.getenv("RAG_GROQ_MODEL", "llama-3.3-70b-versatile")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_BASE_URL = os.getenv(
    "RAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)
GROQ_BASE_URL = os.getenv("RAG_GROQ_BASE_URL", "https://api.groq.com/openai/v1")

LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT_S = float(os.getenv("RAG_LLM_TIMEOUT_S", "60"))


def llm_model() -> str:
    """The model id for the active backend, as reported in `system.llm_model`."""
    return {
        "anthropic": ANTHROPIC_MODEL,
        "gemini": GEMINI_MODEL,
        "groq": GROQ_MODEL,
        "offline": "offline-extractive-v1",
    }.get(LLM_BACKEND, LLM_BACKEND)


# Kept so an existing RAG_LLM_MODEL in someone's .env still selects the Claude
# model, which is what it used to mean.
LLM_MODEL = os.getenv("RAG_LLM_MODEL", ANTHROPIC_MODEL)
if os.getenv("RAG_LLM_MODEL") and LLM_BACKEND == "anthropic":
    ANTHROPIC_MODEL = LLM_MODEL

# --- API behaviour ----------------------------------------------------------
# Test key. Deliberately separate from any production key, per spec section 2.
API_KEYS = [k.strip() for k in os.getenv("RAG_API_KEYS", "test-key-lex-2026").split(",") if k.strip()]

MAX_BATCH_ITEMS = 10
MAX_QUESTION_CHARS = 2000
DEFAULT_TOP_K = 5
# Spec: a sync batch must answer within 120s; past that we switch to async 202.
SYNC_DEADLINE_S = float(os.getenv("RAG_SYNC_DEADLINE_S", "120"))
# Spec: at least 1 batch/second must be accepted.
RATE_LIMIT_MIN_INTERVAL_S = float(os.getenv("RAG_RATE_LIMIT_INTERVAL_S", "1.0"))
# Spec: stored batch results are kept for at least 30 days.
RESULT_RETENTION_DAYS = int(os.getenv("RAG_RESULT_RETENTION_DAYS", "30"))

# How many candidates to pull before reranking down to top_k.
RETRIEVAL_CANDIDATES = int(os.getenv("RAG_RETRIEVAL_CANDIDATES", "40"))

STORE_PATH = Path(os.getenv("RAG_STORE_PATH", str(ROOT / "data" / "batches.sqlite3")))
CORPUS_MANIFEST = ROOT / "data" / "corpus" / "uzbek-legal-corpus" / "manifest.csv"
