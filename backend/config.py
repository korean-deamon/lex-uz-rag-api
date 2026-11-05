"""
Central settings for the /v1 RAG API.

Everything the test spec needs to be able to compare runs across system changes
(`system` object in every response) is pinned here, so a version bump is one edit
in one file rather than a grep across the pipeline.
"""
import os
from dataclasses import dataclass
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
# is the point: the run's `system.llm_provider` and `llm_model` record which one
# produced it, so two runs on different providers stay distinguishable.
#
# Most providers speak the OpenAI chat-completions protocol, so they need no
# code of their own -- only a base URL. The table below is therefore a
# convenience, not a limit: `custom` reaches anything OpenAI-compatible that is
# not listed, including a provider that did not exist when this was written, and
# a locally served model.
#
#   auto       pick a provider from whichever key is present (see resolve())
#   anthropic  Claude -- native API, official SDK
#   gemini     Google Gemini -- native generativelanguage API
#   openai | groq | deepseek | openrouter | together | fireworks | mistral
#     | xai | cerebras | ollama        -- OpenAI-compatible
#   custom     OpenAI-compatible, with RAG_CUSTOM_BASE_URL / _API_KEY / _MODEL
#   offline    deterministic extractive answerer; no credential, no network
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "auto")

# name -> (base URL, env var holding the key, default model or "" if unknown)
#
# Models are left blank where guessing would be worse than asking: a wrong
# default fails with "model not found", which reads like a broken key. Where a
# provider has one obvious workhorse model it is filled in. Either way
# `python scripts/list_models.py <provider>` prints what the key can reach, and
# a pinned version beats an alias -- an alias changes the model under a fixed
# pipeline_version, which is what makes two runs incomparable.
OPENAI_COMPATIBLE: dict[str, tuple[str, str, str]] = {
    "openai":     ("https://api.openai.com/v1",            "OPENAI_API_KEY",     ""),
    "groq":       ("https://api.groq.com/openai/v1",        "GROQ_API_KEY",       ""),
    "deepseek":   ("https://api.deepseek.com/v1",           "DEEPSEEK_API_KEY",   "deepseek-chat"),
    "openrouter": ("https://openrouter.ai/api/v1",          "OPENROUTER_API_KEY", ""),
    "together":   ("https://api.together.xyz/v1",           "TOGETHER_API_KEY",   ""),
    "fireworks":  ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",  ""),
    "mistral":    ("https://api.mistral.ai/v1",             "MISTRAL_API_KEY",    ""),
    "xai":        ("https://api.x.ai/v1",                   "XAI_API_KEY",        ""),
    "cerebras":   ("https://api.cerebras.ai/v1",            "CEREBRAS_API_KEY",   ""),
    # Local servers need no key; both expose the OpenAI protocol.
    "ollama":     ("http://localhost:11434/v1",             "OLLAMA_API_KEY",     ""),
    "lmstudio":   ("http://localhost:1234/v1",              "LMSTUDIO_API_KEY",   ""),
}

ANTHROPIC_MODEL = os.getenv("RAG_ANTHROPIC_MODEL", "claude-haiku-5-5")
GEMINI_MODEL = os.getenv("RAG_GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = os.getenv(
    "RAG_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

# Anything OpenAI-compatible that the table does not name.
CUSTOM_BASE_URL = os.getenv("RAG_CUSTOM_BASE_URL", "")
CUSTOM_API_KEY = os.getenv("RAG_CUSTOM_API_KEY") or os.getenv("RAG_API_KEY")
CUSTOM_MODEL = os.getenv("RAG_CUSTOM_MODEL", "")

# Presets that serve a local model and so need no credential.
LOCAL_PRESETS = ("ollama", "lmstudio")

# Order `auto` searches. Native APIs first -- a key for one of those cannot be
# used through the OpenAI protocol, so detecting it correctly matters most.
#
# The local presets are deliberately excluded: having no key, they always
# "resolve", so auto would pick Ollama on localhost even with nothing running
# there and fail every question with a connection error instead of falling back
# to the offline generator. A local server has to be named outright.
AUTO_ORDER = ("anthropic", "gemini", "custom") + tuple(
    name for name in OPENAI_COMPATIBLE if name not in LOCAL_PRESETS
)

LLM_MAX_TOKENS = int(os.getenv("RAG_LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT_S = float(os.getenv("RAG_LLM_TIMEOUT_S", "60"))

# Per-question retry for a *retryable* generation failure -- almost always the
# provider's own 429. Without this, a free-tier per-minute quota turns a
# 100-question run into dozens of LLM_ERROR results that say nothing about the
# system being evaluated. The budget caps how long one question may spend
# waiting, so a rate-limited provider cannot stall a batch indefinitely; past
# the batch's sync deadline the 202 async path takes over, which is what it is
# there for.
# 45s because free tiers ask for real waits: Groq's free tier answers a
# tokens-per-minute 429 with Retry-After around 33s, and a 25s budget gave up
# just short of it, turning a survivable rate limit into a scored failure.
GEN_MAX_RETRIES = int(os.getenv("RAG_GEN_MAX_RETRIES", "2"))
GEN_RETRY_BUDGET_S = float(os.getenv("RAG_GEN_RETRY_BUDGET_S", "45"))

# Minimum gap between generation calls. 0 disables it. Set this to stay under a
# provider's requests-per-minute limit instead of bursting and retrying: on a
# 30 req/min key, RAG_GEN_MIN_INTERVAL_S=2 keeps a 100-question run inside the
# quota from the start.
GEN_MIN_INTERVAL_S = float(os.getenv("RAG_GEN_MIN_INTERVAL_S", "0"))


@dataclass(frozen=True)
class Provider:
    """Everything the generator needs, resolved from the environment.

    `kind` is what decides the code path, and only three exist: the two native
    APIs, and the OpenAI chat-completions protocol that everything else speaks.
    Adding a provider that speaks that protocol is a table entry, not code.
    """
    name: str                 # "deepseek", "groq", "custom", ...
    kind: str                 # "openai_compatible" | "anthropic" | "gemini" | "offline"
    model: str
    base_url: str = ""
    api_key: str | None = None
    key_var: str = ""
    detected: bool = False    # chosen by `auto` rather than named outright


def _provider_for(name: str) -> Provider | None:
    """Build a Provider for `name`, or None if its credential is absent."""
    if name == "offline":
        return Provider("offline", "offline", "offline-extractive-v1")

    if name == "anthropic":
        # The SDK resolves credentials itself (env var, or an `ant auth login`
        # profile), so only an explicit key is checked here.
        if not os.getenv("ANTHROPIC_API_KEY"):
            return None
        return Provider("anthropic", "anthropic",
                        os.getenv("RAG_LLM_MODEL") or ANTHROPIC_MODEL,
                        api_key=os.getenv("ANTHROPIC_API_KEY"),
                        key_var="ANTHROPIC_API_KEY")

    if name == "gemini":
        if not GEMINI_API_KEY:
            return None
        return Provider("gemini", "gemini", GEMINI_MODEL,
                        base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY,
                        key_var="GEMINI_API_KEY")

    if name == "custom":
        if not CUSTOM_BASE_URL:
            return None
        return Provider("custom", "openai_compatible", CUSTOM_MODEL,
                        base_url=CUSTOM_BASE_URL, api_key=CUSTOM_API_KEY,
                        key_var="RAG_CUSTOM_API_KEY")

    preset = OPENAI_COMPATIBLE.get(name)
    if preset is None:
        return None
    default_url, key_var, default_model = preset
    api_key = os.getenv(key_var)
    base_url = os.getenv(f"RAG_{name.upper()}_BASE_URL", default_url)
    # A local server legitimately has no key.
    local = base_url.startswith(("http://localhost", "http://127.0.0.1"))
    if not api_key and not local:
        return None
    model = os.getenv(f"RAG_{name.upper()}_MODEL", default_model)
    return Provider(name, "openai_compatible", model, base_url=base_url,
                    api_key=api_key or "not-needed", key_var=key_var)


def resolve() -> Provider:
    """The active provider.

    With RAG_LLM_BACKEND=auto (the default) the first provider in AUTO_ORDER
    whose key is present wins. That is what makes "whatever key the tester has"
    work without editing anything but .env: drop in DEEPSEEK_API_KEY and
    DeepSeek is used; drop in OPENAI_API_KEY and OpenAI is. With nothing set it
    falls back to the offline generator rather than failing to start, so the
    service is always runnable.

    Naming a backend explicitly skips detection, and is the right thing for a
    scored run -- it fails loudly if that provider's key is missing instead of
    silently answering with a different model than the one the run claims.
    """
    requested = (LLM_BACKEND or "auto").strip().lower()

    if requested != "auto":
        found = _provider_for(requested)
        if found is not None:
            return found
        # Named but unusable: report it as itself so the error names the right
        # key, rather than silently falling back to another provider.
        if requested == "custom":
            return Provider("custom", "openai_compatible", CUSTOM_MODEL,
                            base_url=CUSTOM_BASE_URL, api_key=None,
                            key_var="RAG_CUSTOM_BASE_URL")
        if requested in OPENAI_COMPATIBLE:
            url, key_var, model = OPENAI_COMPATIBLE[requested]
            return Provider(requested, "openai_compatible",
                            os.getenv(f"RAG_{requested.upper()}_MODEL", model),
                            base_url=url, api_key=None, key_var=key_var)
        if requested == "gemini":
            return Provider("gemini", "gemini", GEMINI_MODEL,
                            base_url=GEMINI_BASE_URL, api_key=None,
                            key_var="GEMINI_API_KEY")
        if requested == "anthropic":
            return Provider("anthropic", "anthropic", ANTHROPIC_MODEL,
                            api_key=None, key_var="ANTHROPIC_API_KEY")
        return Provider(requested, "unknown", requested)

    for name in AUTO_ORDER:
        found = _provider_for(name)
        if found is not None:
            return Provider(**{**found.__dict__, "detected": True})
    return Provider("offline", "offline", "offline-extractive-v1", detected=True)


def llm_model() -> str:
    """The model id for the active provider, as reported in `system.llm_model`."""
    return resolve().model or "(model not set)"


def llm_provider() -> str:
    return resolve().name


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
