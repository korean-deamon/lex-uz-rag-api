"""
Test setup. The environment is configured before any backend import, because
backend.config reads it at import time.

The offline generator is forced so the contract is testable without a model
credential and without spending money on every test run. The tests that
genuinely need a model are marked `needs_llm` and skipped unless
RAG_TEST_LLM_BACKEND names a real provider.
"""
import os
import tempfile
from pathlib import Path

# Overridden, not defaulted. setdefault left the ambient environment in charge,
# so a .env -- or a container started from one -- that points at a real provider
# silently turned the contract suite into a paid, paced run against that
# provider: in the Docker image, where .env sets groq and a 20s interval, a
# 30-second suite does not finish. The contract is about HTTP behaviour and the
# offline generator exercises all of it. Opt in explicitly to use a real model.
_TEST_BACKEND = os.getenv("RAG_TEST_LLM_BACKEND", "offline")
os.environ["RAG_LLM_BACKEND"] = _TEST_BACKEND
if _TEST_BACKEND == "offline":
    os.environ["RAG_GEN_MIN_INTERVAL_S"] = "0"
# A per-run store, so tests never read or write the real batch history.
_STORE_DIR = tempfile.mkdtemp(prefix="rag-test-store-")
os.environ["RAG_STORE_PATH"] = str(Path(_STORE_DIR) / "batches.sqlite3")
# The rate limiter is exercised by its own test; elsewhere it must not reject
# the rapid-fire requests a test suite makes.
os.environ.setdefault("RAG_RATE_LIMIT_INTERVAL_S", "0")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend import config as rag_config  # noqa: E402
from backend.main import app  # noqa: E402

API_KEY = rag_config.API_KEYS[0]
AUTH = {"Authorization": f"Bearer {API_KEY}"}


# pytest's hook signatures fix these argument names, hence `config` here meaning
# pytest's config, not backend.config (imported above as rag_config).
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_llm: requires RAG_LLM_BACKEND=anthropic and a working API key"
    )


def pytest_collection_modifyitems(config, items):
    if rag_config.resolve().kind != "offline":
        return
    skip = pytest.mark.skip(reason="needs a real provider key (semantic judgement)")
    for item in items:
        if "needs_llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def client():
    # The context manager form runs lifespan, which warms the index.
    with TestClient(app) as c:
        yield c


_counter = {"n": 0}


@pytest.fixture
def batch_id():
    """A fresh batch_id per test, so idempotency never bleeds between tests."""
    _counter["n"] += 1
    return f"test-b{_counter['n']:03d}"


def make_batch(batch_id: str, questions, **options):
    """questions: list of (id, question) or (id, question, language)."""
    items = []
    for q in questions:
        item = {"id": q[0], "question": q[1]}
        if len(q) > 2:
            item["language"] = q[2]
        items.append(item)
    return {
        "run_id": "test-run",
        "batch_id": batch_id,
        "batch_index": 1,
        "total_batches": 1,
        "options": {"return_contexts": True, "return_debug": True, **options},
        "items": items,
    }
