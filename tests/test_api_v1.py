"""
The specification's section 7 checklist, as executable tests.

Each test names the checklist line it covers, so "is the API ready for the
test?" is answered by running this file rather than by reading the code.
"""
import json

import pytest

from backend import config
from tests.conftest import AUTH, make_batch

UZ_LATN_Q = "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?"
RU_Q = "Какой общий срок исковой давности установлен Гражданским кодексом?"
UZ_CYRL_Q = "Жиноят кодексига кўра ўғрилик учун жавобгарлик қандай?"


def post_batch(client, body, expect=200):
    r = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert r.status_code == expect, f"expected {expect}, got {r.status_code}: {r.text[:400]}"
    return r


# -- "POST /v1/rag/batch accepts 1-10 questions; 400 if there are 11" --------
def test_accepts_one_question(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)])
    assert len(post_batch(client, body).json()["results"]) == 1


def test_accepts_ten_questions(client, batch_id):
    questions = [(f"q{i:03d}", UZ_LATN_Q) for i in range(1, 11)]
    assert len(post_batch(client, make_batch(batch_id, questions)).json()["results"]) == 10


def test_eleven_questions_rejected_with_400(client, batch_id):
    questions = [(f"q{i:03d}", UZ_LATN_Q) for i in range(1, 12)]
    r = post_batch(client, make_batch(batch_id, questions), expect=400)
    assert r.json()["error"]["code"] == "VALIDATION_FAILED"


def test_zero_questions_rejected_with_400(client, batch_id):
    post_batch(client, make_batch(batch_id, []), expect=400)


def test_duplicate_item_id_rejected_with_400(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q), ("q001", RU_Q)])
    r = post_batch(client, body, expect=400)
    assert "unique" in r.json()["error"]["message"].lower()


def test_empty_question_rejected_with_400(client, batch_id):
    post_batch(client, make_batch(batch_id, [("q001", "   ")]), expect=400)


def test_question_over_2000_chars_rejected_with_400(client, batch_id):
    post_batch(client, make_batch(batch_id, [("q001", "a" * 2001)]), expect=400)


# -- "every id appears once, and results count equals the question count" ---
def test_results_match_items_by_id_exactly(client, batch_id):
    questions = [("q001", UZ_LATN_Q), ("q002", RU_Q), ("q003", UZ_CYRL_Q)]
    data = post_batch(client, make_batch(batch_id, questions)).json()
    ids = [r["id"] for r in data["results"]]
    assert ids == ["q001", "q002", "q003"]
    assert len(ids) == len(set(ids)) == len(questions)


def test_summary_counts_add_up_to_total(client, batch_id):
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q), ("q002", RU_Q)])).json()
    s = data["summary"]
    assert s["total"] == 2
    assert s["ok"] + s["no_answer"] + s["error"] == s["total"]


# -- "citations carry lex.uz doc_id, url, article, edition_date, doc_status, quote" --
REQUIRED_CITATION_FIELDS = (
    "ref", "doc_id", "doc_title", "doc_type", "url", "article", "part",
    "edition_date", "doc_status", "chunk_id", "quote",
)


def test_citations_carry_every_required_field(client, batch_id):
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()
    result = data["results"][0]
    assert result["status"] == "ok", result
    assert result["citations"], "an answered question must cite at least one source"
    for citation in result["citations"]:
        for field in REQUIRED_CITATION_FIELDS:
            assert field in citation, f"citation missing {field}"
            assert citation[field] not in (None, ""), f"citation.{field} is empty"
        assert citation["doc_id"].startswith("lex:")
        assert citation["url"].startswith("https://lex.uz/")
        assert "modda" in citation["article"]


def test_citation_quote_is_verbatim_in_its_context(client, batch_id):
    """The faithfulness guarantee: a quote that is not in the cited context is
    dropped by the pipeline, so any quote that survives must be findable."""
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q), ("q002", RU_Q)])).json()
    for result in data["results"]:
        contexts = {c["chunk_id"]: c["text"] for c in result["contexts"]}
        for citation in result["citations"]:
            text = contexts[citation["chunk_id"]]
            assert citation["quote"] in text, (
                f"{citation['chunk_id']}: quote is not a verbatim substring of the context"
            )


# -- "every answer carries contexts (top-k, with text and scores)" ----------
def test_contexts_shape_and_scores(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)], top_k=5)
    result = post_batch(client, body).json()["results"][0]
    contexts = result["contexts"]
    assert 1 <= len(contexts) <= 5
    assert [c["rank"] for c in contexts] == list(range(1, len(contexts) + 1))
    for c in contexts:
        for field in ("rank", "chunk_id", "doc_id", "article", "text",
                      "retrieval_score", "rerank_score", "used_in_answer"):
            assert field in c, f"context missing {field}"
        assert c["text"].strip(), "a context with no text cannot support an answer"
        assert 0.0 <= c["retrieval_score"] <= 1.0
        assert 0.0 <= c["rerank_score"] <= 1.0
    # Reranked order means scores descend.
    scores = [c["rerank_score"] for c in contexts]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_honoured(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)], top_k=3)
    assert len(post_batch(client, body).json()["results"][0]["contexts"]) <= 3


def test_return_contexts_false_omits_them(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)])
    body["options"]["return_contexts"] = False
    assert post_batch(client, body).json()["results"][0]["contexts"] == []


# -- "every citations[].chunk_id is present in contexts" --------------------
def test_no_citation_outside_contexts(client, batch_id):
    questions = [("q001", UZ_LATN_Q), ("q002", RU_Q), ("q003", UZ_CYRL_Q)]
    data = post_batch(client, make_batch(batch_id, questions)).json()
    for result in data["results"]:
        context_ids = {c["chunk_id"] for c in result["contexts"]}
        for citation in result["citations"]:
            assert citation["chunk_id"] in context_ids, (
                f"{result['id']} cites {citation['chunk_id']}, which the system never retrieved"
            )


def test_answer_markers_resolve_to_citation_refs(client, batch_id):
    import re
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()
    result = data["results"][0]
    refs = {c["ref"] for c in result["citations"]}
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", result["answer"] or "")}
    assert markers <= refs, f"answer references {markers - refs}, which is not in citations"


# -- "an unanswerable question returns no_answer: true and invents nothing" --
def test_no_answer_mechanism_reports_no_citations(client, batch_id):
    """A question with no credible match must come back no_answer, with no
    answer text and no citations to show for it."""
    body = make_batch(batch_id, [("q001", "zzzqqq xyzzy plugh frobnicate")])
    result = post_batch(client, body).json()["results"][0]
    assert result["no_answer"] is True
    assert result["status"] == "no_answer"
    assert result["answer"] in (None, "")
    assert result["citations"] == []


@pytest.mark.needs_llm
def test_out_of_scope_question_is_refused(client, batch_id):
    """The semantic case: the question is well formed and retrieval finds
    plausible articles, but the law does not address it. Needs the model."""
    body = make_batch(
        batch_id,
        [("q001", "Marsda yer uchastkasini qanday roʻyxatdan oʻtkazish kerak?")],
    )
    result = post_batch(client, body).json()["results"][0]
    assert result["no_answer"] is True, result["answer"]


# -- "one question's error does not affect the others" ----------------------
def test_question_level_error_is_isolated(client, batch_id):
    questions = [("q001", UZ_LATN_Q), ("q002", RU_Q, "klingon"), ("q003", UZ_CYRL_Q)]
    data = post_batch(client, make_batch(batch_id, questions)).json()
    assert data["status"] == "completed", "one bad question must not fail the batch"
    by_id = {r["id"]: r for r in data["results"]}
    bad = by_id["q002"]
    assert bad["status"] == "error"
    assert bad["error"]["code"] == "INVALID_LANGUAGE"
    assert bad["error"]["retryable"] is False
    assert by_id["q001"]["status"] == "ok"
    assert by_id["q003"]["status"] == "ok"
    assert data["summary"]["error"] == 1


# -- spec section 5: every listed error code must be reachable --------------
def test_unreachable_index_reports_index_unavailable(client, monkeypatch, batch_id):
    """Section 5 lists INDEX_UNAVAILABLE and RETRIEVAL_FAILED separately, and
    the difference is the point: one says the vector store is down, the other
    that this question's search failed. Reporting an unreachable Qdrant as
    RETRIEVAL_FAILED sends the tester hunting through the question."""
    from backend import retrieval
    from qdrant_client.http.exceptions import ResponseHandlingException

    def unreachable(*args, **kwargs):
        raise ResponseHandlingException("[Errno 111] Connection refused")

    monkeypatch.setattr(retrieval, "retrieve", unreachable)
    result = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == "INDEX_UNAVAILABLE"
    assert result["error"]["retryable"] is True


def test_other_retrieval_failure_stays_retrieval_failed(client, monkeypatch, batch_id):
    from backend import retrieval

    def broken(*args, **kwargs):
        raise ValueError("embedding produced the wrong shape")

    monkeypatch.setattr(retrieval, "retrieve", broken)
    result = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()["results"][0]
    assert result["error"]["code"] == "RETRIEVAL_FAILED"


@pytest.mark.parametrize("error,expected", [
    (ValueError("bad shape"), False),
    (KeyError("article_id"), False),
    (ConnectionError("connection refused"), True),
    (RuntimeError("the request timed out"), True),
    (RuntimeError("failed to connect to host"), True),
])
def test_index_unavailable_classifier(error, expected):
    from backend import pipeline
    assert pipeline._is_index_unavailable(error) is expected


def test_whole_batch_failure_is_503_so_the_tester_retries(client, monkeypatch, batch_id):
    """A service error must come back as 5xx. Returning the stored failure body
    with HTTP 200 made it look like a result: run_eval.py would save a batch of
    zero answers and move on, turning a transient fault into a hole in the run.
    """
    from backend import pipeline

    def explode(*args, **kwargs):
        raise RuntimeError("worker died")

    monkeypatch.setattr(pipeline, "run_batch", explode)
    r = client.post("/v1/rag/batch", json=make_batch(batch_id, [("q001", UZ_LATN_Q)]),
                    headers=AUTH)
    assert r.status_code == 503, r.text[:300]
    body = r.json()
    assert body["status"] == "failed"
    assert body["batch_id"] == batch_id
    assert body["error"]["retryable"] is True

    # GET still serves the stored failed record -- section 5 provides for that.
    got = client.get(f"/v1/rag/batch/{batch_id}", headers=AUTH)
    assert got.status_code == 200
    assert got.json()["status"] == "failed"


# -- "system object: model, embedding, index version, lex.uz snapshot date" --
def test_system_block_is_complete(client, batch_id):
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()
    system = data["system"]
    for field in ("pipeline_version", "llm_model", "embedding_model", "reranker",
                  "index_version", "lex_snapshot_date"):
        assert system.get(field) not in (None, ""), f"system.{field} is missing"
    assert data["timing"]["received_at"] and data["timing"]["completed_at"]
    assert data["timing"]["total_ms"] >= 0


def test_per_question_timing_and_usage_present(client, batch_id):
    result = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()["results"][0]
    assert result["timing"]["total_ms"] >= 0
    assert "retrieval_ms" in result["timing"]
    assert "generation_ms" in result["timing"]
    assert set(result["usage"]) == {"input_tokens", "output_tokens"}


# -- "temperature=0 and seed: the same question returns the same sources" ---
def test_same_question_returns_identical_sources(client):
    def sources_for(bid):
        body = make_batch(bid, [("q001", UZ_LATN_Q)], temperature=0, seed=42)
        result = post_batch(client, body).json()["results"][0]
        return (
            [c["chunk_id"] for c in result["contexts"]],
            [c["chunk_id"] for c in result["citations"]],
            result["answer"],
        )

    first = sources_for("determinism-a")
    second = sources_for("determinism-b")
    assert first == second, "identical request produced different sources or answer"


# -- "resending the same batch_id returns the stored result" ----------------
def test_same_batch_id_replays_stored_result(client, batch_id):
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)])
    first = post_batch(client, body).json()
    again = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert again.status_code == 200
    assert again.headers.get("X-Idempotent-Replay") == "true"
    assert again.json() == first, "a replay must return the stored result unchanged"


def test_same_batch_id_different_questions_is_409(client, batch_id):
    post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)]))
    conflicting = make_batch(batch_id, [("q001", RU_Q)])
    r = client.post("/v1/rag/batch", json=conflicting, headers=AUTH)
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "BATCH_ID_CONFLICT"


# -- "Latin, Cyrillic and Russian all work; oʻ / gʻ are not corrupted" -----
@pytest.mark.parametrize(
    "question,expected_language",
    [(UZ_LATN_Q, "uz-Latn"), (UZ_CYRL_Q, "uz-Cyrl"), (RU_Q, "ru")],
)
def test_three_languages_are_detected_and_answered(client, question, expected_language):
    bid = f"lang-{expected_language}"
    data = post_batch(client, make_batch(bid, [("q001", question)])).json()
    result = data["results"][0]
    assert result["answer_language"] == expected_language
    assert result["status"] in ("ok", "no_answer")
    assert result["contexts"], "every language must retrieve something"


def test_uzbek_special_characters_survive_round_trip(client, batch_id):
    """oʻ / gʻ / ʼ must come back as sent, not mangled into o' or ?."""
    question = "Oʻzbekiston Respublikasining Konstitutsiyasiga koʻra davlat tili qaysi til?"
    body = make_batch(batch_id, [("q001", question)])
    r = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert r.status_code == 200
    assert "charset=utf-8" in r.headers["content-type"].lower().replace(" ", "")
    result = r.json()["results"][0]
    assert result["debug"]["rewritten_query"], "query rewriting must be reported"
    # The retrieved Uzbek text keeps its modifier letters.
    joined = " ".join(c["text"] for c in result["contexts"])
    assert "ʻ" in joined or "ʼ" in joined


# -- "batch <= 120s, or async 202 + GET works" -----------------------------
def test_sync_batch_completes_within_deadline(client, batch_id):
    data = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()
    assert data["status"] == "completed"
    assert data["timing"]["total_ms"] < config.SYNC_DEADLINE_S * 1000


def test_async_202_then_get_returns_full_result(client, monkeypatch, batch_id):
    """Forcing the sync deadline to zero must produce the spec's async path:
    202 + status processing, then the complete section-4 response from GET."""
    import time
    monkeypatch.setattr(config, "SYNC_DEADLINE_S", 0.001)
    body = make_batch(batch_id, [("q001", UZ_LATN_Q)])
    r = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert r.status_code == 202
    assert r.json()["status"] == "processing"
    assert r.json()["batch_id"] == batch_id

    monkeypatch.undo()
    for _ in range(300):
        got = client.get(f"/v1/rag/batch/{batch_id}", headers=AUTH)
        if got.status_code == 200:
            break
        assert got.status_code == 202
        time.sleep(0.1)
    else:
        pytest.fail("async batch never completed")

    data = got.json()
    assert data["status"] == "completed"
    assert len(data["results"]) == 1
    assert data["system"]["index_version"]


def test_get_batch_unknown_id_is_404(client):
    r = client.get("/v1/rag/batch/does-not-exist", headers=AUTH)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "BATCH_NOT_FOUND"


def test_get_batch_returns_same_body_as_post(client, batch_id):
    posted = post_batch(client, make_batch(batch_id, [("q001", UZ_LATN_Q)])).json()
    got = client.get(f"/v1/rag/batch/{batch_id}", headers=AUTH)
    assert got.status_code == 200
    assert got.json() == posted


# -- "GET /v1/meta and GET /v1/health work" --------------------------------
def test_meta_reports_system_index_and_documents(client):
    data = client.get("/v1/meta", headers=AUTH).json()
    assert data["system"]["index_version"]
    assert data["system"]["lex_snapshot_date"]
    assert data["index"]["articles_indexed"] > 0
    assert len(data["documents"]) >= 20
    for doc in data["documents"]:
        assert doc["doc_id"].startswith("lex:")
        assert doc["url"].startswith("https://lex.uz/")
    assert set(data["languages"]) == {"uz-Latn", "uz-Cyrl", "ru"}
    assert data["caveats"], "meta must state the index's known limits"


def test_health_is_public_and_reports_checks(client):
    r = client.get("/v1/health")  # deliberately no auth header
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["qdrant"] == "ok"
    assert body["checks"]["articles_indexed"] > 0


# -- "a separate API key is required" --------------------------------------
def test_missing_auth_header_is_401(client, batch_id):
    r = client.post("/v1/rag/batch", json=make_batch(batch_id, [("q001", UZ_LATN_Q)]))
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHENTICATED"


def test_malformed_auth_header_is_401(client, batch_id):
    r = client.post(
        "/v1/rag/batch",
        json=make_batch(batch_id, [("q001", UZ_LATN_Q)]),
        headers={"Authorization": config.API_KEYS[0]},  # no "Bearer "
    )
    assert r.status_code == 401


def test_unknown_api_key_is_403(client, batch_id):
    r = client.post(
        "/v1/rag/batch",
        json=make_batch(batch_id, [("q001", UZ_LATN_Q)]),
        headers={"Authorization": "Bearer not-a-real-key"},
    )
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN"


def test_meta_requires_auth(client):
    assert client.get("/v1/meta").status_code == 401


# -- rate limit: 429 + Retry-After -----------------------------------------
def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    """The spec requires at least 1 batch/second to be accepted and a
    Retry-After on rejection. With the interval raised, the second immediate
    request must be the one rejected."""
    from backend import store
    monkeypatch.setattr(store.rate_limiter, "min_interval_s", 60.0)
    monkeypatch.setattr(store.rate_limiter, "_last", {})

    first = client.post(
        "/v1/rag/batch", json=make_batch("rl-1", [("q001", UZ_LATN_Q)]), headers=AUTH
    )
    assert first.status_code == 200
    second = client.post(
        "/v1/rag/batch", json=make_batch("rl-2", [("q001", UZ_LATN_Q)]), headers=AUTH
    )
    assert second.status_code == 429
    assert int(second.headers["Retry-After"]) >= 1
    assert second.json()["error"]["retryable"] is True


# -- provider layer: swapping the generator must not change the contract ----
def test_system_block_names_the_provider(client):
    """Two runs on different providers are two different systems, so the run has
    to record which one produced it, not just a model id."""
    data = client.get("/v1/meta", headers=AUTH).json()
    assert data["system"]["llm_provider"] in ("anthropic", "gemini", "groq", "offline")
    assert data["system"]["llm_model"]


@pytest.mark.parametrize(
    "error_text,expect_change,expect_body",
    [
        # Exactly what OpenAI returns for a reasoning model.
        ("Unsupported parameter: 'max_tokens' is not supported with this model. "
         "Use 'max_completion_tokens' instead.",
         "max_tokens -> max_completion_tokens",
         lambda b: "max_tokens" not in b and b["max_completion_tokens"] == 2000),
        ("Unsupported value: 'temperature' does not support 0 with this model.",
         "dropped temperature",
         lambda b: "temperature" not in b),
        ("Unsupported parameter: 'seed' is not supported with this model.",
         "dropped seed",
         lambda b: "seed" not in b),
        ("Invalid parameter: 'response_format' of type 'json_schema' is not supported.",
         "response_format json_schema -> json_object",
         lambda b: b["response_format"] == {"type": "json_object"}),
    ],
)
def test_unsupported_parameters_are_adapted_away(error_text, expect_change, expect_body):
    """Which optional parameters a chat-completions endpoint accepts varies by
    model, so the client gives up whatever the endpoint objects to instead of
    failing the question. Otherwise the project works with one person's key and
    not another's."""
    from backend import generate
    body = {
        "model": "m", "messages": [], "max_tokens": 2000, "temperature": 0, "seed": 42,
        "response_format": generate._json_schema_format(),
    }
    change = generate._adapt(body, error_text)
    assert change == expect_change
    assert expect_body(body), body


def test_unrecognised_400_is_not_adapted():
    """An error that is not about a parameter must surface, not trigger a
    pointless retry loop that hides the real problem."""
    from backend import generate
    body = {"model": "m", "messages": [], "max_tokens": 10}
    assert generate._adapt(body, "You exceeded your current quota") is None
    assert generate._adapt(body, "Invalid value for 'messages'") is None


def test_adaptation_is_idempotent():
    """A parameter already gone cannot be given up twice, or the loop would
    spin on the same error until it ran out of attempts."""
    from backend import generate
    body = {"model": "m", "messages": []}
    assert generate._adapt(body, "Unsupported parameter: 'seed'") is None


def test_every_known_provider_resolves_to_a_handled_kind():
    """Only three code paths exist -- the two native APIs and the OpenAI
    chat-completions protocol. A provider that resolved to anything else would
    reach the dispatcher's "unknown backend" branch only once someone configured
    it, which is the worst time to find out."""
    handled = {"anthropic", "gemini", "openai_compatible", "offline"}
    names = ["anthropic", "gemini", "custom", "offline", *config.OPENAI_COMPATIBLE]
    for name in names:
        original = config.LLM_BACKEND
        try:
            config.LLM_BACKEND = name
            provider = config.resolve()
            assert provider.kind in handled, f"{name} resolved to kind {provider.kind!r}"
            assert provider.name
        finally:
            config.LLM_BACKEND = original


def test_auto_detection_picks_the_provider_from_whichever_key_is_present(monkeypatch):
    """The point of auto: the tester drops any one key into .env and that
    provider is used, with no other edit. Nobody knows in advance which key they
    will have."""
    monkeypatch.setattr(config, "LLM_BACKEND", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "CUSTOM_BASE_URL", "")
    for var in ("ANTHROPIC_API_KEY", *(k for _, k, _ in config.OPENAI_COMPATIBLE.values())):
        monkeypatch.delenv(var, raising=False)

    # Nothing set: runnable rather than broken.
    assert config.resolve().kind == "offline"

    # A DeepSeek key alone is enough to select DeepSeek.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    detected = config.resolve()
    assert detected.name == "deepseek"
    assert detected.kind == "openai_compatible"
    assert detected.base_url.startswith("https://api.deepseek.com")
    assert detected.detected is True

    # A native-API key outranks it, because its key cannot be used over the
    # OpenAI protocol and mis-detecting it would fail every question.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert config.resolve().name == "anthropic"


def test_custom_provider_reaches_an_unlisted_endpoint(monkeypatch):
    """An OpenAI-compatible provider that is not in the table -- or did not exist
    when this was written -- must work from .env alone."""
    monkeypatch.setattr(config, "LLM_BACKEND", "custom")
    monkeypatch.setattr(config, "CUSTOM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(config, "CUSTOM_API_KEY", "whatever-key")
    monkeypatch.setattr(config, "CUSTOM_MODEL", "some-new-model")

    provider = config.resolve()
    assert provider.kind == "openai_compatible"
    assert provider.base_url == "https://llm.example.test/v1"
    assert provider.model == "some-new-model"


def test_local_server_needs_no_key(monkeypatch):
    """Ollama and LM Studio serve the OpenAI protocol on localhost without
    credentials; demanding a key would make a local model unusable."""
    monkeypatch.setattr(config, "LLM_BACKEND", "ollama")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    monkeypatch.setenv("RAG_OLLAMA_MODEL", "llama3.1")
    provider = config.resolve()
    assert provider.kind == "openai_compatible"
    assert provider.base_url.startswith("http://localhost")
    assert provider.api_key, "a local endpoint still needs a placeholder bearer value"


@pytest.mark.parametrize(
    "provider,key_var",
    [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY"),
     ("openai", "OPENAI_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY")],
)
def test_missing_provider_key_is_a_question_level_error(
    client, monkeypatch, provider, key_var
):
    """A provider named without its key must fail the question with a clear code
    naming the variable to set -- not crash the batch, and never fall back to a
    different provider, which would make the run's `system` block a lie."""
    from backend import generate, retrieval
    monkeypatch.setattr(config, "LLM_BACKEND", provider)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.delenv(key_var, raising=False)

    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError) as excinfo:
        generate.generate(UZ_LATN_Q, "uz-Latn", contexts, temperature=0, seed=1)
    assert excinfo.value.code == "LLM_ERROR"
    assert key_var in str(excinfo.value)


def test_provider_without_a_model_says_how_to_find_one(client, monkeypatch):
    """Model line-ups move, so several presets ship no default. The error has to
    point at the command that lists what the key can reach, not just fail."""
    from backend import generate, retrieval
    monkeypatch.setattr(config, "LLM_BACKEND", "together")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.delenv("RAG_TOGETHER_MODEL", raising=False)

    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError) as excinfo:
        generate.generate(UZ_LATN_Q, "uz-Latn", contexts, temperature=0, seed=3)
    assert "list_models.py together" in str(excinfo.value)
    assert "RAG_TOGETHER_MODEL" in str(excinfo.value)


def test_unknown_backend_is_rejected(client, monkeypatch):
    from backend import generate, retrieval
    monkeypatch.setattr(config, "LLM_BACKEND", "not-a-provider")
    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError) as excinfo:
        generate.generate(UZ_LATN_Q, "uz-Latn", contexts, temperature=0, seed=2)
    message = str(excinfo.value)
    assert "not-a-provider" in message
    # The error must tell the reader how to use a provider nobody anticipated.
    assert "RAG_CUSTOM_BASE_URL" in message
    assert "deepseek" in message


def test_gemini_schema_drops_additional_properties():
    """Gemini's responseSchema dialect rejects additionalProperties, so the
    shared contract is filtered rather than duplicated by hand."""
    from backend import generate

    def walk(node):
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(generate.GEMINI_SCHEMA)
    # The parts that matter for scoring must survive the filtering.
    assert generate.GEMINI_SCHEMA["required"] == ["no_answer", "answer", "used", "confidence"]
    assert generate.GEMINI_SCHEMA["properties"]["used"]["items"]["required"] == [
        "ref", "context", "quote"
    ]


# -- the article's place in the code must reach the model -------------------
def test_section_titles_were_recovered():
    """The corpus stores some sections as a bare marker ("X BOʻLIM.") because
    the upstream parser dropped the title line. Those titles are the only thing
    distinguishing 13 identically-titled "Soliq stavkalari" articles, so losing
    them cost a correct answer about the VAT rate."""
    from backend import retrieval
    titles = retrieval._section_titles()
    assert titles, "data/section_titles.json is missing or unreadable"
    assert titles["soliq_kodeksi"]["X BOʻLIM."] == "QOʻSHILGAN QIYMAT SOLIGʻI"
    # And the recovery must actually be applied to the loaded articles.
    assert retrieval._full_section("soliq_kodeksi", "X BOʻLIM.") == (
        "X BOʻLIM. QOʻSHILGAN QIYMAT SOLIGʻI"
    )
    # A section that already carries its title is left alone.
    assert retrieval._full_section("bojxona_kodeksi", "II boʻlim. Bojxona rejimlari") == (
        "II boʻlim. Bojxona rejimlari"
    )
    # An unknown code or empty section must not raise.
    assert retrieval._full_section("no_such_code", "I BOʻLIM.") == "I BOʻLIM."
    assert retrieval._full_section("soliq_kodeksi", "") == ""


@pytest.mark.parametrize("word,stem", [
    # The two forms that caused measured misses.
    ("toʻlovchilari", "toʻlovchi"),
    ("saylovda", "saylov"),
    ("yoshdan", "yosh"),
    ("kodeksining", "kodeksi"),
    # Already a stem, or too short to cut safely -- left alone.
    ("soliq", "soliq"),
    ("nikoh", "nikoh"),
    ("oʻgʻrilik", "oʻgʻrilik"),
])
def test_uzbek_stemmer_strips_only_inflection(word, stem):
    """Uzbek is agglutinative, so BM25 saw `toʻlovchilari` and `toʻlovchilar`
    as unrelated words. Only case and possessive endings are stripped; a stem
    is never cut below four characters."""
    from backend import retrieval
    assert retrieval.uz_stem(word) == stem


def test_stemming_recovers_an_article_it_used_to_miss(client):
    """Measured regression: the voting-age question returned no_answer because
    retrieval never surfaced the article that answers it. The question says
    "yoshdan", the article says "yoshga" -- no shared token before stemming."""
    from backend import retrieval
    contexts, _ = retrieval.retrieve(
        "Saylov kodeksiga koʻra saylovda qatnashish huquqi necha yoshdan beriladi?", top_k=5
    )
    found = [c for c in contexts
             if c.article_meta.code == "saylov_kodeksi" and c.article_meta.article_id == "4"]
    assert found, [c.article for c in contexts]
    assert "oʻn sakkiz" in found[0].text, "4-modda should state the age"


def test_stemming_does_not_regress_the_hand_checked_questions(client):
    """The five questions verified by reading the statutes must still come back
    at rank 1 -- a stemmer that over-cuts would quietly trade these away."""
    from backend import retrieval
    cases = [
        ("Yillik asosiy mehnat taʻtilining eng kam muddati qancha?", "mehnat_kodeksi", "217"),
        ("Jinoyat kodeksiga koʻra oʻgʻirlik uchun javobgarlik qanday?", "jinoyat_kodeksi", "169"),
        ("Жиноят кодексига кўра ўғрилик учун жавобгарлик қандай?", "jinoyat_kodeksi", "169"),
        ("Какой общий срок исковой давности установлен Гражданским кодексом?",
         "fuqarolik_kodeksi_1qism", "150"),
        ("Oʻzbekiston Respublikasi Konstitutsiyasiga koʻra davlat tili qaysi til?",
         "konstitutsiya", "4"),
    ]
    for question, code, article in cases:
        contexts, _ = retrieval.retrieve(question, top_k=5)
        top = contexts[0].article_meta
        assert (top.code, top.article_id) == (code, article), (
            f"{question[:40]!r} -> {[c.article for c in contexts]}"
        )


def test_vat_rate_article_is_retrieved_with_its_section(client):
    """The regression this guards: asked for the VAT rate, retrieval must put
    the VAT-section rate article first, and the context handed to the model must
    say which tax it belongs to -- the article text says only "soliq stavkasi 12
    foiz" and never names VAT."""
    from backend import generate, retrieval
    contexts, _ = retrieval.retrieve("Soliq kodeksiga koʻra QQS stavkasi necha foiz?", top_k=5)
    top = contexts[0]
    assert top.article_meta.article_id == "258", [c.article for c in contexts]
    assert "QOʻSHILGAN QIYMAT" in top.article_meta.section

    prompt = generate.build_prompt("QQS stavkasi?", "uz-Latn", contexts)
    assert "location:" in prompt
    assert "QOʻSHILGAN QIYMAT SOLIGʻI" in prompt, "the model cannot tell which tax this is"


def test_prompt_omits_location_when_unknown():
    """No heading is better than an empty one: a blank `location:` line would be
    noise in every context block for documents that have no sections."""
    from backend import generate, retrieval
    article = retrieval.Article(
        point_id="p", chunk_id="lex:1#art-1", doc_id="lex:1", code="c",
        article_id="1", article_display="1-modda", title="T", text="body",
        url="u", doc_title="D", doc_type="kodeks", edition_date="2026-07-01",
        doc_status="amalda", section="", chapter="",
    )
    context = retrieval.Context(
        rank=1, chunk_id=article.chunk_id, doc_id=article.doc_id,
        article=article.article_display, text=article.text,
        retrieval_score=0.5, rerank_score=0.5, article_meta=article,
    )
    assert "location:" not in generate.build_prompt("q", "uz-Latn", [context])


def test_retryable_generation_error_is_retried(client, monkeypatch):
    """A provider 429 is the provider's quota, not a property of the question.
    It must be retried rather than scored as a permanent failure."""
    from backend import generate, pipeline, retrieval
    monkeypatch.setattr(config, "GEN_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "GEN_RETRY_BUDGET_S", 10)

    calls = {"n": 0}
    good = generate.GenerationResult(
        answer="ok [1]", no_answer=False, confidence=0.9,
        citations=[], model="stub",
    )

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise generate.GenerationError("LLM_ERROR", "rate limited", True, retry_after=0.01)
        return good

    monkeypatch.setattr(generate, "generate", flaky)
    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    notes: list[str] = []
    result = pipeline._generate_with_retry(UZ_LATN_Q, "uz-Latn", contexts, 0, 42, notes)

    assert result is good
    assert calls["n"] == 3, "should have retried twice then succeeded"
    assert len(notes) == 2, notes


def test_non_retryable_generation_error_is_not_retried(client, monkeypatch):
    """A bad credential or an over-long context will fail identically every
    time; retrying only burns the budget."""
    from backend import generate, pipeline, retrieval
    monkeypatch.setattr(config, "GEN_MAX_RETRIES", 3)

    calls = {"n": 0}

    def always_fatal(*args, **kwargs):
        calls["n"] += 1
        raise generate.GenerationError("CONTEXT_TOO_LONG", "too long", False)

    monkeypatch.setattr(generate, "generate", always_fatal)
    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError):
        pipeline._generate_with_retry(UZ_LATN_Q, "uz-Latn", contexts, 0, 42, [])
    assert calls["n"] == 1, "a non-retryable error must not be retried"


def test_retry_gives_up_when_budget_exhausted(client, monkeypatch):
    """A provider asking for longer than the budget allows must not stall the
    batch: give up and report, letting the tester retry the batch instead."""
    from backend import generate, pipeline, retrieval
    monkeypatch.setattr(config, "GEN_MAX_RETRIES", 5)
    monkeypatch.setattr(config, "GEN_RETRY_BUDGET_S", 1)

    def rate_limited(*args, **kwargs):
        raise generate.GenerationError("LLM_ERROR", "rate limited", True, retry_after=600)

    monkeypatch.setattr(generate, "generate", rate_limited)
    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    notes: list[str] = []
    with pytest.raises(generate.GenerationError):
        pipeline._generate_with_retry(UZ_LATN_Q, "uz-Latn", contexts, 0, 42, notes)
    assert any("giving up" in n for n in notes), notes


@pytest.mark.parametrize("message", [
    # Groq uses both wordings. Matching only the first one sent the second to
    # the generic 400 branch, where "token" in the body mislabelled it
    # CONTEXT_TOO_LONG and non-retryable -- observed on a real batch.
    '{"error":{"message":"Failed to validate JSON. Please adjust your prompt."}}',
    '{"error":{"message":"Failed to generate JSON. Please adjust your prompt. '
    'max_tokens reached"}}',
])
def test_provider_json_failure_is_retryable_not_context_too_long(message):
    import httpx
    from backend import generate
    response = httpx.Response(400, text=message)
    error = generate._http_error("groq", httpx.HTTPStatusError(
        "400", request=httpx.Request("POST", "http://x"), response=response))
    assert error.code == "LLM_ERROR", error.code
    assert error.retryable is True


def test_salvageable_json_is_parsed_rather_than_retried():
    """When the provider hands back the output it rejected, our parser gets a
    look before a retry is spent -- its validator is stricter than json.loads."""
    import httpx
    from backend import generate
    good = json.dumps({"no_answer": False, "answer": "x [1]",
                       "used": [{"ref": 1, "context": 1, "quote": "q"}],
                       "confidence": 0.5})
    response = httpx.Response(400, json={"error": {
        "message": "Failed to validate JSON", "failed_generation": good}})
    with pytest.raises(generate._SalvageableJSON) as excinfo:
        generate._http_error("groq", httpx.HTTPStatusError(
            "400", request=httpx.Request("POST", "http://x"), response=response))
    assert excinfo.value.raw == good


def test_retry_after_header_is_honoured():
    """Groq and Gemini both send Retry-After on 429; obeying it beats guessing."""
    import httpx
    from backend import generate
    response = httpx.Response(429, headers={"retry-after": "7"}, text="slow down")
    error = generate._http_error("Groq", httpx.HTTPStatusError(
        "429", request=httpx.Request("POST", "http://x"), response=response))
    assert error.retryable is True
    assert error.retry_after == 7.0

    # An HTTP-date Retry-After is valid but not a number; fall back to backoff.
    dated = httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"})
    error = generate._http_error("Groq", httpx.HTTPStatusError(
        "429", request=httpx.Request("POST", "http://x"), response=dated))
    assert error.retry_after is None


def test_fenced_json_is_parsed(client):
    """Models wrap JSON in ```json fences even when told not to; that is a
    formatting quirk, not a failed question."""
    from backend import generate
    result = generate._parse_answer_json(
        '```json\n{"no_answer": false, "answer": "x [1]", '
        '"used": [{"ref": 1, "context": 1, "quote": "q"}], "confidence": 0.5}\n```',
        "test-model", 10, 20,
    )
    assert result.answer == "x [1]"
    assert result.citations[0].quote == "q"
    assert result.input_tokens == 10 and result.output_tokens == 20


# -- POST /v1/rag/query (optional debug endpoint) --------------------------
def test_query_endpoint_answers_one_question(client):
    r = client.post("/v1/rag/query", json={"question": UZ_LATN_Q}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["system"]["index_version"]
    assert body["result"]["status"] in ("ok", "no_answer")


def test_query_endpoint_rejects_empty_question(client):
    r = client.post("/v1/rag/query", json={"question": ""}, headers=AUTH)
    assert r.status_code == 400
