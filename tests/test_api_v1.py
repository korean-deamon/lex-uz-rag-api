"""
The specification's section 7 checklist, as executable tests.

Each test names the checklist line it covers, so "is the API ready for the
test?" is answered by running this file rather than by reading the code.
"""
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


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_missing_provider_key_is_a_question_level_error(client, monkeypatch, provider):
    """A provider selected without its key must fail the question with a clear
    code -- not crash the batch, and not fall back to a different provider
    silently, which would make the run's `system` block a lie."""
    from backend import generate, retrieval
    monkeypatch.setattr(config, "LLM_BACKEND", provider)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)
    monkeypatch.setattr(config, "GROQ_API_KEY", None)

    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError) as excinfo:
        generate.generate(UZ_LATN_Q, "uz-Latn", contexts, temperature=0, seed=1)
    assert excinfo.value.code == "LLM_ERROR"
    assert provider in str(excinfo.value).lower()


def test_unknown_backend_is_rejected(client, monkeypatch):
    from backend import generate, retrieval
    monkeypatch.setattr(config, "LLM_BACKEND", "not-a-provider")
    contexts, _ = retrieval.retrieve(UZ_LATN_Q, top_k=3)
    with pytest.raises(generate.GenerationError) as excinfo:
        generate.generate(UZ_LATN_Q, "uz-Latn", contexts, temperature=0, seed=2)
    assert "anthropic, gemini, groq or offline" in str(excinfo.value)


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
