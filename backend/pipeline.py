"""
Per-question orchestration: retrieve -> generate -> verify -> spec-shaped result.

The verify step is the part that matters for the evaluation. The model is asked
for verbatim quotes, but "asked for" is not "guaranteed", so every citation is
re-checked here against the context it claims to come from:

  * the context number must exist in what retrieval actually returned,
  * the quote must be an exact substring of that context's text,
  * the chunk_id is taken from the context, never from the model,

which together make the spec's two structural rules hold by construction:
"every citations[].chunk_id is present in contexts" and "the system does not
cite a source it never saw". A citation failing any check is dropped; if none
survive, the answer is downgraded to no_answer rather than published without
grounding -- an ungrounded answer is exactly what the fabricated-source metric
is there to catch.

Bracketed markers in the answer are renumbered to match the surviving
citations, so `[n]` in `answer` always resolves to a real `citations[].ref`.
"""
import re
import time
from datetime import datetime, timezone

from backend import config, corpus_meta, generate, lang, retrieval

STATUS_OK = "ok"
STATUS_NO_ANSWER = "no_answer"
STATUS_ERROR = "error"
STATUS_TIMEOUT = "timeout"

_MARKER = re.compile(r"\[(\d+)\]")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _normalise_for_match(text: str) -> str:
    """Fold the characters that differ only by typography, so a quote that is
    materially verbatim is not rejected over a space or an apostrophe variant.
    Nothing is removed, so a paraphrase still fails the check."""
    return re.sub(r"\s+", " ", lang.normalise(text)).strip()


def compute_part(body: str, quote: str) -> str:
    """Which numbered part (qism) of the article the quote sits in.

    Uzbek statute text separates parts by line break, so the paragraph index of
    the quote is the part number. Articles with a single paragraph are 1-qism.
    """
    paragraphs = [p for p in body.split("\n") if p.strip()]
    needle = _normalise_for_match(quote)
    for i, paragraph in enumerate(paragraphs, start=1):
        if needle and needle in _normalise_for_match(paragraph):
            return f"{i}-qism"
    return "1-qism"


def verify_citations(
    raw: list[generate.RawCitation], contexts: list[retrieval.Context]
) -> tuple[list[dict], dict[int, int], list[str]]:
    """-> (citations, ref_remap old->new, rejection notes)."""
    by_rank = {c.rank: c for c in contexts}
    citations: list[dict] = []
    remap: dict[int, int] = {}
    notes: list[str] = []
    seen: set[tuple[str, str]] = set()

    for item in raw:
        ctx = by_rank.get(item.context_rank)
        if ctx is None:
            notes.append(f"dropped ref {item.ref}: context {item.context_rank} was not retrieved")
            continue

        art = ctx.article_meta
        body = generate._article_body(art)
        quote = item.quote.strip()
        if not quote or _normalise_for_match(quote) not in _normalise_for_match(body):
            notes.append(f"dropped ref {item.ref}: quote is not verbatim in {ctx.chunk_id}")
            continue

        # The same article quoted twice collapses into one citation.
        dedupe_key = (ctx.chunk_id, _normalise_for_match(quote))
        if dedupe_key in seen:
            remap[item.ref] = next(
                c["ref"] for c in citations if c["chunk_id"] == ctx.chunk_id
            )
            continue
        seen.add(dedupe_key)

        new_ref = len(citations) + 1
        remap[item.ref] = new_ref
        ctx.used_in_answer = True
        citations.append(
            {
                "ref": new_ref,
                "doc_id": art.doc_id,
                "doc_title": art.doc_title,
                "doc_type": art.doc_type,
                "url": art.url,
                "article": art.article_display,
                "part": compute_part(body, quote),
                "edition_date": art.edition_date,
                "doc_status": art.doc_status,
                "chunk_id": ctx.chunk_id,
                "quote": quote,
            }
        )
    return citations, remap, notes


def renumber_markers(answer: str, remap: dict[int, int]) -> str:
    """Rewrite [n] markers onto the surviving citation refs; drop the rest."""
    def sub(match: re.Match) -> str:
        old = int(match.group(1))
        new = remap.get(old)
        return f"[{new}]" if new else ""
    return re.sub(r"\s+([.,;:])", r"\1", _MARKER.sub(sub, answer)).strip()


def _is_index_unavailable(error: Exception) -> bool:
    """Tell "the vector store is unreachable" from "this search failed".

    qdrant-client wraps a refused connection or a timeout in
    ResponseHandlingException; matching on the class by name avoids importing a
    private module path, and the message check catches the same condition
    arriving through httpx directly.
    """
    name = type(error).__name__
    if name in ("ResponseHandlingException", "ConnectError", "ConnectTimeout",
                "ReadTimeout", "ConnectionError"):
        return True
    text = str(error).lower()
    return any(
        signal in text
        for signal in ("connection refused", "connection error", "timed out",
                       "name or service not known", "failed to connect")
    )


def _generate_with_retry(
    question: str,
    answer_language: str,
    contexts: list[retrieval.Context],
    temperature: float | None,
    seed: int | None,
    notes: list[str],
) -> generate.GenerationResult:
    """Generate, retrying a retryable failure within a bounded time budget.

    `notes` is appended to in place, so the attempts are visible in `debug`
    whether the question eventually succeeded or not.

    A provider's 429 is not a property of the question, so letting it become a
    permanent `LLM_ERROR` result would charge the pipeline for the provider's
    quota: a free-tier key would score far worse than the same system on a paid
    one. The budget stops a rate-limited provider from stalling the batch -- past
    the sync deadline the 202 async path takes over.

    A non-retryable failure (bad credential, context too long, a refusal) is
    raised immediately. Retrying those would only waste the budget.
    """
    deadline = time.monotonic() + config.GEN_RETRY_BUDGET_S
    last: generate.GenerationError | None = None

    for attempt in range(config.GEN_MAX_RETRIES + 1):
        try:
            return generate.generate(
                question, answer_language, contexts,
                temperature=temperature, seed=seed,
            )
        except generate.GenerationError as e:
            last = e
            if not e.retryable or attempt == config.GEN_MAX_RETRIES:
                raise
            # Exponential backoff unless the provider named a delay.
            wait = e.retry_after if e.retry_after is not None else 2 ** attempt
            remaining = deadline - time.monotonic()
            if remaining <= 0 or wait > remaining:
                notes.append(
                    f"attempt {attempt + 1} failed ({e.code}); "
                    f"needed {wait:.1f}s but only {max(0.0, remaining):.1f}s of "
                    f"retry budget left, giving up"
                )
                raise
            notes.append(f"attempt {attempt + 1} failed ({e.code}); retrying in {wait:.1f}s")
            time.sleep(wait)

    raise last  # unreachable; present so the contract is explicit


def answer_question(item: dict, options: dict) -> dict:
    """One `results[]` entry. Never raises: a failure becomes status=error."""
    started = time.perf_counter()
    question_id = item["id"]
    question = item["question"]
    top_k = int(options.get("top_k") or config.DEFAULT_TOP_K)
    return_contexts = options.get("return_contexts", True)
    return_debug = bool(options.get("return_debug"))

    base = {
        "id": question_id,
        "status": STATUS_ERROR,
        "answer": None,
        "answer_language": None,
        "no_answer": False,
        "confidence": None,
        "citations": [],
        "contexts": [],
        "debug": None,
        "timing": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "error": None,
    }

    requested_language = item.get("language")
    if requested_language is not None and not lang.is_supported(requested_language):
        base["error"] = {
            "code": "INVALID_LANGUAGE",
            "message": f"Unsupported language {requested_language!r}; expected one of {lang.SUPPORTED}",
            "retryable": False,
        }
        base["timing"] = {"total_ms": int((time.perf_counter() - started) * 1000)}
        return base

    answer_language = lang.resolve_answer_language(
        options.get("answer_language"), question
    )
    if options.get("answer_language") in (None, "auto") and requested_language in lang.SUPPORTED:
        # An explicit per-item language overrides sniffing for that item.
        answer_language = requested_language
    base["answer_language"] = answer_language

    # --- retrieval ---
    t_retrieval = time.perf_counter()
    try:
        contexts, debug = retrieval.retrieve(question, top_k=top_k)
    except Exception as e:
        # The spec lists INDEX_UNAVAILABLE and RETRIEVAL_FAILED as separate
        # codes, and the difference is the whole point of having both: one says
        # the vector store cannot be reached at all, the other that the search
        # failed for this question. Reporting an unreachable Qdrant as
        # RETRIEVAL_FAILED would send the tester hunting through the question
        # when the service is simply down.
        base["error"] = {
            "code": "INDEX_UNAVAILABLE" if _is_index_unavailable(e) else "RETRIEVAL_FAILED",
            "message": f"{type(e).__name__}: {e}",
            "retryable": True,
        }
        base["timing"] = {"total_ms": int((time.perf_counter() - started) * 1000)}
        return base
    retrieval_ms = int((time.perf_counter() - t_retrieval) * 1000)

    as_of = options.get("as_of_date")
    if as_of and str(as_of) < config.LEX_SNAPSHOT_DATE:
        debug["as_of_date_warning"] = (
            f"requested as_of_date {as_of} precedes the indexed lex.uz snapshot "
            f"{config.LEX_SNAPSHOT_DATE}; the index holds only that snapshot, so the "
            f"answer reflects it, not the law as of {as_of}"
        )

    # --- generation ---
    t_generation = time.perf_counter()
    retry_notes: list[str] = []
    try:
        result = _generate_with_retry(
            question,
            answer_language,
            contexts,
            temperature=options.get("temperature", 0),
            seed=options.get("seed"),
            notes=retry_notes,
        )
    except generate.GenerationError as e:
        if retry_notes:
            debug["generation_retries"] = retry_notes
        base["status"] = STATUS_TIMEOUT if e.code == "LLM_TIMEOUT" else STATUS_ERROR
        base["error"] = {"code": e.code, "message": e.message, "retryable": e.retryable}
        base["timing"] = {
            "retrieval_ms": retrieval_ms,
            "generation_ms": int((time.perf_counter() - t_generation) * 1000),
            "total_ms": int((time.perf_counter() - started) * 1000),
        }
        if return_contexts:
            base["contexts"] = [c.to_spec_dict() for c in contexts]
        if return_debug:
            base["debug"] = debug
        return base
    generation_ms = int((time.perf_counter() - t_generation) * 1000)

    # --- verify ---
    citations, remap, notes = verify_citations(result.citations, contexts)
    answer = renumber_markers(result.answer, remap)
    no_answer = result.no_answer

    if not no_answer and not citations:
        # Nothing survived verification: there is no grounded answer to publish.
        no_answer = True
        notes.append("downgraded to no_answer: no citation survived quote verification")
        answer = ""

    base.update(
        {
            "status": STATUS_NO_ANSWER if no_answer else STATUS_OK,
            "answer": answer if not no_answer else None,
            "no_answer": no_answer,
            "confidence": result.confidence,
            "citations": citations,
            "contexts": [c.to_spec_dict() for c in contexts] if return_contexts else [],
            "usage": {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            "timing": {
                "retrieval_ms": retrieval_ms,
                "rerank_ms": 0,  # fused inside retrieval; reported for shape parity
                "generation_ms": generation_ms,
                "total_ms": int((time.perf_counter() - started) * 1000),
            },
            "error": None,
        }
    )
    if return_debug:
        debug["generator_model"] = result.model
        debug["verification_notes"] = notes + result.notes
        if retry_notes:
            debug["generation_retries"] = retry_notes
        base["debug"] = debug
    return base


def system_block() -> dict:
    """The `system` object: what a run's results can be attributed to."""
    return {
        "pipeline_version": config.PIPELINE_VERSION,
        # Provider as well as model id: the same run scored on two providers is
        # two different systems, and "llm_model" alone would not say which.
        "llm_provider": config.llm_provider(),
        "llm_model": config.llm_model(),
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker": config.RERANKER_NAME,
        "index_version": config.INDEX_VERSION,
        "lex_snapshot_date": config.LEX_SNAPSHOT_DATE,
    }


def summarise(results: list[dict]) -> dict:
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == STATUS_OK),
        "no_answer": sum(1 for r in results if r["status"] == STATUS_NO_ANSWER),
        "error": sum(1 for r in results if r["status"] in (STATUS_ERROR, STATUS_TIMEOUT)),
    }


def run_batch(items: list[dict], options: dict) -> tuple[list[dict], dict]:
    """Answer every item. One question's failure cannot affect the others."""
    received_at = _now_iso()
    started = time.perf_counter()
    results = [answer_question(item, options) for item in items]
    timing = {
        "received_at": received_at,
        "completed_at": _now_iso(),
        "total_ms": int((time.perf_counter() - started) * 1000),
    }
    return results, timing


def meta_block() -> dict:
    """GET /v1/meta -- including the honest limits of the current index."""
    docs = corpus_meta.doc_meta_by_code()
    return {
        "system": system_block(),
        "index": {
            "collection": config.COLLECTION,
            "articles_indexed": retrieval.index_size(),
            "documents_indexed": len(docs),
            "chunking": "one chunk per article (modda); no fixed-length splitting",
            "lex_snapshot_date": config.LEX_SNAPSHOT_DATE,
            **retrieval.excluded_point_counts(),
        },
        "documents": sorted(
            (
                {
                    "code": code,
                    "doc_id": m["doc_id"],
                    "doc_title": m["doc_title"],
                    "doc_type": m["doc_type"],
                    "url": m["url"],
                    "edition_date": m["edition_date"],
                    "adopted_date": m["adopted_date"],
                    "doc_status": m["doc_status"],
                }
                for code, m in docs.items()
            ),
            key=lambda d: d["code"],
        ),
        "limits": {
            "max_batch_items": config.MAX_BATCH_ITEMS,
            "max_question_chars": config.MAX_QUESTION_CHARS,
            "default_top_k": config.DEFAULT_TOP_K,
            "sync_deadline_s": config.SYNC_DEADLINE_S,
            "min_batch_interval_s": config.RATE_LIMIT_MIN_INTERVAL_S,
            "result_retention_days": config.RESULT_RETENTION_DAYS,
        },
        "languages": list(lang.SUPPORTED),
        # Stated plainly so the tester does not read a constant as a real check.
        "caveats": [
            "edition_date is the corpus snapshot month for the whole document, not a "
            "per-article amendment date: the corpus carries consolidated in-force text "
            "with no revision history.",
            "doc_status is 'amalda' for every indexed document because all 25 are in "
            "force. No repealed document is indexed, so the 'eskirgan norma' metric "
            "cannot currently fail and is not evidence that the check works.",
            "The collection also holds chunks from the older raw-scrape pipeline "
            "(whole lex.uz documents split by character count). They carry no article "
            "structure, so a citation from one could not fill the mandatory `article` "
            "and `part` fields; /v1 serves the article-level corpus only and reports "
            "the excluded count under `index` above.",
            "as_of_date is recorded and echoed, but only the one snapshot above is "
            "indexed; an earlier as_of_date adds a warning to debug instead of serving "
            "a historical edition.",
            _sampling_caveat(),
        ],
    }


def _sampling_caveat() -> str:
    """How far temperature/seed are actually honoured depends on the provider, so
    say which case this run is in rather than making one claim for all of them."""
    shared = (
        " They are always echoed back and always form part of the answer-cache key, "
        "so an identical request returns an identical answer; retrieval is "
        "deterministic by construction."
    )
    provider = config.resolve()
    if provider.kind == "openai_compatible":
        return (
            f"temperature and seed are both sent to {provider.name}, which accepts them "
            "on most models; where a model rejects one it is dropped and the drop is "
            "recorded in debug." + shared
        )
    if provider.kind == "gemini":
        return (
            "temperature is passed to Gemini, which accepts it. Gemini has no seed "
            "parameter, so seed is recorded but not sent." + shared
        )
    if provider.kind == "offline":
        return (
            "temperature and seed are recorded but meaningless on the offline backend, "
            "which does no sampling at all." + shared
        )
    return (
        "temperature and seed are recorded but not sent: the Anthropic API no longer "
        "accepts sampling parameters (temperature is not a parameter of messages.create "
        "in SDK 2.x) and has never had a seed." + shared
    )
