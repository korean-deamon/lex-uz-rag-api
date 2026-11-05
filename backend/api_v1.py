"""
The /v1 RAG API required by the test specification.

  POST /v1/rag/batch             1-10 questions, answers + sources + contexts
  GET  /v1/rag/batch/{batch_id}  re-read a stored result, or poll an async batch
  POST /v1/rag/query             one question (debug)
  GET  /v1/meta                  model, embedding, index version, snapshot date
  GET  /v1/health                liveness

Sync is the primary mode. A batch that is still running when the sync deadline
(120s) passes is not abandoned -- it keeps going on its worker thread, the
endpoint answers 202 with status "processing", and GET serves the result once
the worker stores it. That way a slow batch costs the tester one poll rather
than a lost run.
"""
import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from backend import config, pipeline, retrieval, schemas, store

log = logging.getLogger("rag.v1")

router = APIRouter(prefix="/v1", default_response_class=schemas.Utf8JSONResponse)

# One worker per in-flight batch that outlived the sync wait. Batches are sent
# sequentially by the tester, so this stays small; it exists so an async batch
# can finish writing its result after the HTTP response has gone out.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-batch")
_inflight: dict[str, Future] = {}


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def require_api_key(authorization: str | None = Header(default=None)) -> str:
    """Authorization: Bearer <API_KEY>.

    A missing or malformed header is 401 (the caller did not authenticate); a
    well-formed but unrecognised key is 403 (authenticated shape, no access).
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail=schemas.error_body(
                "UNAUTHENTICATED", "Missing or malformed Authorization: Bearer <API_KEY> header"
            )["error"],
        )
    key = authorization.split(" ", 1)[1].strip()
    if key not in config.API_KEYS:
        raise HTTPException(
            status_code=403,
            detail=schemas.error_body("FORBIDDEN", "Unrecognised API key")["error"],
        )
    return key


def _request_hash(req: schemas.BatchRequest) -> str:
    """Identity of a batch's *content*, so a retry is told apart from a reuse of
    the same batch_id for different questions (409). Options are included: the
    same questions asked with a different top_k is a different run."""
    payload = json.dumps(
        {
            "run_id": req.run_id,
            "items": [[i.id, i.question, i.language] for i in req.items],
            "options": req.options.model_dump(mode="json"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_batch(req: schemas.BatchRequest) -> dict:
    options = req.options.model_dump(mode="json")
    items = [i.model_dump() for i in req.items]
    results, timing = pipeline.run_batch(items, options)
    return {
        "run_id": req.run_id,
        "batch_id": req.batch_id,
        "batch_index": req.batch_index,
        "total_batches": req.total_batches,
        "status": store.STATUS_COMPLETED,
        "system": pipeline.system_block(),
        "timing": timing,
        "summary": pipeline.summarise(results),
        "options_echo": options,
        "results": results,
    }


def _failed_response(stored: dict) -> schemas.Utf8JSONResponse:
    """A whole-batch failure, answered as 503.

    The spec's status table maps a service error to 500/503/504, and that code is
    what tells the tester to retry (5s, 15s, 45s). Returning the stored failure
    body with HTTP 200 instead made it look like a result: run_eval.py would save
    a batch of zero answers and move on, so a transient fault became a hole in
    the run. 503 because the condition is "try again", not "your request was
    wrong"; the body still carries batch_id and the error, and GET keeps serving
    the stored `failed` record, which section 5 explicitly provides for.
    """
    return schemas.Utf8JSONResponse(status_code=503, content=stored)


def _run_and_store(req: schemas.BatchRequest) -> dict:
    """Compute a batch and persist it. Runs on a worker thread."""
    try:
        response = _compute_batch(req)
        store.save(req.batch_id, store.STATUS_COMPLETED, response)
        return response
    except Exception as e:  # a whole-batch failure is still a stored outcome
        log.exception("batch %s failed", req.batch_id)
        failed = {
            "run_id": req.run_id,
            "batch_id": req.batch_id,
            "status": store.STATUS_FAILED,
            "system": pipeline.system_block(),
            "error": {"code": "INTERNAL_ERROR", "message": f"{type(e).__name__}: {e}", "retryable": True},
            "results": [],
        }
        store.save(req.batch_id, store.STATUS_FAILED, failed)
        return failed
    finally:
        _inflight.pop(req.batch_id, None)


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@router.post("/rag/batch")
def post_batch(req: schemas.BatchRequest, response: Response, api_key: str = Depends(require_api_key)):
    wait = store.rate_limiter.check(api_key)
    if wait > 0:
        retry_after = max(1, int(wait + 0.999))
        return schemas.Utf8JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content=schemas.error_body(
                "RATE_LIMITED",
                f"At most one batch per {config.RATE_LIMIT_MIN_INTERVAL_S:g}s; retry in {wait:.2f}s",
                retryable=True,
            ),
        )

    request_hash = _request_hash(req)
    outcome, row = store.reserve(req.batch_id, req.run_id, request_hash)

    if outcome == store.RESERVE_CONFLICT:
        return schemas.Utf8JSONResponse(
            status_code=409,
            content=schemas.error_body(
                "BATCH_ID_CONFLICT",
                f"batch_id {req.batch_id!r} was already used for a different set of "
                f"questions or options. Use a new batch_id.",
            ),
        )

    if outcome == store.RESERVE_EXISTING:
        # Idempotent replay: the stored result, not a recomputation.
        if row and row["response"] is not None:
            if row["status"] == store.STATUS_FAILED:
                return _failed_response(row["response"])
            response.headers["X-Idempotent-Replay"] = "true"
            return row["response"]
        # Reserved but still running (a retry arrived mid-flight).
        return schemas.Utf8JSONResponse(
            status_code=202,
            content={
                "run_id": req.run_id,
                "batch_id": req.batch_id,
                "status": store.STATUS_PROCESSING,
            },
        )

    future = _executor.submit(_run_and_store, req)
    _inflight[req.batch_id] = future
    try:
        result = future.result(timeout=config.SYNC_DEADLINE_S)
        if result.get("status") == store.STATUS_FAILED:
            return _failed_response(result)
        return result
    except FutureTimeout:
        # Spec: past the sync deadline, hand back 202 and let the tester poll.
        # The worker keeps running and will store the result.
        return schemas.Utf8JSONResponse(
            status_code=202,
            content={
                "run_id": req.run_id,
                "batch_id": req.batch_id,
                "status": store.STATUS_PROCESSING,
                "poll": f"/v1/rag/batch/{req.batch_id}",
            },
        )


@router.get("/rag/batch/{batch_id}")
def get_batch(batch_id: str, api_key: str = Depends(require_api_key)):
    row = store.get(batch_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=schemas.error_body(
                "BATCH_NOT_FOUND",
                f"No batch {batch_id!r}. Results are kept for "
                f"{config.RESULT_RETENTION_DAYS} days.",
            )["error"],
        )
    if row["response"] is not None:
        return row["response"]
    return schemas.Utf8JSONResponse(
        status_code=202,
        content={"run_id": row["run_id"], "batch_id": batch_id, "status": row["status"]},
    )


@router.post("/rag/query")
def post_query(req: schemas.QueryRequest, api_key: str = Depends(require_api_key)):
    """Single question, for debugging. Not part of the scored run."""
    options = req.options.model_dump(mode="json")
    result = pipeline.answer_question(
        {"id": "q-debug", "question": req.question, "language": req.language}, options
    )
    return {"system": pipeline.system_block(), "result": result}


@router.get("/meta")
def get_meta(api_key: str = Depends(require_api_key)):
    return pipeline.meta_block()


@router.get("/health")
def get_health():
    """Liveness. Unauthenticated on purpose: a monitor should not need the test key.

    Reports degraded rather than ok when the index cannot be reached, so a
    green health check means the service can actually answer.
    """
    checks: dict[str, object] = {}
    healthy = True
    try:
        collections = [c.name for c in retrieval.qdrant().get_collections().collections]
        indexed = config.COLLECTION in collections
        checks["qdrant"] = "ok" if indexed else f"collection {config.COLLECTION!r} missing"
        healthy = healthy and indexed
    except Exception as e:
        checks["qdrant"] = f"unreachable: {type(e).__name__}"
        healthy = False

    try:
        checks["articles_indexed"] = retrieval.index_size()
        healthy = healthy and retrieval.index_size() > 0
    except Exception as e:
        checks["articles_indexed"] = f"error: {type(e).__name__}"
        healthy = False

    provider = config.resolve()
    checks["llm_provider"] = provider.name + (" (auto-detected)" if provider.detected else "")
    checks["llm_model"] = provider.model or "(not set)"
    body = {"status": "ok" if healthy else "degraded", "checks": checks,
            "system": pipeline.system_block()}
    return schemas.Utf8JSONResponse(status_code=200 if healthy else 503, content=body)
