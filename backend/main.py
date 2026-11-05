"""
FastAPI app: the /v1 RAG API from the test specification.

Run with: uvicorn backend.main:app --port 8000

There used to be a second endpoint here, POST /api/chat, behind a small browser
chat UI, running an earlier single-question pipeline of its own. Both are gone.
It answered the same questions through different retrieval code, so the same
question could come back two different ways depending on which door it entered
by -- confusing to demonstrate and misleading to evaluate. The specification's
work is /v1, and /v1 is now the only way in.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import api_v1, config, retrieval, schemas, store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rag")

ROOT = Path(__file__).resolve().parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    purged = store.purge_expired()
    if purged:
        log.info("purged %d expired batch results", purged)
    # Build the lexical index up front: the spec's latency targets are per
    # question, and a cold first request would otherwise pay the ~4s load.
    try:
        log.info("warming retrieval index...")
        log.info("retrieval index ready: %d articles", retrieval.index_size())
        # The embedding model loads lazily on first use, which otherwise lands
        # entirely on the first question of the first batch (~8s measured).
        retrieval.embed_query("warmup")
        log.info("embedding model ready")
    except Exception as e:
        log.error("retrieval index unavailable at startup: %s: %s", type(e).__name__, e)
    yield


app = FastAPI(
    title="lex.uz RAG API",
    version=config.PIPELINE_VERSION,
    lifespan=lifespan,
    default_response_class=schemas.Utf8JSONResponse,
    description=(
        "Batch RAG API over the Uzbek legal corpus (Constitution + 20 codes + 4 laws, "
        "article-level chunks sourced from lex.uz)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_v1.router)


# --------------------------------------------------------------------------
# Error envelopes
# --------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
def _validation_error(request: Request, exc: RequestValidationError) -> schemas.Utf8JSONResponse:
    """Spec section 5 lists validation failures -- more than 10 questions, a
    duplicate id, an empty question -- as 400. FastAPI's default is 422, so the
    mapping happens here, keeping the reason in the message."""
    problems = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        problems.append(f"{location or 'body'}: {err.get('msg')}")
    return schemas.Utf8JSONResponse(
        status_code=400,
        content=schemas.error_body("VALIDATION_FAILED", "; ".join(problems)),
    )


@app.exception_handler(StarletteHTTPException)
def _http_error(request: Request, exc: StarletteHTTPException) -> schemas.Utf8JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        body = {"error": detail}
    else:
        body = schemas.error_body("HTTP_ERROR", str(detail))
    return schemas.Utf8JSONResponse(status_code=exc.status_code, content=body, headers=exc.headers)


@app.get("/")
def index():
    """A pointer, not a UI. The batch API returns JSON; there is no page to
    render it, so say where the endpoints are instead of 404-ing."""
    return {
        "service": "lex.uz RAG API",
        "version": config.PIPELINE_VERSION,
        "endpoints": {
            "batch": "POST /v1/rag/batch",
            "batch_result": "GET /v1/rag/batch/{batch_id}",
            "single_question": "POST /v1/rag/query",
            "meta": "GET /v1/meta",
            "health": "GET /v1/health",
            "openapi": "GET /docs",
        },
        "auth": "Authorization: Bearer <API_KEY>",
        "docs": "see README.md and TESTING.md",
    }
