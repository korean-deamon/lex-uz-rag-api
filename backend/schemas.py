"""
Request/response models for /v1/rag/*.

Validation lives here so the spec's 400 cases -- more than 10 questions, a
duplicate id, an empty question -- are rejected before any embedding or model
call happens, and are rejected for the stated reason rather than a generic
"invalid body".
"""
from datetime import date

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from backend import config, lang


class Utf8JSONResponse(JSONResponse):
    """Spec section 2 names the response header exactly:
    `Content-Type: application/json; charset=utf-8`. JSON is UTF-8 by definition
    so Starlette omits the charset, but the tester checks the header, and
    spelling it out also stops any intermediary from guessing a legacy encoding
    for oʻ / gʻ. Starlette serialises with ensure_ascii=False, so those
    characters go out as real UTF-8 rather than \\u escapes."""

    media_type = "application/json; charset=utf-8"


class Options(BaseModel):
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=50)
    return_contexts: bool = True
    return_debug: bool = False
    # Accepted and echoed; see the note in generate.py on why the API cannot
    # enforce them any more.
    temperature: float | None = Field(default=0, ge=0, le=1)
    seed: int | None = None
    as_of_date: date | None = None
    answer_language: str = "auto"

    @field_validator("answer_language")
    @classmethod
    def _check_language(cls, v: str) -> str:
        if not lang.is_supported(v):
            raise ValueError(
                f"answer_language must be 'auto' or one of {list(lang.SUPPORTED)}, got {v!r}"
            )
        return v


class QuestionItem(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    language: str | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace only")
        return v

    # items[].language is deliberately NOT validated here. The spec lists
    # INVALID_LANGUAGE as a question-level error code, and its 400 list covers
    # only "more than 10 questions, duplicate id, empty question" -- so a bad
    # language tag on one item must fail that item, not the whole batch.


class BatchRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=128)
    batch_index: int = Field(ge=1)
    total_batches: int = Field(ge=1)
    options: Options = Field(default_factory=Options)
    items: list[QuestionItem] = Field(min_length=1, max_length=config.MAX_BATCH_ITEMS)

    @model_validator(mode="after")
    def _unique_ids(self) -> "BatchRequest":
        ids = [i.id for i in self.items]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"items[].id must be unique within a batch; duplicated: {duplicates}")
        if self.batch_index > self.total_batches:
            raise ValueError(
                f"batch_index ({self.batch_index}) cannot exceed total_batches ({self.total_batches})"
            )
        return self


class QueryRequest(BaseModel):
    """POST /v1/rag/query -- the optional single-question debug endpoint."""
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS)
    language: str | None = None
    options: Options = Field(default_factory=Options)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace only")
        return v


def error_body(code: str, message: str, retryable: bool = False) -> dict:
    """Request-level error envelope, matching the shape of the per-question one."""
    return {"error": {"code": code, "message": message, "retryable": retryable}}
