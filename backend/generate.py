"""
Answer generation, grounded strictly in the retrieved contexts.

The model is asked for structured JSON rather than prose, because the spec needs
machine-checkable parts of the answer that free text cannot carry reliably:
which context each claim came from, and the verbatim quote that supports it. The
quote is what makes `faithfulness` and `citation accuracy` measurable at all --
pipeline.py re-checks every quote against the context it claims to come from and
drops any citation that is not verbatim, so a fabricated reference cannot reach
the response.

Four backends, selected by RAG_LLM_BACKEND. The prompt, the schema and all of
the verification are shared; only the HTTP call differs, so a provider swap
cannot quietly change what is being measured:

  anthropic  Claude, official SDK, structured outputs via output_config.
  gemini     Google Gemini, REST, structured outputs via responseSchema.
  groq       Groq, OpenAI-compatible REST, JSON mode + schema in the prompt.
  offline    Deterministic extractive answerer. No credential, no network. It
             quotes the top article instead of writing prose, so it is honest
             about what it is: retrieval output. It cannot judge no_answer and
             cannot honour answer_language.

On `temperature` and `seed`: the spec asks for both, and whether they can be
honoured depends on the provider.
  * Anthropic no longer accepts sampling parameters at all -- `temperature` is
    not a parameter of messages.create in SDK 2.x, and `seed` never existed.
  * Gemini accepts `temperature`; it has no seed.
  * Groq accepts both `temperature` and `seed`.
They are therefore passed through wherever the provider takes them, always
echoed back, and always part of the answer-cache key -- so a repeated question
returns a byte-identical answer on every backend. Retrieval, the part the spec
actually tests for reproducibility ("bir xil savol bir xil manbalarni
qaytaradi"), is deterministic by construction. GET /v1/meta states which of
these applies to the run being scored.
"""
import hashlib
import json
import re
from dataclasses import dataclass, field

import anthropic
import httpx

from backend import config, lang
from backend.retrieval import Context

# Keyed by (question, answer_language, chunk_ids, backend, model, temperature,
# seed) so an identical request is answered identically. Bounded to keep a long
# eval run from growing without limit.
_ANSWER_CACHE: dict[str, "GenerationResult"] = {}
_ANSWER_CACHE_MAX = 2000


class GenerationError(Exception):
    """A question-level failure. Carries the spec's error vocabulary so one bad
    question degrades to `status: "error"` instead of failing the batch."""

    def __init__(self, code: str, message: str, retryable: bool):
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


@dataclass
class RawCitation:
    ref: int           # the [n] marker in the answer text
    context_rank: int  # which CONTEXT the quote was taken from
    quote: str


@dataclass
class GenerationResult:
    answer: str
    no_answer: bool
    confidence: float | None
    citations: list[RawCitation]
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    notes: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You answer questions about Uzbekistan legislation using ONLY the numbered CONTEXT articles given to you. The contexts are articles of the Constitution, the codes and the laws of the Republic of Uzbekistan, taken from lex.uz.

Hard rules:
1. Use ONLY the CONTEXT. Never use outside knowledge, and never invent an article number, a code name, a figure or a date that is not in the CONTEXT.
2. If the CONTEXT does not contain the answer, set no_answer to true and leave answer empty. A question the law does not address must come back as no_answer, not as a plausible guess. This is more important than being helpful.
3. Every factual claim in the answer must be marked with a bracketed reference -- [1], [2] -- matching an entry you list in `used`.
4. For each `used` entry, `quote` must be copied CHARACTER FOR CHARACTER from that context's text. Do not paraphrase, shorten with "...", fix spelling, or join two separate passages. A quote that is not an exact substring of the context is discarded, and the citation with it.
5. Quote the narrowest passage that actually supports the claim -- normally one sentence or one numbered item, not the whole article.
6. Numbers, time limits, amounts, percentages and BHM multiples must be copied exactly as the context writes them.
7. Answer in the requested language. Keep it short and factual: what the law says, not advice."""

# The shared answer contract. Anthropic takes this as-is; Gemini needs a variant
# without `additionalProperties` (its schema dialect rejects it); Groq gets it
# embedded in the prompt alongside JSON mode.
ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["no_answer", "answer", "used", "confidence"],
    "properties": {
        "no_answer": {
            "type": "boolean",
            "description": "true if the contexts do not answer the question",
        },
        "answer": {
            "type": "string",
            "description": "The answer, with [n] markers. Empty string when no_answer is true.",
        },
        "used": {
            "type": "array",
            "description": "One entry per bracketed reference used in the answer, in order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "context", "quote"],
                "properties": {
                    "ref": {"type": "integer", "description": "The [n] marker this supports."},
                    "context": {
                        "type": "integer",
                        "description": "The CONTEXT number the quote is taken from.",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Verbatim substring of that context's text.",
                    },
                },
            },
        },
        "confidence": {
            "type": "number",
            "description": "0..1 confidence that the answer is correct and grounded.",
        },
    },
}


def _strip_unsupported(schema: dict) -> dict:
    """Gemini's responseSchema is an OpenAPI subset and rejects
    `additionalProperties`. Everything else in the contract carries over."""
    out = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if isinstance(value, dict):
            out[key] = _strip_unsupported(value)
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: _strip_unsupported(v) for k, v in value.items()}
        else:
            out[key] = value
    return out


GEMINI_SCHEMA = _strip_unsupported(ANSWER_SCHEMA)


def build_prompt(question: str, answer_language: str, contexts: list[Context]) -> str:
    blocks = []
    for ctx in contexts:
        art = ctx.article_meta
        blocks.append(
            f"CONTEXT {ctx.rank}\n"
            f"document: {art.doc_title}\n"
            f"article: {art.article_display}"
            + (f" ({_article_title(art)})" if _article_title(art) else "")
            + f"\ntext:\n{_article_body(art)}"
        )
    language_name = lang.LANGUAGE_NAMES.get(answer_language, answer_language)
    return (
        "\n\n".join(blocks)
        + f"\n\n---\nQUESTION:\n{question}\n\n"
        f"Answer in: {language_name}\n"
    )


def _article_title(art) -> str:
    """The parenthesised title the ingest step appended to the header line."""
    m = re.search(r"\(([^)]*)\)\s*$", art.title)
    return m.group(1) if m else ""


def _article_body(art) -> str:
    """Article text without the header line that ingest prepended to every chunk."""
    body = art.text
    if body.startswith(art.title):
        body = body[len(art.title):]
    return body.lstrip("\n")


def _cache_key(question: str, answer_language: str, contexts: list[Context],
               temperature: float | None, seed: int | None) -> str:
    payload = json.dumps(
        {
            "q": lang.normalise(question),
            "l": answer_language,
            "c": [c.chunk_id for c in contexts],
            "b": config.LLM_BACKEND,
            "m": config.llm_model(),
            "t": temperature,
            "s": seed,
            "p": config.PIPELINE_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate(
    question: str,
    answer_language: str,
    contexts: list[Context],
    temperature: float | None = 0,
    seed: int | None = None,
) -> GenerationResult:
    if not contexts:
        return GenerationResult(
            answer="", no_answer=True, confidence=0.0, citations=[],
            model=config.llm_model(), notes=["no contexts retrieved"],
        )

    key = _cache_key(question, answer_language, contexts, temperature, seed)
    cached = _ANSWER_CACHE.get(key)
    if cached is not None:
        return cached

    backend = config.LLM_BACKEND
    if backend == "offline":
        result = _generate_offline(question, answer_language, contexts)
    elif backend == "anthropic":
        result = _generate_anthropic(question, answer_language, contexts)
    elif backend == "gemini":
        result = _generate_gemini(question, answer_language, contexts, temperature)
    elif backend == "groq":
        result = _generate_groq(question, answer_language, contexts, temperature, seed)
    else:
        raise GenerationError(
            "LLM_ERROR",
            f"Unknown RAG_LLM_BACKEND {backend!r}; expected anthropic, gemini, groq or offline",
            False,
        )

    if len(_ANSWER_CACHE) < _ANSWER_CACHE_MAX:
        _ANSWER_CACHE[key] = result
    return result


# --------------------------------------------------------------------------
# Shared response handling
# --------------------------------------------------------------------------
def _parse_answer_json(text: str, model: str, input_tokens: int,
                       output_tokens: int) -> GenerationResult:
    """Turn a provider's JSON string into a GenerationResult.

    Some models wrap JSON in a ```json fence even when asked not to, so the
    fence is stripped before parsing rather than failing the question over
    formatting.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise GenerationError("LLM_ERROR", f"Model returned non-JSON output: {e}", True) from e
    if not isinstance(parsed, dict):
        raise GenerationError("LLM_ERROR", "Model returned JSON that is not an object", True)

    used: list[RawCitation] = []
    for position, entry in enumerate(parsed.get("used") or [], start=1):
        try:
            used.append(
                RawCitation(
                    ref=int(entry.get("ref", position)),
                    context_rank=int(entry["context"]),
                    quote=str(entry["quote"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # a malformed entry is dropped, not fatal

    confidence = parsed.get("confidence")
    return GenerationResult(
        answer=(parsed.get("answer") or "").strip(),
        no_answer=bool(parsed.get("no_answer")),
        confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
        citations=used,
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        model=model,
    )


def _http_error(provider: str, e: httpx.HTTPStatusError) -> GenerationError:
    status = e.response.status_code
    body = e.response.text[:300]
    if status == 429:
        return GenerationError("LLM_ERROR", f"{provider} rate limited: {body}", True)
    if status in (400, 404) and "token" in body.lower():
        return GenerationError("CONTEXT_TOO_LONG", f"{provider}: {body}", False)
    if status in (401, 403):
        return GenerationError("LLM_ERROR", f"{provider} rejected the credential: {body}", False)
    return GenerationError("LLM_ERROR", f"{provider} HTTP {status}: {body}", status >= 500)


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # max_retries=0: retries are the caller's business here, because the
        # spec wants a per-question LLM timeout surfaced as LLM_TIMEOUT rather
        # than silently multiplying the batch's wall-clock.
        _client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT_S, max_retries=0)
    return _client


def _generate_anthropic(question: str, answer_language: str,
                        contexts: list[Context]) -> GenerationResult:
    prompt = build_prompt(question, answer_language, contexts)
    try:
        resp = client().messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
                # Grounded extraction from five short articles; low effort keeps
                # this inside the spec's p50 <= 5s latency target.
                "effort": "low",
            },
        )
    except anthropic.APITimeoutError as e:
        raise GenerationError(
            "LLM_TIMEOUT", f"Generation exceeded {config.LLM_TIMEOUT_S:.0f}s", True
        ) from e
    except anthropic.RateLimitError as e:
        raise GenerationError("LLM_ERROR", f"Rate limited by the model API: {e}", True) from e
    except anthropic.BadRequestError as e:
        code = "CONTEXT_TOO_LONG" if "token" in str(e).lower() else "LLM_ERROR"
        raise GenerationError(code, str(e), False) from e
    except anthropic.APIStatusError as e:
        raise GenerationError("LLM_ERROR", str(e), e.status_code >= 500) from e
    except anthropic.APIConnectionError as e:
        raise GenerationError("LLM_ERROR", f"Could not reach the model API: {e}", True) from e

    if resp.stop_reason == "refusal":
        raise GenerationError("LLM_ERROR", "Model declined to answer this question", False)

    text = "".join(b.text for b in resp.content if b.type == "text")
    return _parse_answer_json(
        text,
        resp.model or config.ANTHROPIC_MODEL,
        resp.usage.input_tokens,
        resp.usage.output_tokens,
    )


# --------------------------------------------------------------------------
# Gemini (generativelanguage REST)
# --------------------------------------------------------------------------
def _generate_gemini(question: str, answer_language: str, contexts: list[Context],
                     temperature: float | None) -> GenerationResult:
    if not config.GEMINI_API_KEY:
        raise GenerationError(
            "LLM_ERROR", "GEMINI_API_KEY is not set but RAG_LLM_BACKEND=gemini", False
        )

    url = f"{config.GEMINI_BASE_URL}/models/{config.GEMINI_MODEL}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": build_prompt(question, answer_language, contexts)}]}
        ],
        "generationConfig": {
            "temperature": 0 if temperature is None else temperature,
            "maxOutputTokens": config.LLM_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
        },
    }
    try:
        resp = httpx.post(
            url,
            json=body,
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            timeout=config.LLM_TIMEOUT_S,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise GenerationError(
            "LLM_TIMEOUT", f"Gemini exceeded {config.LLM_TIMEOUT_S:.0f}s", True
        ) from e
    except httpx.HTTPStatusError as e:
        raise _http_error("Gemini", e) from e
    except httpx.RequestError as e:
        raise GenerationError("LLM_ERROR", f"Could not reach Gemini: {e}", True) from e

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        # Gemini reports a blocked prompt here instead of as an HTTP error.
        reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
        raise GenerationError("LLM_ERROR", f"Gemini returned no answer: {reason}", False)

    candidate = candidates[0]
    finish = candidate.get("finishReason")
    if finish == "MAX_TOKENS":
        raise GenerationError(
            "CONTEXT_TOO_LONG", "Gemini hit maxOutputTokens before finishing the JSON", False
        )
    if finish in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"):
        raise GenerationError("LLM_ERROR", f"Gemini stopped: {finish}", False)

    text = "".join(
        part.get("text", "") for part in (candidate.get("content") or {}).get("parts") or []
    )
    usage = data.get("usageMetadata") or {}
    return _parse_answer_json(
        text,
        data.get("modelVersion") or config.GEMINI_MODEL,
        usage.get("promptTokenCount", 0),
        usage.get("candidatesTokenCount", 0),
    )


# --------------------------------------------------------------------------
# Groq (OpenAI-compatible REST)
# --------------------------------------------------------------------------
def _generate_groq(question: str, answer_language: str, contexts: list[Context],
                   temperature: float | None, seed: int | None) -> GenerationResult:
    if not config.GROQ_API_KEY:
        raise GenerationError(
            "LLM_ERROR", "GROQ_API_KEY is not set but RAG_LLM_BACKEND=groq", False
        )

    # Groq's JSON mode guarantees valid JSON but not this shape, so the schema
    # goes in the prompt as well. The verification in pipeline.py is what makes
    # a wrong shape harmless -- unparseable entries are simply dropped.
    system = (
        SYSTEM_PROMPT
        + "\n\nReply with a single JSON object and nothing else, matching this schema:\n"
        + json.dumps(ANSWER_SCHEMA, ensure_ascii=False)
    )
    body: dict = {
        "model": config.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": build_prompt(question, answer_language, contexts)},
        ],
        "max_tokens": config.LLM_MAX_TOKENS,
        "temperature": 0 if temperature is None else temperature,
        "response_format": {"type": "json_object"},
    }
    # Groq is the one provider here that accepts a seed, so the spec's
    # reproducibility request is honoured literally on this backend.
    if seed is not None:
        body["seed"] = seed

    try:
        resp = httpx.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            timeout=config.LLM_TIMEOUT_S,
        )
        resp.raise_for_status()
    except httpx.TimeoutException as e:
        raise GenerationError(
            "LLM_TIMEOUT", f"Groq exceeded {config.LLM_TIMEOUT_S:.0f}s", True
        ) from e
    except httpx.HTTPStatusError as e:
        raise _http_error("Groq", e) from e
    except httpx.RequestError as e:
        raise GenerationError("LLM_ERROR", f"Could not reach Groq: {e}", True) from e

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise GenerationError("LLM_ERROR", "Groq returned no choices", True)
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise GenerationError(
            "CONTEXT_TOO_LONG", "Groq hit max_tokens before finishing the JSON", False
        )

    usage = data.get("usage") or {}
    return _parse_answer_json(
        (choice.get("message") or {}).get("content") or "",
        data.get("model") or config.GROQ_MODEL,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


# --------------------------------------------------------------------------
# Offline backend -- deterministic, no network
# --------------------------------------------------------------------------
# Below this fused rerank score the top article is not a credible answer, so the
# offline backend reports no_answer instead of quoting something unrelated.
#
# This threshold is the whole of the offline backend's no_answer ability, and it
# is not enough: a question the law does not address ("how do I register a land
# parcel on Mars") still matches real articles lexically and comes back answered.
# The spec's "no_answer aniqligi >= 9/10" and "faithfulness" targets therefore
# need a real model -- judging that retrieved text does not answer the question is
# a reading task, not a scoring one. The offline backend exists to exercise the
# HTTP contract and retrieval without a credential, and it also cannot honour
# answer_language, because it quotes the (Uzbek) source verbatim.
OFFLINE_NO_ANSWER_BELOW = 0.22
_SENTENCE = re.compile(r"(?<=[.;:])\s+")


def _generate_offline(question: str, answer_language: str,
                      contexts: list[Context]) -> GenerationResult:
    top = contexts[0]
    if top.rerank_score < OFFLINE_NO_ANSWER_BELOW:
        return GenerationResult(
            answer="", no_answer=True, confidence=round(top.rerank_score, 3), citations=[],
            model="offline-extractive-v1",
            notes=[f"top rerank_score {top.rerank_score:.3f} below {OFFLINE_NO_ANSWER_BELOW}"],
        )

    body = _article_body(top.article_meta)
    # Quote the first sentence: it is the one that states the rule in Uzbek
    # statute drafting, and it is guaranteed verbatim because it is sliced
    # straight out of the context text.
    sentences = [s for s in _SENTENCE.split(body.replace("\n", " ")) if s.strip()]
    quote = sentences[0].strip() if sentences else body.strip()
    if quote not in body:
        # Paragraph joining can break an exact match; fall back to the first line.
        quote = body.split("\n")[0].strip()

    art = top.article_meta
    return GenerationResult(
        answer=f"{art.doc_title}, {art.article_display}: {quote} [1]",
        no_answer=False,
        confidence=round(min(1.0, top.rerank_score), 3),
        citations=[RawCitation(ref=1, context_rank=1, quote=quote)],
        model="offline-extractive-v1",
        notes=[
            "offline extractive backend: quotes the top-ranked article verbatim, so the "
            f"answer is in the source language regardless of answer_language={answer_language!r}, "
            "and no_answer is decided by score threshold only"
        ],
    )


def answer_cache_size() -> int:
    return len(_ANSWER_CACHE)
