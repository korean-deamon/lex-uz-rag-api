"""
List the models a provider key can actually reach, and check that generation works.

Model line-ups change faster than any hardcoded default, so rather than guessing
which id is valid, point this at whichever key you have and it prints the real
list, then offers to run one grounded question end to end.

Usage:
  python scripts/list_models.py gemini
  python scripts/list_models.py groq
  python scripts/list_models.py anthropic
  python scripts/list_models.py gemini --probe        # also answer one question
  python scripts/list_models.py gemini --probe --model gemini-2.5-flash
"""
import argparse
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402

PROBE_QUESTION = "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?"


def list_gemini() -> list[str]:
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set in .env")
    r = httpx.get(
        f"{config.GEMINI_BASE_URL}/models",
        headers={"x-goog-api-key": config.GEMINI_API_KEY},
        timeout=30,
    )
    r.raise_for_status()
    names = []
    for model in r.json().get("models", []):
        # Only models that can answer a prompt are useful here; the list also
        # includes embedding models.
        if "generateContent" in (model.get("supportedGenerationMethods") or []):
            names.append(model["name"].removeprefix("models/"))
    return sorted(names)


def list_groq() -> list[str]:
    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set in .env")
    r = httpx.get(
        f"{config.GROQ_BASE_URL}/models",
        headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def list_anthropic() -> list[str]:
    import anthropic
    client = anthropic.Anthropic()
    return [m.id for m in client.models.list()]


LISTERS = {"gemini": list_gemini, "groq": list_groq, "anthropic": list_anthropic}


def probe(provider: str, model: str | None) -> int:
    """Run one real question through the chosen provider, end to end."""
    import os
    os.environ["RAG_LLM_BACKEND"] = provider
    if model:
        os.environ[f"RAG_{provider.upper()}_MODEL"] = model

    # config was imported above, so re-point the values this run needs.
    config.LLM_BACKEND = provider
    if model:
        setattr(config, f"{provider.upper()}_MODEL", model)

    from backend import generate, retrieval

    print(f"\nprobe: {provider} / {config.llm_model()}")
    contexts, _ = retrieval.retrieve(PROBE_QUESTION, top_k=5)
    print(f"  retrieved {len(contexts)} contexts, top = {contexts[0].chunk_id}")
    try:
        result = generate.generate(PROBE_QUESTION, "uz-Latn", contexts)
    except generate.GenerationError as e:
        print(f"  FAILED  code={e.code} retryable={e.retryable}")
        print(f"          {e.message[:300]}")
        return 1

    print(f"  model      {result.model}")
    print(f"  no_answer  {result.no_answer}")
    print(f"  confidence {result.confidence}")
    print(f"  tokens     in={result.input_tokens} out={result.output_tokens}")
    print(f"  answer     {result.answer[:300]}")
    for citation in result.citations:
        quote = citation.quote
        body = generate._article_body(contexts[citation.context_rank - 1].article_meta)
        verbatim = "verbatim" if quote in body else "NOT VERBATIM (will be dropped)"
        print(f"  cite [{citation.ref}] context {citation.context_rank}: {verbatim}")
        print(f"       {quote[:160]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(LISTERS))
    parser.add_argument("--probe", action="store_true",
                        help="also answer one real question with this provider")
    parser.add_argument("--model", default=None, help="model id to probe with")
    args = parser.parse_args()

    try:
        models = LISTERS[args.provider]()
    except httpx.HTTPStatusError as e:
        print(f"{args.provider} rejected the request: HTTP {e.response.status_code}")
        print(e.response.text[:400])
        return 1
    except Exception as e:
        print(f"could not list {args.provider} models: {type(e).__name__}: {e}")
        return 1

    print(f"{args.provider}: {len(models)} models reachable with this key")
    for name in models:
        marker = "  <- configured" if name == config.llm_model() else ""
        print(f"  {name}{marker}")

    if args.probe:
        return probe(args.provider, args.model)
    print(f"\nSet it with:  RAG_{args.provider.upper()}_MODEL=<id>  in .env")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
