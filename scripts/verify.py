"""
One command that answers "is this ready for the test?".

Runs every check that can be run locally, in dependency order, and prints a
single verdict at the end. Each step says what it proved, not just pass/fail,
so a FAIL tells you what to fix rather than that something is wrong.

What it can and cannot establish:
  * Steps 1-5 need no model credential. They cover the whole HTTP contract,
    retrieval quality, and the spec's section 7 checklist.
  * Step 6 is the only one that needs a provider key. Without one, the verdict
    is "ready except generation" -- which is an honest state, not a pass.

Usage:
  python scripts/verify.py
  python scripts/verify.py --quick        # skip the 100-question eval
  python scripts/verify.py --base-url http://localhost:8000   # also test a live server
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

# Questions whose answering article is known from reading the statute, used to
# check that retrieval still finds them. Kept small and verified by hand -- this
# is a smoke test, not the eval.
RETRIEVAL_CHECKS = [
    ("Yillik asosiy mehnat taʻtilining eng kam muddati qancha?", "mehnat_kodeksi", "217"),
    ("Jinoyat kodeksiga koʻra oʻgʻirlik uchun javobgarlik qanday?", "jinoyat_kodeksi", "169"),
    ("Жиноят кодексига кўра ўғрилик учун жавобгарлик қандай?", "jinoyat_kodeksi", "169"),
    ("Какой общий срок исковой давности установлен Гражданским кодексом?",
     "fuqarolik_kodeksi_1qism", "150"),
    ("Oʻzbekiston Respublikasi Konstitutsiyasiga koʻra davlat tili qaysi til?",
     "konstitutsiya", "4"),
]


class Report:
    def __init__(self) -> None:
        self.steps: list[tuple[str, str, list[str]]] = []

    def add(self, name: str, status: str, details: list[str]) -> str:
        self.steps.append((name, status, details))
        symbol = {PASS: "PASS", FAIL: "FAIL", WARN: "WARN", SKIP: "SKIP"}[status]
        print(f"[{symbol}] {name}")
        for line in details:
            print(f"       {line}")
        print()
        return status

    def count(self, status: str) -> int:
        return sum(1 for _, s, _ in self.steps if s == status)


def step_environment(report: Report) -> str:
    details = [f"python {sys.version.split()[0]}"]
    missing = []
    for module in ("fastapi", "pydantic", "qdrant_client", "sentence_transformers",
                   "httpx", "pytest", "anthropic"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if sys.version_info < (3, 11):
        details.append("python 3.11+ is required")
        return report.add("Environment", FAIL, details)
    if missing:
        details.append(f"missing packages: {', '.join(missing)}")
        details.append("fix: pip install -r requirements.txt")
        return report.add("Environment", FAIL, details)
    details.append("all required packages importable")

    # A missing .env is the most common first-run mistake. The code default is
    # RAG_LLM_BACKEND=anthropic -- deliberately, so a misconfigured provider
    # fails loudly instead of silently serving the offline answerer's extractive
    # quotes as if a model had written them. But that means a forgotten .env
    # surfaces later as a confusing generation failure, so name it here instead.
    if not (ROOT / ".env").exists():
        details.append("no .env file -- run: cp .env.example .env")
        details.append("without it the provider defaults to anthropic, and the")
        details.append("Generation step below will FAIL for lack of a key")
        return report.add("Environment", WARN, details)

    return report.add("Environment", PASS, details)


def step_qdrant(report: Report) -> str:
    from backend import config, retrieval
    try:
        client = retrieval.qdrant()
        names = [c.name for c in client.get_collections().collections]
    except Exception as e:
        return report.add("Qdrant", FAIL, [
            f"cannot reach {config.QDRANT_HOST}:{config.QDRANT_PORT} ({type(e).__name__})",
            "fix: docker compose up -d",
        ])

    if config.COLLECTION not in names:
        return report.add("Qdrant", FAIL, [
            f"collection {config.COLLECTION!r} does not exist (found: {names})",
            "fix: python ingest/ingest_corpus.py",
        ])

    count = client.count(config.COLLECTION, exact=True).count
    details = [f"collection {config.COLLECTION!r}: {count} points"]
    if count == 0:
        details.append("fix: python ingest/ingest_corpus.py")
        return report.add("Qdrant", FAIL, details)
    return report.add("Qdrant", PASS, details)


def step_index(report: Report) -> str:
    from backend import retrieval
    t0 = time.monotonic()
    try:
        size = retrieval.index_size()
    except Exception as e:
        return report.add("Article index", FAIL, [f"{type(e).__name__}: {e}"])

    details = [f"{size} articles loaded in {time.monotonic() - t0:.1f}s"]
    excluded = retrieval.excluded_point_counts()
    if excluded.get("non_article_points_excluded"):
        details.append(
            f"{excluded['non_article_points_excluded']} non-article points excluded "
            "(raw-scrape chunks; they cannot carry article/part)"
        )
    if size < 5000:
        details.append(f"expected ~7383 articles; only {size} loaded")
        return report.add("Article index", WARN, details)
    return report.add("Article index", PASS, details)


def step_retrieval(report: Report) -> str:
    from backend import retrieval
    details = []
    hits_at_1 = 0
    hits_at_5 = 0
    for question, code, article in RETRIEVAL_CHECKS:
        contexts, _ = retrieval.retrieve(question, top_k=5)
        found = [
            i for i, c in enumerate(contexts, start=1)
            if c.article_meta.code == code and c.article_meta.article_id == article
        ]
        rank = found[0] if found else None
        if rank == 1:
            hits_at_1 += 1
            mark = "rank 1"
        elif rank:
            hits_at_5 += 1
            mark = f"rank {rank}"
        else:
            mark = "NOT FOUND"
        details.append(f"{mark:10s} expected {code} {article}-modda  <- {question[:46]}")

    total = len(RETRIEVAL_CHECKS)
    details.append(f"{hits_at_1}/{total} at rank 1, {hits_at_1 + hits_at_5}/{total} in top-5")
    if hits_at_1 + hits_at_5 < total:
        return report.add("Retrieval (3 languages)", FAIL, details)
    if hits_at_1 < total - 1:
        return report.add("Retrieval (3 languages)", WARN, details)
    return report.add("Retrieval (3 languages)", PASS, details)


def step_contract_tests(report: Report) -> str:
    """The spec's section 7 checklist, as the test suite.

    Forced onto the offline generator. The checklist is about the HTTP contract
    and retrieval, neither of which depends on the provider, and importing
    backend.config above has already loaded .env -- so without this the
    subprocess inherits whatever provider .env names. That made the suite call a
    real model for every question: three times slower, costing money, and flaky,
    because a free tier's rate limit turns into a failed test that says nothing
    about the contract. Generation is checked on its own, in the next step.
    """
    env = {**os.environ, "RAG_LLM_BACKEND": "offline", "RAG_GEN_MIN_INTERVAL_S": "0"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    tail = [line for line in proc.stdout.strip().splitlines() if line.strip()][-1:]
    details = tail or ["pytest produced no summary line"]
    if proc.returncode != 0:
        details.append("fix: run `pytest tests/ -q` to see which checklist item broke")
        return report.add("Spec section 7 checklist", FAIL, details)
    return report.add("Spec section 7 checklist", PASS, details)


def step_provider(report: Report) -> str:
    """The only step that needs a credential."""
    from backend import config, generate, retrieval

    provider = config.resolve()
    how = "auto-detected" if provider.detected else "set in .env"
    details = [
        f"provider: {provider.name} ({how})",
        f"model:    {provider.model or '(not set)'}",
    ]
    if provider.kind == "offline":
        details.append("offline generator: no credential needed, and no real answers")
        details.append("it cannot judge no_answer and cannot honour answer_language")
        details.append("fix: put any provider key in .env -- with RAG_LLM_BACKEND=auto")
        details.append("     the key itself picks the provider. See TESTING.md.")
        return report.add("Generation", SKIP, details)

    question = RETRIEVAL_CHECKS[0][0]
    contexts, _ = retrieval.retrieve(question, top_k=5)
    try:
        result = generate.generate(question, "uz-Latn", contexts, temperature=0, seed=42)
    except generate.GenerationError as e:
        details.append(f"{e.code} (retryable={e.retryable}): {e.message[:160]}")
        details.append("the API still answers 200 with per-question errors, as the spec requires,")
        details.append("but no answers are generated until this is fixed")
        return report.add("Generation", FAIL, details)

    details.append(f"answered by {result.model}, tokens in={result.input_tokens} out={result.output_tokens}")
    details.append(f"no_answer={result.no_answer} confidence={result.confidence}")
    details.append(f"answer: {result.answer[:110]}")

    # The citation must survive verification, or the answer is not grounded.
    verified = 0
    for citation in result.citations:
        body = generate._article_body(contexts[citation.context_rank - 1].article_meta)
        if citation.quote in body:
            verified += 1
    details.append(f"{verified}/{len(result.citations)} citations verbatim in their context")
    if result.citations and verified == 0:
        details.append("every quote failed verification, so every citation would be dropped")
        return report.add("Generation", WARN, details)
    return report.add("Generation", PASS, details)


def step_live_server(report: Report, base_url: str) -> str:
    import httpx
    from backend import config
    try:
        health = httpx.get(f"{base_url}/v1/health", timeout=10).json()
    except Exception as e:
        return report.add("Live server", FAIL, [
            f"cannot reach {base_url} ({type(e).__name__})",
            "fix: uvicorn backend.main:app --port 8000",
        ])

    details = [f"{base_url} health={health.get('status')} {health.get('checks')}"]
    headers = {"Authorization": f"Bearer {config.API_KEYS[0]}"}
    body = {
        "run_id": "verify", "batch_id": f"verify-{int(time.time())}",
        "batch_index": 1, "total_batches": 1,
        "options": {"top_k": 5, "return_contexts": True},
        "items": [{"id": "q001", "question": RETRIEVAL_CHECKS[0][0]}],
    }
    try:
        r = httpx.post(f"{base_url}/v1/rag/batch", json=body, headers=headers, timeout=180)
    except Exception as e:
        details.append(f"batch request failed: {type(e).__name__}: {e}")
        return report.add("Live server", FAIL, details)

    details.append(f"POST /v1/rag/batch -> HTTP {r.status_code}")
    if r.status_code != 200:
        details.append(r.text[:200])
        return report.add("Live server", FAIL, details)

    data = r.json()
    result = data["results"][0]
    details.append(f"summary={data['summary']}  status={result['status']}")
    details.append(f"contexts={len(result['contexts'])} citations={len(result['citations'])}")

    # Auth must actually be enforced, or the test key means nothing.
    unauth = httpx.post(f"{base_url}/v1/rag/batch", json=body, timeout=30)
    details.append(f"no auth header -> HTTP {unauth.status_code} (expected 401)")
    if unauth.status_code != 401:
        return report.add("Live server", FAIL, details)
    return report.add("Live server", PASS, details)


def step_full_eval(report: Report, base_url: str | None) -> str:
    """The spec's scenario: 100 questions as 10 sequential batches, then scored."""
    if not base_url:
        return report.add("100-question eval", SKIP, [
            "needs a running server; re-run with --base-url http://localhost:8000",
        ])

    run_id = f"verify-{time.strftime('%Y%m%d-%H%M%S')}"
    proc = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--run-id", run_id, "--base-url", base_url],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return report.add("100-question eval", FAIL, [
            "run_eval.py failed:",
            *proc.stdout.strip().splitlines()[-4:],
            *proc.stderr.strip().splitlines()[-3:],
        ])

    scored = subprocess.run(
        [sys.executable, "scripts/score_eval.py", "--run-id", run_id],
        cwd=ROOT, capture_output=True, text=True,
    )
    lines = scored.stdout.splitlines()
    table = [l for l in lines if any(t in l for t in ("PASS", "FAIL", "not measurable"))]
    failures = [l for l in table if "FAIL" in l]
    details = [f"run_id={run_id}", *table]
    if failures:
        details.append(f"{len(failures)} metric(s) below target")
        return report.add("100-question eval", WARN, details)
    return report.add("100-question eval", PASS, details)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=None,
                        help="also check a running server, and run the 100-question eval")
    parser.add_argument("--quick", action="store_true", help="skip the 100-question eval")
    args = parser.parse_args()

    print("=" * 62)
    print("lex.uz RAG API -- readiness check")
    print("=" * 62)
    print()

    report = Report()

    if step_environment(report) == FAIL:
        print("Environment is not usable; nothing else can be checked.")
        return 1
    if step_qdrant(report) == FAIL:
        print("Qdrant or the index is not ready; retrieval cannot be checked.")
        return 1

    step_index(report)
    step_retrieval(report)
    step_contract_tests(report)
    generation = step_provider(report)

    if args.base_url:
        step_live_server(report, args.base_url)
    if not args.quick:
        step_full_eval(report, args.base_url)

    # --- verdict ---
    print("=" * 62)
    failed = report.count(FAIL)
    warned = report.count(WARN)
    print(f"{report.count(PASS)} passed, {warned} warning(s), {failed} failed, "
          f"{report.count(SKIP)} skipped")
    print()

    if failed:
        print("VERDICT: NOT READY. Fix the FAIL steps above.")
        return 1
    if generation in (SKIP, WARN):
        print("VERDICT: READY EXCEPT GENERATION.")
        print()
        print("  The API contract, retrieval and the section 7 checklist all pass, so")
        print("  the service is testable today. What is missing is a working provider")
        print("  key: until one is set, questions come back with status 'error' /")
        print("  LLM_ERROR (contexts still returned), and the spec's no_answer,")
        print("  faithfulness and answer-correctness metrics cannot be scored.")
        print()
        print("  Next: put a key in .env, set RAG_LLM_BACKEND=gemini|groq|anthropic,")
        print("  then:  python scripts/list_models.py <provider> --probe")
        return 0

    print("VERDICT: READY.")
    if warned:
        print("Review the warnings above before the scored run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
