"""
Score a saved run against the test set: the metrics table from spec section 6.

What this computes from the run files alone:
  Hit@5 (document)     expected_doc_id present in the top-5 contexts
  Hit@5 (article)      expected_article present in the top-5 contexts
  MRR                  1/rank of the first correct article, over answerable questions
  Context precision    share of top-5 contexts from the expected document
  Faithfulness         every citation's quote verbatim in a returned context
  Citation accuracy    every citation points at the expected document
  Fabricated sources   citations outside contexts, or doc_ids not in the corpus
  Stale norms          citations whose doc_status is not "amalda"
  No-answer accuracy   unanswerable questions answered with no_answer: true
  Latency              per-question total_ms, p50 / p95
  Error share          status error or timeout

What it cannot compute, and says so instead of guessing:
  Answer correctness   needs expected_answer (0/1/2 by a human or an LLM judge).
  Hit@5 (article)/MRR  degrade to "not measurable" when expected_article is
                       blank in the test set, which it is until a human fills it.

Error attribution follows the spec's rule: each wrong answer is charged to
exactly one cause -- retrieval (the right article was never retrieved),
generation (it was retrieved but the answer is wrong or absent), or citation
(the answer is fine but the reference is not).

Usage:
  python scripts/score_eval.py --run-id eval-2026-11-03
  python scripts/score_eval.py --run-id eval-2026-11-03 --json report.json
"""
import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTSET = ROOT / "data" / "eval" / "testset_100.csv"
RUNS_DIR = ROOT / "data" / "runs"

# Spec section 6 target column, for the pass/fail marker in the report.
TARGETS = {
    "hit@5_document": (0.90, "ge"),
    "hit@5_article": (0.80, "ge"),
    "mrr": (0.70, "ge"),
    "context_precision": (0.60, "ge"),
    "faithfulness": (0.95, "ge"),
    "citation_accuracy": (0.90, "ge"),
    "fabricated_sources": (0, "eq"),
    "stale_norms": (0, "eq"),
    "no_answer_accuracy": (0.90, "ge"),
    "latency_p50_s": (5.0, "le"),
    "latency_p95_s": (12.0, "le"),
    "error_share": (0.02, "le"),
}

_APOSTROPHES = "‘’ʼ′`´'ʻ"


def normalise(text: str) -> str:
    """Compare article labels tolerantly: '480¹-modda', '480.1', '480-1' are the
    same article written three ways, and the test set may use any of them."""
    text = unicodedata.normalize("NFKD", (text or "").strip().lower())
    for ch in _APOSTROPHES:
        text = text.replace(ch, "")
    text = text.replace("modda", "").replace("статья", "")
    text = re.sub(r"[^0-9a-z]+", "", text)
    return text


def load_testset(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def load_run(run_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    """-> (results by question id, batch envelopes)."""
    results: dict[str, dict] = {}
    batches: list[dict] = []
    for path in sorted(run_dir.glob("b*.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        batches.append(batch)
        for result in batch.get("results", []):
            results[result["id"]] = result
    return results, batches


def score(full_testset: dict[str, dict], results: dict[str, dict]) -> dict:
    corpus_doc_ids = _corpus_doc_ids()

    # Score only the questions the run actually covers. Using the whole test set
    # as the denominator would make a partial run (--batches b01,b02) look
    # catastrophic rather than incomplete: 20 answered questions over 90
    # answerable rows reads as Hit@5 = 0.19. Coverage is reported separately so
    # a partial run is never mistaken for a full one.
    testset = {qid: row for qid, row in full_testset.items() if qid in results}
    not_run = [qid for qid in full_testset if qid not in results]

    answerable = [qid for qid, row in testset.items() if row["answerable"] == "true"]
    unanswerable = [qid for qid, row in testset.items() if row["answerable"] != "true"]
    labelled_article = [
        qid for qid in answerable if normalise(testset[qid]["expected_article"])
    ]

    counts = {
        "doc_hit": 0, "article_hit": 0, "no_answer_correct": 0,
        "faithful": 0, "citation_correct": 0, "fabricated": 0, "stale": 0,
        "errors": 0, "answered": 0, "missing": 0, "cited_total": 0,
    }
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    latencies: list[float] = []
    attribution = {"retrieval": 0, "generation": 0, "citation": 0}
    per_category: dict[str, dict[str, int]] = {}

    for qid, row in testset.items():
        category = row["category"]
        bucket = per_category.setdefault(category, {"n": 0, "doc_hit": 0, "ok": 0})
        bucket["n"] += 1

        result = results.get(qid)
        if result is None:
            counts["missing"] += 1
            continue

        if result["timing"].get("total_ms") is not None:
            latencies.append(result["timing"]["total_ms"] / 1000.0)

        if result["status"] in ("error", "timeout"):
            counts["errors"] += 1
            continue

        contexts = result.get("contexts") or []
        citations = result.get("citations") or []
        context_ids = {c["chunk_id"] for c in contexts}
        expected_doc = row["expected_doc_id"].strip()
        expected_article = normalise(row["expected_article"])

        # --- retrieval ---
        doc_hit = bool(expected_doc) and any(c["doc_id"] == expected_doc for c in contexts)
        if doc_hit:
            counts["doc_hit"] += 1
            bucket["doc_hit"] += 1

        article_rank = None
        if expected_article:
            for c in contexts:
                if normalise(c["article"]) == expected_article and (
                    not expected_doc or c["doc_id"] == expected_doc
                ):
                    article_rank = c["rank"]
                    break
            if article_rank:
                counts["article_hit"] += 1
                reciprocal_ranks.append(1.0 / article_rank)
            else:
                reciprocal_ranks.append(0.0)

        if expected_doc and contexts:
            precisions.append(
                sum(1 for c in contexts if c["doc_id"] == expected_doc) / len(contexts)
            )

        # --- no-answer safety ---
        if row["answerable"] != "true":
            if result["no_answer"]:
                counts["no_answer_correct"] += 1
            continue

        if result["no_answer"]:
            # An answerable question refused: charge it to whichever stage lost it.
            attribution["retrieval" if not doc_hit else "generation"] += 1
            continue

        counts["answered"] += 1
        if result["status"] == "ok":
            bucket["ok"] += 1

        # --- faithfulness and citation accuracy ---
        question_faithful = bool(citations)
        question_cited_right = bool(citations)
        for citation in citations:
            counts["cited_total"] += 1
            chunk_ok = citation["chunk_id"] in context_ids
            doc_known = citation["doc_id"] in corpus_doc_ids
            if not chunk_ok or not doc_known:
                counts["fabricated"] += 1
                question_faithful = False
            else:
                quote = citation.get("quote") or ""
                text = next(c["text"] for c in contexts if c["chunk_id"] == citation["chunk_id"])
                if _squash(quote) not in _squash(text):
                    question_faithful = False
            if citation.get("doc_status") != "amalda":
                counts["stale"] += 1
            if expected_doc and citation["doc_id"] != expected_doc:
                question_cited_right = False

        if question_faithful:
            counts["faithful"] += 1
        if question_cited_right:
            counts["citation_correct"] += 1

        # Attribution for a question that was answered but cited the wrong document.
        if not doc_hit:
            attribution["retrieval"] += 1
        elif not question_cited_right:
            attribution["citation"] += 1

    n_answerable = len(answerable) or 1
    metrics = {
        "hit@5_document": counts["doc_hit"] / n_answerable,
        "hit@5_article": (
            counts["article_hit"] / len(labelled_article) if labelled_article else None
        ),
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None,
        "context_precision": statistics.fmean(precisions) if precisions else None,
        "answer_correctness": None,  # needs expected_answer
        "faithfulness": counts["faithful"] / counts["answered"] if counts["answered"] else None,
        "citation_accuracy": (
            counts["citation_correct"] / counts["answered"] if counts["answered"] else None
        ),
        "fabricated_sources": counts["fabricated"],
        "stale_norms": counts["stale"],
        "no_answer_accuracy": (
            counts["no_answer_correct"] / len(unanswerable) if unanswerable else None
        ),
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "error_share": counts["errors"] / (len(testset) or 1),
    }
    return {
        "metrics": metrics,
        "coverage": {
            "testset_questions": len(full_testset),
            "questions_scored": len(testset),
            "not_run": len(not_run),
            "complete": not not_run,
        },
        "counts": counts | {
            "questions": len(testset),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "article_labelled": len(labelled_article),
            "citations": counts["cited_total"],
        },
        "error_attribution": attribution,
        "per_category": per_category,
    }


def _squash(text: str) -> str:
    out = unicodedata.normalize("NFC", text or "")
    for ch in "‘’ʼ′`´'":
        out = out.replace(ch, "ʻ")
    return re.sub(r"\s+", " ", out).strip()


def _corpus_doc_ids() -> set[str]:
    manifest = ROOT / "data" / "corpus" / "uzbek-legal-corpus" / "manifest.csv"
    if not manifest.exists():
        return set()
    with open(manifest, encoding="utf-8") as f:
        return {f"lex:{row['lex_uz_doc'].lstrip('-')}" for row in csv.DictReader(f)}


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank percentile: unambiguous and stable at n=100.
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[index], 3)


def render(report: dict, batches: list[dict]) -> str:
    metrics = report["metrics"]
    lines = []

    coverage = report["coverage"]
    if not coverage["complete"]:
        lines.append(
            f"PARTIAL RUN: {coverage['questions_scored']} of "
            f"{coverage['testset_questions']} questions were answered; "
            f"{coverage['not_run']} were not sent."
        )
        lines.append("Every metric below is over the answered subset only.")
        lines.append("")
    systems = {json.dumps(b.get("system", {}), sort_keys=True) for b in batches}
    lines.append("System under test")
    if len(systems) == 1 and batches:
        for key, value in (batches[0].get("system") or {}).items():
            lines.append(f"  {key:20s} {value}")
    else:
        lines.append(f"  WARNING: {len(systems)} different system versions across batches;")
        lines.append("  these results cannot be compared as one run.")
    lines.append("")

    lines.append(f"{'Metric':22s} {'Value':>10s} {'Target':>9s}  Verdict")
    lines.append("-" * 58)
    for name, value in metrics.items():
        target = TARGETS.get(name)
        if value is None:
            lines.append(f"{name:22s} {'n/a':>10s} {'':>9s}  not measurable")
            continue
        shown = f"{value:.3f}" if isinstance(value, float) else str(value)
        if not target:
            lines.append(f"{name:22s} {shown:>10s} {'':>9s}")
            continue
        threshold, direction = target
        passed = (
            value >= threshold if direction == "ge"
            else value <= threshold if direction == "le"
            else value == threshold
        )
        symbol = {"ge": ">=", "le": "<=", "eq": "=="}[direction]
        lines.append(
            f"{name:22s} {shown:>10s} {symbol + str(threshold):>9s}  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    lines.append("")
    lines.append("Counts")
    for key, value in sorted(report["counts"].items()):
        lines.append(f"  {key:20s} {value}")

    lines.append("")
    lines.append("Error attribution (one cause per wrong answer)")
    for key, value in report["error_attribution"].items():
        lines.append(f"  {key:20s} {value}")

    lines.append("")
    lines.append(f"{'Category':18s} {'n':>4s} {'doc hit@5':>10s} {'answered ok':>12s}")
    for category, bucket in sorted(report["per_category"].items()):
        rate = bucket["doc_hit"] / bucket["n"] if bucket["n"] else 0
        lines.append(f"{category:18s} {bucket['n']:>4d} {rate:>10.2f} {bucket['ok']:>12d}")

    if metrics["hit@5_article"] is None or metrics["answer_correctness"] is None:
        lines.append("")
        lines.append("Not measurable from the run files alone:")
        if metrics["hit@5_article"] is None:
            lines.append("  Hit@5 (modda) and MRR -- fill expected_article in the test set.")
        if metrics["answer_correctness"] is None:
            lines.append("  Answer correctness -- fill expected_answer, then grade 0/1/2")
            lines.append("  by hand or with an LLM judge.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    if not run_dir.exists():
        print(f"no run at {run_dir}", file=sys.stderr)
        return 1

    testset = load_testset(args.testset)
    results, batches = load_run(run_dir)
    if not results:
        print(f"no batch files in {run_dir}", file=sys.stderr)
        return 1

    report = score(testset, results)
    print(f"run_id: {args.run_id}   batches: {len(batches)}   results: {len(results)}")
    print()
    print(render(report, batches))

    if args.json:
        args.json.write_text(
            json.dumps({"run_id": args.run_id, **report}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
