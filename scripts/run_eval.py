"""
Tester-side runner: send the 100-question test set as 10 sequential batches.

This is the test scenario from spec section 1, executed literally:

  1. read the test set (questions only -- expected answers are never sent),
  2. split it into the batches the CSV already assigns (b01..b10),
  3. send them one at a time, each only after the previous response arrived,
  4. save every response as data/runs/<run_id>/bNN.json,
  5. retry a retryable failure up to 3 times with the spec's 5s / 15s / 45s
     backoff, reusing the same batch_id so the server's idempotency applies,
  6. follow the async path when the server answers 202.

Already-saved batches are skipped, so an interrupted run resumes instead of
re-sending (and re-paying for) work that is already on disk.

Usage:
  python scripts/run_eval.py                        # new run, id from today
  python scripts/run_eval.py --run-id eval-2026-11-03
  python scripts/run_eval.py --base-url http://localhost:8000 --api-key ...
  python scripts/run_eval.py --batches b01,b02      # only some batches
"""
import argparse
import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_TESTSET = ROOT / "data" / "eval" / "testset_100.csv"
RUNS_DIR = ROOT / "data" / "runs"

# Spec section 5: up to 3 retries, at 5s, 15s, 45s.
RETRY_DELAYS = (5, 15, 45)
POLL_INTERVAL_S = 3
POLL_TIMEOUT_S = 600
# The server guarantees at least 1 batch/second; an idempotent replay returns
# fast enough for two sends to land inside the same second and trip the limiter,
# so the client paces itself rather than earning a 429 it then has to retry.
MIN_SEND_INTERVAL_S = 1.1


def load_batches(testset: Path) -> dict[str, list[dict]]:
    batches: dict[str, list[dict]] = {}
    with open(testset, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            batches.setdefault(row["batch"], []).append(row)
    return dict(sorted(batches.items()))


def build_request(run_id: str, batch_name: str, index: int, total: int,
                  rows: list[dict], top_k: int) -> dict:
    """Only id / question / language cross the wire. expected_* stay here."""
    items = []
    for row in rows:
        item = {"id": row["id"], "question": row["question"]}
        if row.get("language"):
            item["language"] = row["language"]
        items.append(item)
    return {
        "run_id": run_id,
        "batch_id": f"{run_id}-{batch_name}",
        "batch_index": index,
        "total_batches": total,
        "options": {
            "top_k": top_k,
            "return_contexts": True,
            "return_debug": True,
            "temperature": 0,
            "seed": 42,
            "answer_language": "auto",
        },
        "items": items,
    }


def poll_until_done(client: httpx.Client, batch_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
        r = client.get(f"/v1/rag/batch/{batch_id}")
        if r.status_code == 200:
            return r.json()
        if r.status_code != 202:
            raise RuntimeError(f"poll failed {r.status_code}: {r.text[:300]}")
        print(f"      ... still processing")
    raise RuntimeError(f"batch {batch_id} did not finish within {POLL_TIMEOUT_S}s")


def send_batch(client: httpx.Client, body: dict) -> dict:
    batch_id = body["batch_id"]
    last_error = None

    for attempt in range(len(RETRY_DELAYS) + 1):
        if attempt:
            delay = RETRY_DELAYS[attempt - 1]
            print(f"      retry {attempt}/{len(RETRY_DELAYS)} in {delay}s ({last_error})")
            time.sleep(delay)

        try:
            r = client.post("/v1/rag/batch", json=body)
        except httpx.RequestError as e:
            last_error = f"connection: {type(e).__name__}"
            continue

        if r.status_code == 200:
            if r.headers.get("X-Idempotent-Replay") == "true":
                print("      (stored result replayed by the server)")
            return r.json()

        if r.status_code == 202:
            print(f"      accepted async, polling {batch_id}")
            return poll_until_done(client, batch_id)

        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "1"))
            print(f"      rate limited, waiting {wait}s")
            time.sleep(wait)
            last_error = "429"
            continue

        if r.status_code in (500, 503, 504):
            last_error = f"{r.status_code}"
            continue

        # 400 / 401 / 403 / 409 are our mistake, not a transient one.
        raise RuntimeError(f"{r.status_code}: {r.text[:400]}")

    raise RuntimeError(f"batch {batch_id} failed after retries ({last_error})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"eval-{date.today().isoformat()}")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None,
                        help="defaults to the first key in RAG_API_KEYS")
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batches", default=None, help="comma-separated, e.g. b01,b02")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        from backend import config
        api_key = config.API_KEYS[0]

    batches = load_batches(args.testset)
    if args.batches:
        wanted = {b.strip() for b in args.batches.split(",")}
        batches = {k: v for k, v in batches.items() if k in wanted}
    if not batches:
        print("no batches to run", file=sys.stderr)
        return 1

    out_dir = RUNS_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_id   : {args.run_id}")
    print(f"base_url : {args.base_url}")
    print(f"batches  : {len(batches)} x {len(next(iter(batches.values())))} questions")
    print(f"output   : {out_dir}")
    print()

    client = httpx.Client(
        base_url=args.base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=args.timeout,
    )

    total = len(batches)
    started = time.monotonic()
    with client:
        # A health check first: a whole run against a degraded index is wasted.
        try:
            health = client.get("/v1/health").json()
            print(f"health   : {health['status']} ({health['checks']})")
            print()
        except Exception as e:
            print(f"cannot reach {args.base_url}: {e}", file=sys.stderr)
            return 1

        last_send = 0.0
        for index, (batch_name, rows) in enumerate(batches.items(), start=1):
            target = out_dir / f"{batch_name}.json"
            if target.exists():
                print(f"[{index}/{total}] {batch_name}: already saved, skipping")
                continue

            gap = MIN_SEND_INTERVAL_S - (time.monotonic() - last_send)
            if last_send and gap > 0:
                time.sleep(gap)
            last_send = time.monotonic()

            body = build_request(args.run_id, batch_name, index, total, rows, args.top_k)
            print(f"[{index}/{total}] {batch_name}: sending {len(rows)} questions...")
            t0 = time.monotonic()
            try:
                response = send_batch(client, body)
            except Exception as e:
                print(f"      FAILED: {e}", file=sys.stderr)
                return 1

            target.write_text(
                json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary = response.get("summary", {})
            print(
                f"      {time.monotonic() - t0:.1f}s  ok={summary.get('ok')} "
                f"no_answer={summary.get('no_answer')} error={summary.get('error')} "
                f"-> {target.name}"
            )

    print()
    print(f"done in {time.monotonic() - started:.1f}s. Score it with:")
    print(f"  python scripts/score_eval.py --run-id {args.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
