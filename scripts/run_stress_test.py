"""
Stress test: run every question in data/stress_test_100_uz.json through the real RAG
pipeline (backend.rag.answer), batch by batch, appending each result to a JSONL file
as it completes so nothing is lost if the run gets interrupted.

Run: python scripts/run_stress_test.py [start_index] [end_index]
  no args -> runs the whole file
  with args -> runs only that slice (for resuming / chunked runs)
"""
import sys
import json
import time
from pathlib import Path

from backend.rag import answer

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "data" / "stress_test_100_uz.json"
RESULTS_PATH = ROOT / "data" / "stress_test_results.jsonl"


def already_done_indices() -> set[int]:
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["idx"])
        except Exception:
            pass
    return done


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(questions)

    done = already_done_indices()

    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        for i in range(start, min(end, len(questions))):
            if i in done:
                continue
            item = questions[i]
            t0 = time.time()
            try:
                r = answer(item["q"])
                record = {
                    "idx": i,
                    "code": item["code"],
                    "question": item["q"],
                    "answer": r["answer"],
                    "sources": r["sources"],
                    "elapsed_s": round(time.time() - t0, 2),
                    "error": None,
                }
            except Exception as e:
                record = {
                    "idx": i,
                    "code": item["code"],
                    "question": item["q"],
                    "answer": None,
                    "sources": [],
                    "elapsed_s": round(time.time() - t0, 2),
                    "error": str(e),
                }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}] {item['code']}: {'OK' if not record['error'] else 'ERROR: ' + record['error']}")


if __name__ == "__main__":
    main()
