# Testing guide

Everything you need to set this up and test it, assuming you have not seen the
project before. Written for the person doing the testing — if you only want to
know what the thing *is*, read [README.md](README.md) instead.

**What you are testing:** a REST API that answers up to 10 questions per request
about Uzbekistan legislation, returning for each question an answer, the lex.uz
sources (document + article), and the context chunks it retrieved. It is built
to the specification in `RAG API — test spetsifikatsiyasi (lex.uz).pdf`, which is
in this repo — the spec is the contract, this guide is how to check it.

**Read this first:** two things are *expected* to fail right now. Jump to
[Known failures](#known-failures-do-not-report-these-as-bugs) before you file
anything, so you don't spend time on them.

---

## 1. Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 or 3.13 recommended; tested on 3.14 |
| Docker | any recent | Only used to run Qdrant (the vector store) |
| RAM | 4 GB free | The embedding model runs on CPU |
| Disk | ~3 GB free | ~1.1 GB embedding model + ~1.5 GB Python packages |
| Network | once, at setup | Downloads packages and the embedding model. The API itself needs no internet except to reach your LLM provider |
| GPU | not needed | Everything runs on CPU |

**An LLM provider key is optional for most of this guide.** Steps 1–7 need no
key. Only [step 8](#8-generation-needs-a-provider-key) does. See
[Providers](#providers) for which keys work.

The repo ships the legal corpus it needs (`data/corpus/.../data/articles/`,
~12 MB, CC0). You do not need to download it separately, and you should not
replace it with a newer copy — the version is pinned so your scores stay
comparable with ours.

`requirements.txt` is the API, the tests and the evaluation. You do **not** need
`requirements-scraper.txt`: that is only for the lex.uz scraper and its daily
Airflow job in `orchestration/`, which nothing in this guide uses, and it pulls
in Airflow plus a ~400 MB browser. That job writes to a separate Qdrant
collection (`lex_uz_raw`) and cannot affect what you are testing.

---

## 2. Setup

Pick your platform. The code is identical; only shell syntax differs.

### Linux

```bash
git clone <repo-url>
cd <repo-dir>
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
```

On a CPU-only machine, install torch first so pip does not pull several GB of
CUDA packages you will never use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### macOS

Same as Linux. Install Python with Homebrew (`brew install python@3.12`) and
Docker Desktop from docker.com, then **launch Docker Desktop** before
`docker compose up -d`. Apple Silicon needs no extra setup.

### Windows (PowerShell)

Install Python from python.org — tick **Add python.exe to PATH** — and Docker
Desktop with the WSL2 backend.

```powershell
git clone <repo-url>
cd <repo-dir>
Copy-Item .env.example .env
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
```

If activating the venv is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Do this in every PowerShell window before running anything**, or Uzbek letters
print as `?`. This is your console's encoding, not the API's data:

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
```

Also: in PowerShell, `curl` is an alias for `Invoke-WebRequest` and does **not**
accept `-d`. Every `curl` example below has a PowerShell version next to it. If
you prefer real curl, call `curl.exe` explicitly.

### Build the index (once)

```bash
python ingest/ingest_corpus.py
```

Takes 10–25 minutes on CPU the first time: it downloads the embedding model
(~1.1 GB) and embeds 7383 articles. It is resumable — if you interrupt it, run
it again and it picks up where it stopped. Skip this if the collection already
has points (the next step tells you).

---

## 3. One command: is it ready?

```bash
python scripts/verify.py --quick
```

This is the fastest way to know whether anything is wrong. It checks, in order:
the environment, Qdrant, the article index, retrieval against five
hand-verified questions in three languages, the full spec checklist (46 tests),
and the configured LLM provider — then prints one verdict.

**Expected output right now** (with no provider key):

```
[PASS] Environment
[PASS] Qdrant                     collection 'lex_uz': 7443 points
[PASS] Article index              7383 articles loaded
[PASS] Retrieval (3 languages)    5/5 at rank 1, 5/5 in top-5
[PASS] Spec section 7 checklist   46 passed, 1 skipped
[SKIP] Generation                 RAG_LLM_BACKEND=offline

5 passed, 0 warning(s), 0 failed, 1 skipped
VERDICT: READY EXCEPT GENERATION.
```

If a step says `FAIL`, it also prints the fix. Any `FAIL` here is a real problem
worth reporting. See [Troubleshooting](#troubleshooting) for the common ones.

The longer form additionally starts a 100-question evaluation against a running
server (details in [step 7](#7-the-100-question-evaluation)):

```bash
# terminal 1
uvicorn backend.main:app --port 8000
# terminal 2
python scripts/verify.py --base-url http://localhost:8000
```

---

## 4. Start the server

```bash
uvicorn backend.main:app --port 8000
```

First start takes ~30 seconds: it loads the 7383-article index and warms the
embedding model so the first question is not slow. Wait for
`retrieval index ready` and `embedding model ready` in the log.

Check it:

```bash
curl -s localhost:8000/v1/health
```
```powershell
Invoke-RestMethod localhost:8000/v1/health | ConvertTo-Json -Depth 5
```

Expected: `status: "ok"`, `qdrant: "ok"`, `articles_indexed: 7383`.

`status: "degraded"` with HTTP 503 means it cannot retrieve — that is the health
check doing its job, not a false alarm. Fix the cause before testing anything
else.

The API key for all requests below is `test-key-lex-2026` (set in `.env` as
`RAG_API_KEYS`). Change it there if you want to test with a different one.

---

## 5. Endpoint tests

Five endpoints. The spec requires four of them; `/v1/rag/query` is optional and
exists for debugging.

### 5.1 `POST /v1/rag/batch` — the main one

```bash
curl -s -X POST localhost:8000/v1/rag/batch \
  -H 'Authorization: Bearer test-key-lex-2026' \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{
    "run_id": "manual", "batch_id": "manual-b01",
    "batch_index": 1, "total_batches": 1,
    "options": {"top_k": 5, "return_contexts": true, "return_debug": true,
                "temperature": 0, "seed": 42},
    "items": [
      {"id": "q001", "question": "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?"},
      {"id": "q002", "question": "Какой общий срок исковой давности установлен Гражданским кодексом?", "language": "ru"},
      {"id": "q003", "question": "Жиноят кодексига кўра ўғрилик учун жавобгарлик қандай?", "language": "uz-Cyrl"}
    ]
  }' | python3 -m json.tool
```

```powershell
$body = @{
  run_id = "manual"; batch_id = "manual-b01"; batch_index = 1; total_batches = 1
  options = @{ top_k = 5; return_contexts = $true; return_debug = $true
               temperature = 0; seed = 42 }
  items = @(
    @{ id = "q001"; question = "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?" },
    @{ id = "q002"; question = "Какой общий срок исковой давности установлен Гражданским кодексом?"; language = "ru" },
    @{ id = "q003"; question = "Жиноят кодексига кўра ўғрилик учун жавобгарлик қандай?"; language = "uz-Cyrl" }
  )
} | ConvertTo-Json -Depth 6

Invoke-RestMethod localhost:8000/v1/rag/batch -Method Post `
  -Headers @{ Authorization = "Bearer test-key-lex-2026" } `
  -ContentType "application/json; charset=utf-8" -Body $body |
  ConvertTo-Json -Depth 8
```

**Check in the response:**

| What | Expected |
|---|---|
| HTTP status | `200` |
| `Content-Type` header | `application/json; charset=utf-8` |
| `status` | `"completed"` |
| `results` length | exactly 3, one per `items` entry |
| `results[].id` | `q001`, `q002`, `q003` — each exactly once |
| `summary` | `ok + no_answer + error` equals `total` |
| `system` | all six fields non-empty: `pipeline_version`, `llm_provider`, `llm_model`, `embedding_model`, `reranker`, `index_version`, `lex_snapshot_date` |
| `timing` | `received_at`, `completed_at`, `total_ms` present |
| `results[].answer_language` | `uz-Latn`, `ru`, `uz-Cyrl` respectively |
| `results[].citations[]` | each has `doc_id` starting `lex:`, a `https://lex.uz/` URL, an `article` containing `modda`, plus `part`, `edition_date`, `doc_status`, `chunk_id`, `quote` — none empty |
| `results[].contexts[]` | ≤ 5 entries, `rank` 1..n ascending, `rerank_score` descending, every `text` non-empty |
| Uzbek letters | `oʻ` and `gʻ` intact, not `o'`, not `?`, not `ʻ` |

**Two rules worth checking by hand — they are the point of the whole design:**

1. **Every `citations[].chunk_id` must also appear in that question's
   `contexts[]`.** If it does not, the system cited a source it never retrieved.
2. **Every `citations[].quote` must be an exact substring of the `text` of the
   context with that `chunk_id`.** If it is not, the answer is not grounded in
   the source.

Both are enforced in code and covered by tests, but check one by eye anyway —
it takes a minute and it is the thing the spec cares most about:

```bash
python3 - <<'EOF'
import json, subprocess
r = subprocess.run(["curl","-s","-X","POST","localhost:8000/v1/rag/batch",
  "-H","Authorization: Bearer test-key-lex-2026",
  "-H","Content-Type: application/json",
  "-d",json.dumps({"run_id":"chk","batch_id":"chk-b01","batch_index":1,"total_batches":1,
    "options":{"return_contexts":True},
    "items":[{"id":"q001","question":"Yillik asosiy mehnat taʻtilining eng kam muddati qancha?"}]})],
  capture_output=True, text=True)
for res in json.loads(r.stdout)["results"]:
    ctx = {c["chunk_id"]: c["text"] for c in res["contexts"]}
    for c in res["citations"]:
        in_ctx = c["chunk_id"] in ctx
        verbatim = in_ctx and c["quote"] in ctx[c["chunk_id"]]
        print(f"{res['id']} ref{c['ref']}: chunk_in_contexts={in_ctx} quote_verbatim={verbatim}")
EOF
```

Both must print `True`.

### 5.2 `GET /v1/rag/batch/{batch_id}` — re-read a result

```bash
curl -s localhost:8000/v1/rag/batch/manual-b01 \
  -H 'Authorization: Bearer test-key-lex-2026' | python3 -m json.tool
```

Expected: HTTP `200` and **byte-identical** to what the POST returned.
Unknown id → HTTP `404`, `error.code = "BATCH_NOT_FOUND"`.

### 5.3 `GET /v1/meta`

```bash
curl -s localhost:8000/v1/meta -H 'Authorization: Bearer test-key-lex-2026' | python3 -m json.tool
```

Expected: the `system` block, `index` (7383 articles, 25 documents, chunking
described), all 25 documents with `lex:` ids and lex.uz URLs, the three
languages, the limits, and a `caveats` array. **Read the caveats** — they state
the index's real limitations, and several of the items in
[Known failures](#known-failures-do-not-report-these-as-bugs) are explained
there.

### 5.4 `GET /v1/health`

Deliberately unauthenticated, so a monitor does not need the test key.

```bash
curl -s -o /dev/null -w "%{http_code}\n" localhost:8000/v1/health     # 200
```

### 5.5 `POST /v1/rag/query` — single question (optional endpoint)

```bash
curl -s -X POST localhost:8000/v1/rag/query \
  -H 'Authorization: Bearer test-key-lex-2026' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Oila kodeksiga koʻra nikoh yoshi qancha?"}' | python3 -m json.tool
```

---

## 6. Error and edge cases

Each row is a separate request. Change `batch_id` every time unless the row says
otherwise, or you will hit the idempotency rule instead of the case you meant to
test. **Also wait ~1 second between requests** — the server accepts at most one
batch per second and will answer `429` otherwise (which is itself correct
behaviour; see the rate-limit row).

| # | What to send | Expected |
|---|---|---|
| 1 | 11 items in `items` | `400`, `error.code = "VALIDATION_FAILED"` |
| 2 | `items: []` | `400` |
| 3 | Two items with the same `id` | `400`, message mentions uniqueness |
| 4 | An item with `"question": "   "` | `400` |
| 5 | A question longer than 2000 characters | `400` |
| 6 | No `Authorization` header | `401`, `error.code = "UNAUTHENTICATED"` |
| 7 | `Authorization: test-key-lex-2026` (no `Bearer `) | `401` |
| 8 | `Authorization: Bearer wrong-key` | `403`, `error.code = "FORBIDDEN"` |
| 9 | **Same** `batch_id` as 5.1, **same** questions | `200`, header `X-Idempotent-Replay: true`, body identical to the first response — and it must not take as long, because it was not recomputed |
| 10 | **Same** `batch_id` as 5.1, **different** questions | `409`, `error.code = "BATCH_ID_CONFLICT"` |
| 11 | Two batches back to back with no pause | second one `429`, with a `Retry-After` header ≥ 1 |
| 12 | An item with `"language": "klingon"` alongside two valid items | `200` overall; that item `status = "error"`, `error.code = "INVALID_LANGUAGE"`; the **other two still answered** — one bad question must not fail the batch |
| 13 | `GET /v1/meta` with no auth header | `401` |
| 14 | `"options": {"return_contexts": false}` | `200`, `contexts` is `[]` |
| 15 | `"options": {"top_k": 3}` | at most 3 contexts per question |

Example for #1 and #12:

```bash
# #1 -- eleven questions
python3 -c "
import json
print(json.dumps({'run_id':'e','batch_id':'err-11','batch_index':1,'total_batches':1,
 'items':[{'id':f'q{i:03d}','question':'test savol'} for i in range(1,12)]}))" > /tmp/eleven.json
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/v1/rag/batch \
  -H 'Authorization: Bearer test-key-lex-2026' -H 'Content-Type: application/json' \
  -d @/tmp/eleven.json        # expect 400

# #12 -- one bad language among good questions
curl -s -X POST localhost:8000/v1/rag/batch \
  -H 'Authorization: Bearer test-key-lex-2026' -H 'Content-Type: application/json' \
  -d '{"run_id":"e","batch_id":"err-lang","batch_index":1,"total_batches":1,
       "items":[{"id":"q001","question":"Oila kodeksiga koʻra nikoh yoshi qancha?"},
                {"id":"q002","question":"test","language":"klingon"},
                {"id":"q003","question":"Yer kodeksiga koʻra yer uchastkasi nima?"}]}' \
  | python3 -c "import json,sys; [print(r['id'], r['status'], r['error']['code'] if r['error'] else '') for r in json.load(sys.stdin)['results']]"
```

### Determinism

Send the same question twice under two different `batch_id`s with
`"temperature": 0, "seed": 42`. The `contexts[].chunk_id` list and the
`citations[].chunk_id` list must be identical in the same order.

Note what this does and does not claim: **retrieval is deterministic by
construction**, and identical requests return identical answers because they are
cached by request identity. Whether the *model* honours `temperature`/`seed` is
a provider matter — see the table in [Providers](#providers). `GET /v1/meta`
states which case applies to your configuration.

### Async mode (202)

The spec requires the async path when a batch exceeds 120 seconds. On this
hardware a 10-question batch finishes in ~2 seconds, so you cannot trigger it
naturally — force it by lowering the deadline:

```bash
# stop the server, then restart it with:
RAG_SYNC_DEADLINE_S=0.001 uvicorn backend.main:app --port 8000
```
```powershell
$env:RAG_SYNC_DEADLINE_S = "0.001"
uvicorn backend.main:app --port 8000
```

Now a batch returns `202` with `status: "processing"`. Poll
`GET /v1/rag/batch/{batch_id}`: it answers `202` while running, then `200` with
the complete response. **Restart the server without that variable afterwards.**
(There is also an automated test for this — `test_async_202_then_get_returns_full_result`.)

---

## 7. The 100-question evaluation

This is the spec's section-1 scenario, scripted. The server must be running.

```bash
python scripts/build_testset.py                    # once; writes data/eval/testset_100.csv
python scripts/run_eval.py --run-id yourname-1
python scripts/score_eval.py --run-id yourname-1
```

`run_eval.py` sends 10 batches of 10 questions **sequentially** (each only after
the previous response arrives), retries a retryable failure 3 times at 5s/15s/45s
reusing the same `batch_id`, honours `Retry-After`, follows the 202 path, and
skips batches already saved — so if you interrupt it, just run it again.
Responses land in `data/runs/yourname-1/b01.json … b10.json`.

With the offline generator this takes about 25 seconds. With a real LLM provider
expect 3–10 minutes depending on the provider.

### Baseline to compare against

Our run, 100 questions, offline generator:

| Metric | Our result | Target | |
|---|---|---|---|
| Hit@5 (document) | 0.978 | ≥ 0.90 | PASS |
| Context precision | 0.960 | ≥ 0.60 | PASS |
| Faithfulness | 1.000 | ≥ 0.95 | PASS |
| Citation accuracy | 0.966 | ≥ 0.90 | PASS |
| Fabricated sources | 0 | 0 | PASS |
| Stale norms | 0 | 0 | PASS |
| Latency p50 / p95 | 0.23s / 0.27s | ≤ 5s / ≤ 12s | PASS |
| Error share | 0.000 | ≤ 0.02 | PASS |
| No-answer accuracy | 0.000 | ≥ 0.90 | **FAIL** (expected) |
| Hit@5 (article), MRR | not measurable | | needs gold labels |
| Answer correctness | not measurable | | needs gold labels |

Your retrieval numbers should match these closely — retrieval does not depend on
the provider. If Hit@5 (document) comes out far below 0.90, something is wrong
with your index; report it.

Latency will be higher once a provider is wired in (expect p50 in the 1–3s
range), because the numbers above exclude generation.

If you score a partial run (`--batches b01,b02`), the report says
`PARTIAL RUN` at the top and every metric is over the answered subset only.

### What you need to fill in

`data/eval/testset_100.csv` has two columns left **deliberately blank**:

- `expected_article` — the article (modda) that should answer the question
- `expected_answer` — the correct answer

They are blank because filling them requires reading the statutes, and filling
them from the system's own output would make the evaluation grade the system
against itself. The spec puts the test set on the tester's side for this reason.

Fill them in and `score_eval.py` computes **Hit@5 (article)**, **MRR** and
**answer correctness** automatically — no code changes. Answer correctness is
graded 0/1/2 by hand or with an LLM judge against `expected_answer`.

Article labels are matched tolerantly, so `480¹-modda`, `480.1` and `480-1` all
work.

---

## 8. Generation (needs a provider key)

Everything above works without an LLM. Answer *writing* does not.

### Providers

| `RAG_LLM_BACKEND` | Key variable | `temperature` | `seed` | Notes |
|---|---|---|---|---|
| `gemini` | `GEMINI_API_KEY` | yes | no | Native JSON schema output; best Uzbek quality of these |
| `groq` | `GROQ_API_KEY` | yes | **yes** | The only one that honours the spec's `seed` literally |
| `anthropic` | `ANTHROPIC_API_KEY` | not accepted | not accepted | Needs a Console key (`sk-ant-api03-…`) |
| `offline` | none | n/a | n/a | Default. Deterministic, no network, no real answers |

Once you have a key, find out which model ids it can actually reach — model
line-ups change, so do not trust the default — and test one real question:

```bash
# in .env:  GEMINI_API_KEY=...   and   RAG_LLM_BACKEND=gemini
python scripts/list_models.py gemini
python scripts/list_models.py gemini --probe
```

`--probe` retrieves contexts for one question, calls the provider, and reports
whether the returned quote is verbatim. That is the fastest way to tell a working
key from a broken one.

Then set `RAG_GEMINI_MODEL=<id>` in `.env`, restart the server, and re-run:

```bash
python scripts/verify.py --quick          # "Generation" should now be PASS
python scripts/run_eval.py --run-id gemini-1
python scripts/score_eval.py --run-id gemini-1
```

With a working provider, these should now also pass:

- **No-answer accuracy ≥ 9/10.** The test set has 10 questions the corpus cannot
  answer ("how do I register a land parcel on Mars"). A real model should return
  `no_answer: true` for them instead of a plausible-looking answer.
- **`answer_language`.** Ask in Russian, get Russian back. The offline generator
  cannot do this — it quotes the Uzbek source verbatim.

The provider and model are recorded in every response's `system` block
(`llm_provider`, `llm_model`), so runs on different providers stay
distinguishable. `score_eval.py` warns if one run's batches disagree about
`system` — those results cannot be compared as a single run.

---

## Known failures (do not report these as bugs)

1. **`no_answer_accuracy` is 0.000 and `Generation` is SKIP.** There is no
   working provider key in the repo. The `ANTHROPIC_API_KEY` that was in the
   developer's `.env` is a Claude Code subscription OAuth token
   (`sk-ant-oat01-…`): it authenticates but the Messages API rate-limits it, so
   every model returns `429`. `.env.example` therefore ships
   `RAG_LLM_BACKEND=offline`. Add a key and this resolves with no code change.
2. **With `RAG_LLM_BACKEND=anthropic` and that OAuth key, every question comes
   back `status: "error"`, `LLM_ERROR`, `retryable: true`.** That is the
   specified behaviour for a generation failure, not a crash — the batch still
   returns `200` and `contexts` are still populated, so retrieval errors stay
   distinguishable from generation errors. Worth verifying it behaves this way;
   not worth reporting as broken.
3. **`Hit@5 (article)`, `MRR` and `answer correctness` print
   `not measurable`.** Expected until the two gold columns are filled —
   see [What you need to fill in](#what-you-need-to-fill-in).
4. **`doc_status` is always `amalda`** for every citation. All 25 indexed
   documents are in force, so the "stale norm" metric *cannot* fail. A `0` there
   is not evidence that the check works. Testing it properly needs a repealed
   document in the index.
5. **`edition_date` is the same for every article of a document** (`2026-07-01`).
   The corpus is a dated snapshot of consolidated in-force text with no revision
   history, so the snapshot month is the only defensible edition marker.
6. **`as_of_date` does not change the answer.** Only one snapshot is indexed. An
   earlier `as_of_date` adds a warning to `debug` rather than silently answering
   from the wrong edition.
7. **`/v1/meta` reports 60 excluded points.** The Qdrant collection also holds
   chunks from an older raw-scrape pipeline — whole lex.uz documents split by
   character count, with no article structure. A citation from one could not fill
   the mandatory `article` and `part` fields, so `/v1` serves the article-level
   corpus only and reports the gap instead of hiding it.
8. **One test is skipped** (`test_out_of_scope_question_is_refused`). It needs a
   real provider; it un-skips automatically when `RAG_LLM_BACKEND` is not
   `offline`.

---

## Reporting a problem

Please include:

1. The output of `python scripts/verify.py --quick` (whole thing).
2. Your OS and `python --version`.
3. For an API issue: the **exact request** (full JSON body and headers, with the
   key redacted), the **HTTP status**, and the **full response body**.
4. For a quality issue: the question, what you expected, and the `contexts` array
   from the response. The contexts are what separate "retrieval did not find the
   right article" from "the right article was found but the answer is wrong" —
   without them a quality report cannot be acted on.
5. For an eval issue: the `data/runs/<run-id>/` directory and the
   `score_eval.py` output.

A single useful command for most reports:

```bash
python scripts/verify.py --base-url http://localhost:8000 > verify-report.txt 2>&1
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `verify.py`: Qdrant `cannot reach localhost:6333` | Qdrant not running | `docker compose up -d` (on macOS/Windows, launch Docker Desktop first) |
| `verify.py`: `collection 'lex_uz' does not exist` | Index never built | `python ingest/ingest_corpus.py` |
| `verify.py`: `collection 'lex_uz': 0 points` | Ingest interrupted | run `python ingest/ingest_corpus.py` again; it resumes |
| `verify.py`: missing packages | venv not activated, or deps not installed | activate the venv, then `pip install -r requirements.txt` |
| Server takes 30s to start | Loading 7383 articles + warming the embedding model | normal; wait for `embedding model ready` |
| First question slow, rest fast | Embedding model loaded lazily | normal on older builds; current code warms it at startup |
| Uzbek letters show as `?` in terminal | Console encoding, not the API | Windows: set `OutputEncoding` and `PYTHONUTF8` as in [Setup](#windows-powershell) |
| `curl: option -d` error on Windows | PowerShell's `curl` is `Invoke-WebRequest` | use the PowerShell examples, or `curl.exe` |
| Every request returns `429` | Rate limit is 1 batch/second per key | wait a second between requests; `run_eval.py` already paces itself |
| `409 BATCH_ID_CONFLICT` while testing | Reused a `batch_id` with different questions | use a new `batch_id` |
| Response identical and instant on retry | Idempotent replay — the specified behaviour | use a new `batch_id` to force recomputation |
| pip pulls gigabytes of `nvidia-*` packages | Default torch build includes CUDA | install the CPU torch wheel first, as shown in [Setup](#linux) |
| `pytest` not found | Not installed in the venv | `pip install -r requirements.txt` |

---

## Quick reference

```bash
python scripts/verify.py --quick                      # is it ready?
python scripts/verify.py --base-url http://localhost:8000   # + live server + full eval
pytest tests/ -q                                      # the spec checklist (46 tests)
uvicorn backend.main:app --port 8000                  # run the server
python ingest/ingest_corpus.py                        # build/repair the index
python scripts/build_testset.py                       # regenerate the test set
python scripts/run_eval.py --run-id NAME              # 10 batches x 10 questions
python scripts/score_eval.py --run-id NAME            # the spec's metric table
python scripts/list_models.py gemini --probe          # check a provider key
```

The spec itself is `RAG API — test spetsifikatsiyasi (lex.uz).pdf` in this repo.
Where this guide and the spec disagree, the spec wins — tell us.
