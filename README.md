# lex.uz RAG API

Batch RAG API over Uzbekistan legislation, built to the test specification in
`RAG API — test spetsifikatsiyasi (lex.uz).pdf`. It answers up to 10 questions
per request and returns, for every question, an answer, the lex.uz sources
(document + article) and the retrieved context chunks — so retrieval quality and
generation quality can be scored separately.

Corpus: the Constitution, 20 codes and 4 laws of the Republic of Uzbekistan,
**7383 articles**, one chunk per article (*modda*), from the Uzbek Legal Corpus
snapshot `2026-07` (CC0, sourced from lex.uz).

Three guides, by audience:

| File | For | Language |
|---|---|---|
| `README.md` | whoever works on the code: architecture, retrieval, results | English |
| `TESTING.md` | whoever runs it: per-platform setup, Linux / macOS / Windows | English |
| `OQITUVCHI_UCHUN.md` | the reviewer testing the API against the spec | Uzbek |

---

## What was built

The project previously had one endpoint, `POST /api/chat`, taking a single
question and returning `{answer, sources}`. None of the `/v1` endpoints the
specification requires existed. Added:

- **The five `/v1` endpoints**, with `Bearer` auth, 1–10 question batches, the
  full section-4 response shape (`system`, `timing`, `summary`, per-question
  `citations` / `contexts` / `debug` / `usage` / `error`), and every status code
  from section 5 (`400` / `401` / `403` / `409` / `429` + `Retry-After` /
  `202` async).
- **A batch store** (SQLite) giving idempotent replay of a `batch_id`, `409` when
  the same id arrives with different questions, async batch state, and 30-day
  retention.
- **Hybrid retrieval with real reranking** — dense (e5) + BM25, fused on an
  absolute score scale, with Cyrillic→Latin transliteration so Russian and
  Cyrillic questions work against a Latin-script corpus. Measured 0.978 Hit@5
  (document) across 100 questions.
- **Citation verification.** Every quote the model returns is re-checked against
  the context it claims to come from; a quote that is not verbatim is dropped,
  and an answer left with no surviving citation is downgraded to `no_answer`
  rather than published ungrounded. Fabricated sources are structurally
  impossible, not merely discouraged.
- **A pluggable generator**: Anthropic, Gemini, Groq, or an offline
  deterministic answerer. Same prompt, same schema, same verification for all
  four, so swapping providers cannot quietly change what is being measured.
- **The evaluation harness**: a 100-question test set in the spec's CSV format,
  a runner that executes the section-1 scenario literally (10 sequential
  batches, retries, async polling, resume), and a scorer that computes the
  section-6 metric table against the spec's targets.
- **The section-7 checklist as 46 executable tests** (`pytest tests/`), so
  "is the API ready for the test?" is answered by running them.

Two defects were found and fixed along the way, both caught by the harness
rather than by reading the code: the collection turned out to hold 60 chunks
from an older raw-scrape pipeline with no article structure, which produced
degenerate citations; and rank-based fusion (RRF) was burying the correct
article, because e5 cosines on this corpus carry almost no ranking signal.

---

## Endpoints

| Method | Path | Purpose | Required by spec |
|---|---|---|---|
| POST | `/v1/rag/batch` | 1–10 questions, answers + sources + contexts | yes |
| GET | `/v1/rag/batch/{batch_id}` | re-read a stored result, or poll an async batch | yes |
| POST | `/v1/rag/query` | one question, for debugging | no |
| GET | `/v1/meta` | model, embedding, index version, snapshot date, limits, caveats | yes |
| GET | `/v1/health` | liveness (unauthenticated) | yes |

Auth: `Authorization: Bearer <API_KEY>`. The test key lives in `RAG_API_KEYS`
and defaults to `test-key-lex-2026`. Responses are
`application/json; charset=utf-8`; `oʻ`, `gʻ` and `ʼ` pass through unchanged.

---

## Setup

### One command

```bash
docker compose up
```

That is the whole setup, on every platform. Qdrant starts, the api container
waits for it to be healthy, builds the index if the collection is missing, and
then serves on `http://localhost:8000` with the test key
`test-key-lex-2026`. No `.env` is required: the compose defaults answer with
the offline generator, which needs no credential. Copy `.env.example` to `.env`
only to point generation at a provider of your own.

Two of those steps are slow, and both happen once:

| Step | Cost | Kept in |
|---|---|---|
| Build the image | ~5 min | Docker image cache |
| Download `multilingual-e5-base` | ~1.1 GB | `hf_models` volume |
| Embed 7383 articles | ~85 min on CPU | `qdrant_storage` volume |

So the first `docker compose up` takes about an hour and a half and every later
one takes seconds. The index build streams progress every 64 articles and
resumes rather than restarting if it is interrupted — the progress file lives on
the `app_state` volume for exactly that reason. `docker compose down` keeps all
three volumes; `down -v` is what discards them.

Scripts run inside the container:

```bash
docker compose exec api python scripts/verify.py --quick
docker compose exec api python -m pytest tests/ -q
docker compose exec api python scripts/run_eval.py --run-id r1
```

`data/eval/` and `data/runs/` are bind-mounted, so the test set you edit on the
host is the one the container reads, and run output lands on the host.

### Without Docker

Useful if you want to work on the code rather than just run it. You need
**Python 3.11+** and **Docker** (for Qdrant alone). The instructions below
differ only in shell syntax — the code is identical on all three platforms.

#### Linux

```bash
git clone <repo> && cd "chatbot vibe"
cp .env.example .env                       # then edit .env, see "Providers" below
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d                       # Qdrant on localhost:6333
python ingest/ingest_corpus.py             # only if the collection is empty
uvicorn backend.main:app --port 8000
```

On a CPU-only machine, install torch first to avoid pulling several GB of CUDA
packages you will not use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### macOS

Same as Linux. Install the prerequisites with Homebrew and use Docker Desktop:

```bash
brew install python@3.12
# Docker Desktop: https://www.docker.com/products/docker-desktop  (then launch it)

cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python ingest/ingest_corpus.py
uvicorn backend.main:app --port 8000
```

On Apple Silicon the embedding model runs on the CPU via PyTorch's default
build — no extra setup. The first run downloads `multilingual-e5-base`
(~1.1 GB) into `~/.cache/huggingface`, which takes a few minutes once.

#### Windows

Use PowerShell. Install Python from python.org (tick **Add python.exe to PATH**)
and Docker Desktop with the WSL2 backend.

```powershell
cd "chatbot vibe"
Copy-Item .env.example .env                # then edit .env
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker compose up -d
python ingest\ingest_corpus.py
uvicorn backend.main:app --port 8000
```

If activating the venv is blocked, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Two Windows-specific notes:

- **Use PowerShell, not `cmd.exe`, for the `curl` examples below.** `curl` in
  PowerShell is an alias for `Invoke-WebRequest` and does not take `-d`; either
  call `curl.exe` explicitly, or use the PowerShell form shown after each
  example.
- **Set the console to UTF-8 before running anything that prints Uzbek**, or
  `oʻ` and `gʻ` come out as `?` in the terminal (the API responses themselves
  are fine — this is the console's encoding, not the data's):

  ```powershell
  [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
  $env:PYTHONUTF8 = "1"
  ```

Everything else — the server, the tests, the eval scripts — behaves identically.
The scripts use `pathlib` throughout, so paths work unchanged.

### Is it ready? One command

```bash
python scripts/verify.py --quick                          # no server needed
python scripts/verify.py --base-url http://localhost:8000 # also live server + 100-question eval
```

It checks, in dependency order: the environment, Qdrant, the article index,
retrieval against five hand-verified questions in all three languages, the
section-7 checklist (the test suite), and the configured provider — then prints
one verdict. A `FAIL` names the fix. `READY EXCEPT GENERATION` means everything
but the model credential is in place.

Last full run: **6 passed, 0 failed, 0 skipped** — the warning is
`no_answer_accuracy` and the skip is generation, both from limitation 1 below.

### Check it came up

```bash
curl -s localhost:8000/v1/health
```
```powershell
Invoke-RestMethod localhost:8000/v1/health | ConvertTo-Json -Depth 5
```

`status: "ok"` means Qdrant is reachable **and** the article index loaded.
`degraded` (HTTP 503) means it would answer badly, so the check fails loudly
rather than passing a service that cannot retrieve.

### One question, end to end

```bash
curl -s -X POST localhost:8000/v1/rag/batch \
  -H 'Authorization: Bearer test-key-lex-2026' \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{
    "run_id": "demo", "batch_id": "demo-b01",
    "batch_index": 1, "total_batches": 1,
    "options": {"top_k": 5, "return_contexts": true, "return_debug": true,
                "temperature": 0, "seed": 42},
    "items": [{"id": "q001",
               "question": "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?"}]
  }' | python3 -m json.tool
```

```powershell
$body = @{
  run_id = "demo"; batch_id = "demo-b01"; batch_index = 1; total_batches = 1
  options = @{ top_k = 5; return_contexts = $true; return_debug = $true
               temperature = 0; seed = 42 }
  items = @(@{ id = "q001"
               question = "Yillik asosiy mehnat taʻtilining eng kam muddati qancha?" })
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri localhost:8000/v1/rag/batch -Method Post `
  -Headers @{ Authorization = "Bearer test-key-lex-2026" } `
  -ContentType "application/json; charset=utf-8" -Body $body |
  ConvertTo-Json -Depth 8
```

---

## Providers

Retrieval is the same whatever generates the answers; only the generation half
changes. Pick one in `.env`:

| `RAG_LLM_BACKEND` | Key needed | `temperature` | `seed` | Notes |
|---|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | not accepted | not accepted | Console key (`sk-ant-api03-…`) |
| `gemini` | `GEMINI_API_KEY` | yes | no | Native JSON schema output |
| `groq` | `GROQ_API_KEY` | yes | **yes** | The only one honouring the spec's `seed` literally |
| `offline` | none | n/a | n/a | Deterministic, no network; cannot judge `no_answer` |

Once you have a key, find out which model ids it can actually reach rather than
trusting a hardcoded default, and test one real question:

```bash
python scripts/list_models.py gemini
python scripts/list_models.py gemini --probe --model gemini-2.0-flash
```

Then set it:

```ini
RAG_LLM_BACKEND=gemini
RAG_GEMINI_MODEL=gemini-2.0-flash
GEMINI_API_KEY=...
```

The provider and model id are recorded in every response's `system` block
(`llm_provider`, `llm_model`), so two runs on different providers stay
distinguishable when their scores are compared. `GET /v1/meta` also states which
of the `temperature` / `seed` cases above applies to the running configuration,
rather than making one claim for all of them.

**Quality note for Uzbek legal text:** the task is grounded extraction with
verbatim quoting, not open generation, so a small fast model is enough — but it
must handle Uzbek well. Gemini Flash handles Uzbek noticeably better than the
Llama models on Groq, so if you have both keys, start with Gemini and use Groq
when you specifically want `seed` reproducibility. Whatever you pick, the
faithfulness check in `pipeline.py` drops any quote the model did not copy
exactly, so a weaker model loses citations rather than inventing them.

---

## Running the evaluation

The spec's scenario (section 1) is scripted end to end.

```bash
# 1. build the 100-question test set (10 batches x 10, mixed categories)
python scripts/build_testset.py          # -> data/eval/testset_100.csv

# 2. send b01..b10 sequentially, saving each response
python scripts/run_eval.py --run-id eval-2026-11-03
#    -> data/runs/eval-2026-11-03/b01.json ... b10.json

# 3. score it against the spec's metric table
python scripts/score_eval.py --run-id eval-2026-11-03
```

The runner sends each batch only after the previous response arrives, retries a
retryable failure 3 times at 5s / 15s / 45s with the **same** `batch_id` (so the
server's idempotency returns the stored result instead of recomputing), honours
`Retry-After` on 429, follows the 202 async path, and skips batches already on
disk so an interrupted run resumes.

To compare two system versions, bump `PIPELINE_VERSION` (or `INDEX_VERSION`) and
run again under a new `run_id`. The scorer warns if the batches of one run
report different `system` blocks, because those results cannot be compared as
one run.

---

## How the spec checklist (section 7) is met

Every line is covered by a test in `tests/test_api_v1.py`. Run them with
`pytest tests/ -q` — 86 pass, 1 is skipped unless a working provider key is set.

| Checklist line | How | Test |
|---|---|---|
| accepts 1–10 questions, 400 on 11 | validated before any embedding or model call | `test_accepts_ten_questions`, `test_eleven_questions_rejected_with_400` |
| each `id` once, count equals question count | `id`-keyed, order preserved | `test_results_match_items_by_id_exactly` |
| citations: `doc_id`, `url`, `article`, `edition_date`, `doc_status`, `quote` | joined from the corpus manifest | `test_citations_carry_every_required_field` |
| contexts with top-k, text and scores | real cosine + fused rerank score | `test_contexts_shape_and_scores` |
| every `citations[].chunk_id` is in `contexts` | `chunk_id` is taken from the context, never from the model | `test_no_citation_outside_contexts` |
| unanswerable → `no_answer: true`, nothing invented | quote verification; ungrounded answers are downgraded | `test_no_answer_mechanism_reports_no_citations` |
| one question's error does not affect the others | per-question try/except → `status: error` + `error` object | `test_question_level_error_is_isolated` |
| `system`: model, embedding, index version, snapshot date | `pipeline.system_block()` | `test_system_block_is_complete` |
| `temperature=0` + `seed` → same sources | retrieval is deterministic; answers cached by request identity | `test_same_question_returns_identical_sources` |
| same `batch_id` resent → stored result | SQLite reservation + content hash | `test_same_batch_id_replays_stored_result` |
| Latin, Cyrillic and Russian; `oʻ`/`gʻ` intact | script detection + Cyrillic→Latin transliteration | `test_three_languages_are_detected_and_answered` |
| batch ≤ 120s, or async 202 + GET | sync wait with deadline, worker keeps going | `test_async_202_then_get_returns_full_result` |
| `/v1/meta` and `/v1/health` work | — | `test_meta_reports_system_index_and_documents` |
| separate test API key | `RAG_API_KEYS`, 401/403 | `test_unknown_api_key_is_403` |

Error codes behave as section 5 specifies: `400` validation, `401`/`403` auth,
`409` same `batch_id` with different content, `429` + `Retry-After`, and
per-question `RETRIEVAL_FAILED` / `LLM_TIMEOUT` / `LLM_ERROR` /
`CONTEXT_TOO_LONG` / `INVALID_LANGUAGE`.

---

## How retrieval works

Three stages, in `backend/retrieval.py`:

1. **Candidates** — dense search (`multilingual-e5-base`) over all articles,
   BM25 over the same articles, and, when the question names a specific code, a
   third dense search scoped to that code.
2. **Rerank** — the two retrievers are fused on an **absolute score** scale, not
   by rank. Rank fusion (RRF) was tried first and is wrong for this corpus: e5
   cosines sit in a very narrow band (a question's candidates typically span
   0.79–0.81), so dense *rank* is close to a coin flip, while BM25 margins are
   large and meaningful — an exact article-title match scores 12.5 where the next
   article scores 7.9. RRF discards exactly the signal that matters and buried
   the right article; score fusion puts it back at rank 1.
3. **Report** — every context carries its real cosine as `retrieval_score` and
   its fused score as `rerank_score`, both on a 0–1 absolute scale, so threshold
   analysis across questions is meaningful.

Two things that measurably mattered:

- **Cyrillic → Latin transliteration.** The corpus is Latin Uzbek, so a Cyrillic
  question (`ўғрилик`) shared no lexical token with the article answering it
  (`oʻgʻrilik`) and fell back to dense search alone, which was not enough.
- **The code bonus is flat.** It started as an extra dense contribution for
  code-scoped hits, which silently doubled dense's weight whenever a code was
  named and let a generic article ("Jinoyatning tushunchasi", cosine 0.845)
  outscore the article whose title *is* the question's decisive word. Flat means
  it cancels between candidates from the same code and leaves the retrievers to
  decide.

---

## Measured results

100 questions through the real HTTP endpoint (`run_id=offline-clean-1`, offline
generator — see limitation 1):

| Metric | Value | Target | |
|---|---|---|---|
| Hit@5 (document) | 0.978 | ≥ 0.90 | PASS |
| Context precision | 0.960 | ≥ 0.60 | PASS |
| Faithfulness | 1.000 | ≥ 0.95 | PASS |
| Citation accuracy | 0.966 | ≥ 0.90 | PASS |
| Fabricated sources | 0 | 0 | PASS |
| Stale norms | 0 | 0 | PASS |
| Latency p50 / p95 | 0.23s / 0.27s | ≤ 5s / ≤ 12s | PASS |
| Error share | 0.000 | ≤ 0.02 | PASS |
| No-answer accuracy | 0.000 | ≥ 0.90 | **FAIL** — needs a model, see limitation 2 |
| Hit@5 (article), MRR | not measurable | | needs `expected_article` |
| Answer correctness | not measurable | | needs `expected_answer` |

Latency here excludes generation. Expect p50 in the 1–3s range once a provider
is wired in; the targets leave ample headroom.

Retrieval across a spot-check of 8 questions in all three languages: 7/8 at
rank 1, 8/8 within top-5.

---

## Known limitations

These are stated here and in `GET /v1/meta`'s `caveats`, so a constant is not
mistaken for a working check.

1. **Generation is not currently running.** The `ANTHROPIC_API_KEY` in `.env` is
   a Claude Code subscription OAuth token (`sk-ant-oat01-…`). It authenticates,
   but the Messages API rate-limits it: every model returns `429`. The pipeline
   degrades exactly as the spec requires — HTTP 200, per-question
   `status: "error"`, `LLM_ERROR`, `retryable: true`, contexts still returned —
   but no answers are generated. `.env` therefore ships with
   `RAG_LLM_BACKEND=offline` so the service runs today. Put a working key in and
   switch the backend; no code changes.
2. **No-answer accuracy needs a model.** The offline generator decides
   `no_answer` by score threshold alone, which cannot catch a well-formed
   question the law does not address ("how do I register a land parcel on
   Mars") — retrieval legitimately finds land-code articles. Judging that
   retrieved text does not answer a question is a reading task. The `≥ 9/10`
   target is reachable only on a real provider;
   `test_out_of_scope_question_is_refused` covers it and is skipped until then.
3. **`temperature` and `seed` support varies by provider** — see the Providers
   table. Both are always echoed and always part of the answer-cache key, so an
   identical request returns an identical answer on every backend; retrieval,
   the part the spec actually tests for reproducibility, is deterministic by
   construction.
4. **`edition_date` is per document, not per article.** The corpus is a dated
   snapshot of consolidated in-force text with no revision history, so the only
   defensible edition marker is the snapshot month. Per-article amendment dates
   need a lex.uz revision scrape.
5. **`doc_status` is always `amalda`.** All 25 indexed documents are in force, so
   the "stale norm" metric cannot currently fail — a `0` there is not evidence
   that the check works. Testing it properly requires indexing a repealed
   document.
6. **`as_of_date` is recorded, not served.** Only the one snapshot is indexed; an
   earlier `as_of_date` adds a warning to `debug` instead of silently answering
   from the wrong edition.
7. **`Hit@5 (modda)`, `MRR` and answer correctness are unmeasured.**
   `expected_article` and `expected_answer` are deliberately blank in the
   generated test set: naming the article a question "should" land on requires
   reading the statute, and filling it from the pipeline's own output would make
   the eval grade the system against itself. The spec puts the test set on the
   tester's side — fill those two columns and the scorer computes all three.
8. **60 raw-scrape chunks are excluded.** The collection also holds whole lex.uz
   documents split by character count (from `ingest/ingest.py`). They carry no
   article structure, so a citation from one could not fill the mandatory
   `article` and `part` fields. `/v1` serves the article-level corpus only and
   reports the excluded count under `index` in `/v1/meta`.

---

## The spec's open questions (section 7)

- **When was the lex.uz data taken, and how often is it refreshed?**
  Snapshot `2026-07` (`manifest.csv`), reported as `lex_snapshot_date`. It is a
  one-off import; there is no refresh job for the corpus yet. The separate
  Airflow job in `orchestration/` refreshes scraped lex.uz documents daily into
  a **separate** Qdrant collection (`lex_uz_raw`), which `/v1` does not read, so
  it can run as often as it likes without touching the article-level index.
  Refreshing the corpus itself would mean re-fetching it and re-running
  `ingest/ingest_corpus.py`.
- **Is chunking by article or by fixed length?** By article. One chunk is one
  *modda*, which is why `citations[].article` is exact rather than inferred.
- **Are repealed documents kept in the index, and how are they marked?** None are
  indexed. The `doc_status` field exists and carries the `amalda` /
  `kuchini yoʻqotgan` vocabulary, but only the first value ever occurs
  (limitation 5).

---

## Layout

```
backend/
  api_v1.py       the /v1 endpoints: auth, rate limit, idempotency, sync/async
  schemas.py      request validation (the 400 cases) + the utf-8 JSON response
  pipeline.py     per-question orchestration, citation verification, /v1/meta
  retrieval.py    hybrid dense + BM25 retrieval and score fusion
  generate.py     grounded generation for all four providers + the offline one
  corpus_meta.py  document metadata for citations, from the corpus manifest
  lang.py         uz-Latn / uz-Cyrl / ru detection, apostrophe normalisation
  store.py        SQLite batch store: idempotency, 409, async state, retention
  config.py       one place for every version string, limit and provider setting
  main.py         app wiring, error envelopes
  embedding.py    the shared embedding model (indexing and querying must match)
scripts/
  build_testset.py  assemble the 100-question test set in the spec's CSV format
  run_eval.py       send the 10 batches sequentially, save each response
  score_eval.py     compute the section 6 metrics from saved runs
  list_models.py    what a provider key can reach, and a one-question probe
  verify.py         one-command readiness check: "is this ready for the test?"
tests/
  test_api_v1.py    the section 7 checklist, executable
ingest/
  ingest_corpus.py  article-level corpus -> Qdrant `lex_uz` (what /v1 serves)
  ingest.py         raw-scrape pipeline -> Qdrant `lex_uz_raw` (not served)
scraper/
  scraper.py        playwright scraper for lex.uz documents
orchestration/
  dags/             the daily scrape + refresh job (Airflow)
```

Removed along the way, and worth knowing if you saw an earlier version: a
browser chat UI (`frontend/`) and the single-question pipeline behind it
(`backend/rag.py`, `POST /api/chat`). It answered the same questions through
different retrieval code, so one question could come back two different ways
depending on which endpoint it entered by -- confusing to demonstrate and
misleading to evaluate. `/v1` is now the only way in, and `GET /` returns a
list of the endpoints.

The lex.uz scraper (`scraper/`) and its daily job (`orchestration/`) are kept
but unused by `/v1`, which serves the committed corpus instead (limitation 8).
Their dependencies live in `requirements-scraper.txt` so the API, the tests and
the evaluation do not drag in playwright, a 400 MB browser and Airflow.

Two things about that job were wrong and are now fixed. It wrote into the same
Qdrant collection `/v1` serves, with points carrying no `code` or `article_id` --
which is where the 60 malformed points and the earlier `lex:#art-` citations
came from; it now writes to `lex_uz_raw`. And the folder was called `airflow/`,
which shadowed the real package: `import airflow` resolved to the empty project
directory and reported no version, so the job could never have run. It is now
`orchestration/`.
