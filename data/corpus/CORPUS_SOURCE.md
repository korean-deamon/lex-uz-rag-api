# Corpus provenance

The legal corpus under `uzbek-legal-corpus/` is third-party data, not ours.

| | |
|---|---|
| Upstream | https://github.com/javohirmat/uzbek-legal-corpus |
| Commit | `7fabb822cca04439ac6197bfa1441eca6112bc19` (2026-10-09) |
| Author | Javohir Matniyazov |
| Licence | CC0 for the data (`LICENSE`), see `LICENSE-CODE` for the repo's code |
| Snapshot | `2026-07` (the `snapshot_date` column in `manifest.csv`) |

## What is committed here, and what is not

Committed — the two things the `/v1` API reads:

- `uzbek-legal-corpus/data/articles/*.jsonl` — 7383 articles across 25 documents,
  one JSON object per article
- `uzbek-legal-corpus/manifest.csv` — per-document metadata (lex.uz document
  number, adoption date, article count, snapshot date), which is where
  `citations[].doc_id`, `doc_title`, `doc_type`, `url` and `edition_date` come
  from
- `uzbek-legal-corpus/LICENSE`, `LICENSE-CODE`, `README.md`, `README_HF.md`

Not committed (see the project `.gitignore` for the exact patterns):

- the upstream repo's own `.git` directory — it was removed so these files are
  plain files rather than a nested repository. Git treats a nested repo as a
  submodule and would commit a pointer instead of the data, which is the
  opposite of what is wanted here
- `data/raw/` — 8.8 MB of source `.txt` files that nothing in this project reads
- `rag/`, `scripts/` — the upstream project's own demo RAG implementation. This
  project does not use or run it
- `.claude/`, `.agents/`, `skills-lock.json` — third-party agent and skill
  definitions, i.e. executable instructions we did not review. They have no
  bearing on the corpus data

## Why the data is pinned rather than fetched

Every response's `system.index_version` claims the index was built from snapshot
`2026-07`, and eval runs are only comparable when they ran against the same
articles. If each person cloned the corpus fresh, an upstream update would
silently change the index and make two runs' scores incomparable while both
still claimed the same `index_version`.

So: do not replace these files with a newer clone as part of normal testing. To
deliberately move to a newer snapshot, update the files, bump `INDEX_VERSION`
and `LEX_SNAPSHOT_DATE` in `.env`, re-run `python ingest/ingest_corpus.py`, and
record the new commit above — then old and new runs stay distinguishable.
