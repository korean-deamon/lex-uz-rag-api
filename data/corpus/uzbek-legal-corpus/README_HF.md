---
license: cc0-1.0
language:
  - uz
pretty_name: Uzbek Legal Corpus
tags:
  - legal
  - law
  - uzbekistan
  - government
  - rag
task_categories:
  - text-retrieval
  - question-answering
  - text-generation
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files: data/articles/*.jsonl
---

# Uzbek Legal Corpus (Oʻzbek huquqiy korpusi)

25 cleaned, audited, machine-readable texts of the Republic of Uzbekistan's Constitution, 20 codes, and 4 major laws — in Uzbek (Latin script), sourced from the official National Database of Legislation ([lex.uz](https://lex.uz)). Snapshot: **July 2026**. 7,368 articles.

Built for NLP / LLM / RAG work on Uzbek legal text. Released by [Tomaris AI](https://tomaris.ai).

## Contents

| Path | What it is |
|---|---|
| `data/raw/*.txt` | Cleaned plain text, one file per code/law |
| `data/articles/*.jsonl` | One JSON record per article (canonical structured form) |
| `manifest.csv` | Per-file: official title, lex.uz doc id, adoption date, article count, last article, SHA-256, snapshot date |
| `scripts/clean.py` | The normalization pipeline (lex.uz text → `data/raw`) |
| `scripts/parse_articles.py` | `data/raw` → `data/articles` + `manifest.csv` |

## JSONL schema

```json
{"seq": 496, "code": "soliq_kodeksi",
 "code_title": "Oʻzbekiston Respublikasining Soliq kodeksi",
 "article_id": "480.1", "article_display": "480¹-modda", "article_raw": "4801-modda",
 "kind": "insert", "title": "Umumiy qoidalar",
 "section": "XXI BOʻLIM. ...", "chapter": "71¹-bob. ...",
 "text": "...", "lex_uz_doc": "-4674902", "snapshot_date": "2026-07"}
```

Special records: `article_id: "_front"` (preamble/front matter, e.g. the Constitution's Muqaddima) and `"_end"` (trailing matter, e.g. the Criminal Code's legal-terms glossary, SAKKIZINCHI BOʻLIM).

## Conventions you must know

**1. Encoding standard (do not "fix" it).** The okina in *oʻ/gʻ* is **U+02BB (ʻ)** and the tutuq belgisi in *maʼlumot* is **U+02BC (ʼ)** — uniformly, corpus-wide. Never U+2018/U+2019 or ASCII `'`. Mixed encodings silently split an embedding space; this corpus has exactly one.

**2. Superscript articles are digit-flattened in the source.** lex.uz renders inserted article 480¹ as `4801`. Raw text keeps that form; the JSONL resolves it **by document position** into `article_id: "480.1"`. The raw number is ambiguous ("261" is 26¹ in one place and article 261 in another) — always use `article_id`, or reparse positionally. Chapter strings are kept raw (`711-bob` = 71¹-bob).

**3. Gaps are repealed articles, not missing data.** Current lex.uz editions drop repealed article headings entirely. Every gap below was verified against lex.uz / consolidated editions during the audit (clean seams, zero dangling internal cross-references):

- Konstitutsiya (155), Mehnat (581), Suv (165), Saylov (103), Shaharsozlik (85), Raqobat (49), Investitsiyalar (69), Qimmatli qogʻozlar (64): **dense, no gaps**
- Soliq (→483): 324, 383, 392, 459
- Fuqarolik I (→385): 63, 65–66, 70–72, 176–177, 179 · Fuqarolik II (386→1199): 1052–1055, 1057, 1064, 1071–1072, 1080–1081
- Fuqarolik protsessual (→462): 260, 373–382, 392, 408–409, 412, 420–436 (2021 appeal reform; replaced by 372¹⁻⁴, 419¹⁻²⁹)
- Jinoyat (→302): 53, 174, 187, 224, 272 (+ repealed bases 48, 84 whose inserts survive)
- Jinoyat-protsessual (→623): 5–10 (repealed 2010, OʻRQ-262), 216, 229, 231–233, 246, 339–343, 399–400, 404, 419, 505–509 (reform inserts 504¹⁻³, 509⁵⁻¹⁴ etc.)
- Jinoyat-ijroiya (→197): 33–43, 51, 70, 83, 134–135, 148, 162 (+ base 44)
- Maʼmuriy javobgarlik (→348): 6, 73, 122, 141, 156, 162, 222, 246, 253, 320 (+ bases 184, 266)
- Maʼmuriy sud ishlari (→288): 190–193, 246 · Iqtisodiy protsessual (→347): 304
- Budjet (→192): 37–43, 53–59, 74–80 · Bojxona (→412): 133, 236–238, 278–280, 283, 285
- Oila (→238): 163, 168, 176–193 (guardianship chapters superseded by the 2014 vasiylik law)
- Yer (→91 + 91¹): 19, 34, 47, 50, 54, 57 · Havo (→135): 96–97 · Tadbirkorlik (→52): none (has 18¹, 30¹)

**4. What was removed vs. kept.** Removed: lex.uz amendment annotations (parenthesized gazette references), navigation/UI cruft, BOMs, control characters. Kept: all statute text, including inline legal parentheticals, repeal-list articles inside laws (which legitimately cite gazettes), the Criminal Code glossary, and transitional articles.

**5. Heading tolerance for your own parsers.** Match `\d+\s*-\s*modda`, not the strict form — the source occasionally emits `599- modda`.

## Provenance & currency

Texts reflect the in-force consolidated editions on lex.uz as of the snapshot date, including the new Mehnat kodeksi (2022), Suv kodeksi (2025, in force 31.10.2025), Raqobat qonuni (2023), and the 2023 Constitution. The Havo kodeksi is the 1993 code, which remains in force (an ICAO-aligned new edition is in drafting as of the snapshot). **This is an unofficial copy for research and machine learning. Laws change; lex.uz is the sole authoritative source.** Blank `lex_uz_doc` cells in the manifest simply mean the id wasn't captured during the audit — search the title on lex.uz.

## Legal status & license

Under Uzbek law, official documents — laws, decisions, resolutions and the like, and their official translations — **are not objects of copyright** (Law "On Copyright and Related Rights", OʻRQ-42 of 20.07.2006, art. 8). The statute texts are therefore public domain.

- **Data** (`data/`, `manifest.csv`): released under **CC0 1.0** (`LICENSE`) — no rights asserted over the compilation/cleaning either.
- **Code** (`scripts/`): **MIT** (`LICENSE-CODE`).

Attribution to lex.uz and a link back to this repository are appreciated but not required.

## Cite

```
Uzbek Legal Corpus (2026-07 snapshot). Tomaris AI.
Source texts: Qonunchilik maʼlumotlari milliy bazasi (lex.uz).
```

## Load

```python
from datasets import load_dataset
ds = load_dataset("javohirmat/uzbek-legal-corpus", split="train")   # configs make this arg-free
```

## Stats

| Code | Articles | Words |
|---|---:|---:|
| bojxona_kodeksi | 419 | 59,190 |
| budjet_kodeksi | 172 | 24,903 |
| fuqarolik_kodeksi_1qism | 386 | 34,255 |
| fuqarolik_kodeksi_2qism | 811 | 76,905 |
| fuqarolik_protsessual_kodeksi | 508 | 53,736 |
| havo_kodeksi | 136 | 11,367 |
| investitsiyalar_faoliyati_qonuni | 69 | 8,643 |
| iqtisodiy_protsessual_kodeksi | 385 | 49,133 |
| jinoyat-ijroiya_kodeksi | 196 | 18,384 |
| jinoyat_kodeksi | 392 | 53,498 |
| jinoyat_protsessual_kodeksi | 762 | 101,971 |
| konstitutsiya | 155 | 10,563 |
| mamuriy_javobgarlik_kodeksi | 660 | 76,664 |
| mamuriy_sud_ishlarni_yuritish_kodeksi | 306 | 34,286 |
| markaziy_bank_qonuni | 76 | 8,956 |
| mehnat_kodeksi | 581 | 80,270 |
| oila_kodeksi | 220 | 17,486 |
| qimmatli_qogozlar_bozori_qonuni | 65 | 10,262 |
| raqobat_qonuni | 49 | 11,248 |
| saylov_kodeksi | 110 | 14,857 |
| shaharsozlik_kodeksi | 85 | 14,059 |
| soliq_kodeksi | 496 | 149,577 |
| suv_kodeksi | 165 | 26,338 |
| tadbirkorlik_faoliyati_qonuni | 54 | 6,455 |
| yer_kodeksi | 110 | 18,300 |
| **Total** | **7368** | **971,306** |

Word counts cover article titles + bodies (front/end matter excluded). Expect roughly 2.5–3.5 tokens per Uzbek word depending on tokenizer.

## Updates

Snapshot releases only (this is `2026-07`); no continuous sync with lex.uz. Laws change — lex.uz is the sole authoritative source.

## Citation

```bibtex
@misc{uzbek_legal_corpus_2026,
  title  = {Uzbek Legal Corpus (2026-07 snapshot)},
  author = {{Tomaris AI}},
  year   = {2026},
  note   = {Source texts: Qonunchilik ma'lumotlari milliy bazasi (lex.uz)},
  url    = {https://huggingface.co/datasets/javohirmat/uzbek-legal-corpus}
}
```
