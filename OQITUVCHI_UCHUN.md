# O'qituvchi uchun — API'ni qanday test qilish

Bu hujjat bitta savolga javob beradi: **spetsifikatsiyada talab qilingan narsalarni
qanday tekshirish mumkin.**

Hujjat ikki xil tekshiruvni ko'rsatadi:

1. **Tez tekshiruv** (~5 daqiqa) — servis ishlayaptimi, kontrakt to'g'rimi
2. **To'liq baholash** (~30-60 daqiqa) — 100 savol, spetsifikatsiya §6 metrikalari

Agar vaqt kam bo'lsa, 1-2-3-bo'limlar yetarli.

Texnik tafsilotlar (arxitektura, algoritm, texnologiya tanlovi) — `README.md`.
Platformaga xos batafsil yo'riqnoma (Windows / macOS / Linux) — `TESTING.md`.

---

## Mundarija

1. [Nima qurilgan](#1-nima-qurilgan)
2. [Ishga tushirish](#2-ishga-tushirish)
3. [Tez tekshiruv — 5 daqiqa](#3-tez-tekshiruv--5-daqiqa)
4. [Birinchi so'rov — `curl`](#4-birinchi-sorov--curl)
5. [Javobda nima qaytadi](#5-javobda-nima-qaytadi)
6. [Status kodlarini tekshirish](#6-status-kodlarini-tekshirish)
7. [To'liq baholash — 100 savol](#7-toliq-baholash--100-savol)
8. [Metrikalar va maqsadlar](#8-metrikalar-va-maqsadlar)
9. [O'z LLM kalitingizni qo'yish](#9-oz-llm-kalitingizni-qoyish)
10. [Normal, lekin xato kabi ko'rinadigan narsalar](#10-normal-lekin-xato-kabi-korinadigan-narsalar)
11. [Nima o'lchangan, nima o'lchanmagan](#11-nima-olchangan-nima-olchanmagan)
12. [Muammo chiqsa](#12-muammo-chiqsa)

---

## 1. Nima qurilgan

O'zbekiston qonunchiligi bo'yicha **batch RAG API**. Bitta so'rovda 1-10 savol
qabul qiladi va har bir savol uchun uchta narsa qaytaradi:

| Nima | Nega kerak |
|---|---|
| `answer` | javob matni, `[1]` `[2]` belgilari bilan |
| `citations[]` | manba: hujjat, modda, qism, lex.uz havolasi, **harfma-harf ko'chirilgan parcha** |
| `contexts[]` | qidiruv topgan moddalar — ya'ni **qidiruvni alohida baholash mumkin** |

**Indeks:** 25 hujjat (Konstitutsiya, 20 kodeks, 4 qonun), **7 383 modda**,
2026-07 snapshot. Bitta bo'lak = bitta modda, shuning uchun `citations[].article`
taxmin emas.

**Muhim xususiyat:** model ko'chirgan har bir parcha **kod tomonidan** manba
matniga qarshi tekshiriladi. Parcha moddada aynan topilmasa — o'sha sitata
tashlanadi; hech qaysi sitata qolmasa — javob berilmaydi (`no_answer`).
Havola (`chunk_id`, `url`) **modeldan olinmaydi**, qidiruv natijasidan olinadi —
shuning uchun tizim ko'rmagan manbaga havola **chiqishi mumkin emas**.

---

## 2. Ishga tushirish

### Kerakli narsalar

- Python 3.11+
- Docker (Qdrant uchun)
- ~2 GB disk (embedding modeli)
- LLM provayder kaliti — **ixtiyoriy**, 9-bo'limga qarang

### Buyruqlar

```bash
# 1. sozlama fayli
cp .env.example .env

# 2. Python muhiti
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. vektor bazasi
docker compose up -d

# 4. indeks qurish — BIR MARTA, ~85 daqiqa CPU'da
python ingest/ingest_corpus.py

# 5. server
uvicorn backend.main:app --port 8000
```

> **4-qadam haqida.** 7 383 moddani vektorga aylantirish CPU'da ~85 daqiqa
> oladi (o'lchangan: ~690 ms har modda). Uzilsa qo'rqmang — dastur qayerga
> yetganini yozib boradi va qayta yurgizganda **qoldirgan joydan** davom etadi.
> Bu ish **bir marta** bajariladi; keyin Qdrant diskda saqlanadi.

> **Windows uchun.** PowerShell'da konsolni UTF-8 ga o'tkazish kerak, aks holda
> `oʻ` va `gʻ` harflari buzilib ko'rinadi:
> ```powershell
> [Console]::OutputEncoding = [Text.Encoding]::UTF8
> $env:PYTHONUTF8 = "1"
> ```
> To'liq yo'riqnoma — `TESTING.md`.

### Sozlamalar

`.env` fayldagi muhim qiymatlar:

```
RAG_API_KEYS=test-key-lex-2026     # test kaliti
RAG_LLM_BACKEND=offline            # provayder (9-bo'limga qarang)
QDRANT_COLLECTION=lex_uz
RAG_SYNC_DEADLINE_S=120            # bundan oshsa 202 qaytadi
RAG_RATE_LIMIT_INTERVAL_S=1.0      # sekundda bitta batch
RAG_RESULT_RETENTION_DAYS=30       # natijalar necha kun saqlanadi
```

---

## 3. Tez tekshiruv — 5 daqiqa

Bitta buyruq hamma narsani tekshiradi: muhit, Qdrant ulanishi, indeks holati,
uch tilda qidiruv, 86 kontrakt testi, provayder sozlamasi.

```bash
python scripts/verify.py --quick
```

Kutilgan natija — oxirida:

```
VERDICT: READY
```

Agar server ham ishlab turgan bo'lsa va 100 savollik baholashni ham
yurgizmoqchi bo'lsangiz:

```bash
python scripts/verify.py --base-url http://localhost:8000
```

(`--quick` bo'lmasa, u 100 savolli baholashni ham bajaradi — bu uzoq.)

---

## 4. Birinchi so'rov — `curl`

Server ishlab turganda. Autentifikatsiya: `Authorization: Bearer <kalit>`.

### Servis tirikmi

```bash
curl http://localhost:8000/v1/health
```

Bu yagona endpoint — **kalit talab qilmaydi**.

### Indeks va versiyalar

```bash
curl -H "Authorization: Bearer test-key-lex-2026" \
     http://localhost:8000/v1/meta
```

Bu yerda `index_version`, `pipeline_version`, qaysi model ishlatilayotgani,
cheklovlar (`limits`) va **halol ogohlantirishlar** (`caveats`) ko'rinadi.

### Batch so'rov — asosiy endpoint

```bash
curl -X POST http://localhost:8000/v1/rag/batch \
  -H "Authorization: Bearer test-key-lex-2026" \
  -H "Content-Type: application/json" \
  -d '{
    "batch_id": "test-001",
    "items": [
      {"id": "q1", "question": "Yillik asosiy mehnat taʼtili necha kun?"},
      {"id": "q2", "question": "Oʻgʻrilik uchun qanday jazo belgilangan?"},
      {"id": "q3", "question": "Какой срок исковой давности по трудовым спорам?"}
    ]
  }'
```

> **Windows PowerShell'da** `curl` boshqacha ishlaydi. Oson yo'li — so'rovni
> faylga yozib yuborish:
> ```powershell
> curl.exe -X POST http://localhost:8000/v1/rag/batch `
>   -H "Authorization: Bearer test-key-lex-2026" `
>   -H "Content-Type: application/json" `
>   --data "@sorov.json"
> ```

### Saqlangan natijani qayta olish

```bash
curl -H "Authorization: Bearer test-key-lex-2026" \
     http://localhost:8000/v1/rag/batch/test-001
```

### Bitta savol (debug uchun, ixtiyoriy endpoint)

```bash
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer test-key-lex-2026" \
  -H "Content-Type: application/json" \
  -d '{"question": "Fuqarolik huquqiga koʻra shartnoma qachon bekor qilinadi?"}'
```

### Endpointlar ro'yxati

| Endpoint | Vazifasi | Spec |
|---|---|---|
| `POST /v1/rag/batch` | 1-10 savol, javob + manba + kontekst | majburiy |
| `GET /v1/rag/batch/{id}` | natijani qayta olish, async poll | majburiy |
| `GET /v1/meta` | model, indeks versiyasi, cheklovlar | majburiy |
| `GET /v1/health` | servis holati (kalit kerak emas) | majburiy |
| `POST /v1/rag/query` | bitta savol, debug | ixtiyoriy |

Javoblar `application/json; charset=utf-8` bilan qaytadi — `oʻ`, `gʻ`, `ʼ`
buzilmaydi.

---

## 5. Javobda nima qaytadi

Bitta savolning javobi (qisqartirilgan):

```json
{
  "id": "q1",
  "status": "ok",
  "answer": "Har yilgi asosiy eng kam mehnat taʼtili yigirma bir kalendar kun [1].",
  "citations": [
    {
      "ref": 1,
      "doc_id": "lex:6257288",
      "document": "Oʻzbekiston Respublikasining Mehnat kodeksi",
      "article": "217-modda",
      "part": null,
      "url": "https://lex.uz/docs/-6257288",
      "quote": "yigirma bir kalendar kunni tashkil etadi",
      "doc_status": "amalda",
      "edition_date": "2026-07-01"
    }
  ],
  "contexts": [
    {
      "rank": 1,
      "chunk_id": "lex:6257288#art-217",
      "document": "Oʻzbekiston Respublikasining Mehnat kodeksi",
      "article": "217-modda",
      "score": 1.63,
      "text": "Har yilgi asosiy eng kam mehnat taʼtilining davomiyligi ..."
    }
  ],
  "timing_ms": {"retrieval": 188, "generation": 2104, "total": 2292}
}
```

### Nimaga qarash kerak

| Maydon | Nimani ko'rsatadi |
|---|---|
| `status` | `ok` · `no_answer` · `error` |
| `citations[].quote` | **modda matnida aynan bor** — tekshirilgan |
| `citations[].article` | aniq modda, taxmin emas |
| `contexts[]` | qidiruv nimani topgan — **qidiruvni alohida baholash uchun** |
| `contexts[].rank` | to'g'ri modda nechanchi o'rinda (Hit@5 / MRR shundan) |
| `timing_ms` | qidiruv va generatsiya vaqti **alohida** |

### `status` qiymatlarining ma'nosi

**`ok`** — javob bor, sitata tekshiruvdan o'tgan.

**`no_answer`** — javob **ataylab** berilmagan. Ikki sabab bo'lishi mumkin:
kontekstda javob yo'q, yoki model ko'chirgan parcha moddada topilmadi.
Ikkinchi holatda `debug.notes` sababini yozadi. **Bu xato emas** — tizim
asossiz javob berishdan bosh tortdi.

**`error`** — savol qayta ishlanmadi. `error.code` sababni aytadi:

| Kod | Ma'nosi |
|---|---|
| `RETRIEVAL_FAILED` | shu savolning qidiruvi yiqildi |
| `INDEX_UNAVAILABLE` | vektor baza bilan aloqa yo'q |
| `LLM_TIMEOUT` | provayder javob bermadi |
| `LLM_ERROR` | provayder xatosi (**deyarli doim kvota** — 10-bo'limga qarang) |
| `CONTEXT_TOO_LONG` | kontekst modelga sig'madi |
| `INVALID_LANGUAGE` | so'ralgan til qo'llanmaydi |

Bitta savoldagi xato **butun batch'ni yiqitmaydi**: batch `200` qaytaradi va
xato bo'lgan savolning `contexts[]` **to'ldirilgan** bo'ladi — shuning uchun
qidiruv xatosini generatsiya xatosidan ajratish mumkin.

---

## 6. Status kodlarini tekshirish

Spetsifikatsiya §5 talab qilgan hamma holat ishlaydi. Har birini qo'lda
tekshirish mumkin:

| Kod | Qanday chaqirish |
|---|---|
| `200` | normal batch |
| `202` | 120 sekunddan oshsa (yoki `RAG_SYNC_DEADLINE_S=1` qo'yib sinang) |
| `400` | 11 savol yuborish, yoki takroriy `id`, yoki bo'sh savol, yoki 2000+ belgi |
| `401` | `Authorization` header'siz so'rov |
| `403` | noto'g'ri kalit bilan so'rov |
| `404` | mavjud bo'lmagan `batch_id` ni `GET` qilish |
| `409` | bir xil `batch_id`, **boshqa savollar** bilan |
| `429` | ikki batch'ni darhol ketma-ket yuborish |
| `503` | butun batch yiqilsa (masalan provayder umuman ishlamasa) |

### Misollar

```bash
# 401 — kalit yo'q
curl -i -X POST http://localhost:8000/v1/rag/batch \
  -H "Content-Type: application/json" \
  -d '{"batch_id":"x","items":[{"id":"q1","question":"test savol"}]}'

# 403 — kalit noto'g'ri
curl -i -X POST http://localhost:8000/v1/rag/batch \
  -H "Authorization: Bearer notogri-kalit" \
  -H "Content-Type: application/json" \
  -d '{"batch_id":"x","items":[{"id":"q1","question":"test savol"}]}'

# 409 — bir xil batch_id, boshqa savollar
# (avval test-001 ni yuborgan bo'lsangiz)
curl -i -X POST http://localhost:8000/v1/rag/batch \
  -H "Authorization: Bearer test-key-lex-2026" \
  -H "Content-Type: application/json" \
  -d '{"batch_id":"test-001","items":[{"id":"z9","question":"butunlay boshqa savol"}]}'
```

### Idempotentlik

Bir xil `batch_id` **bir xil savollar** bilan qayta kelsa — saqlangan natija
qaytadi, **qayta hisoblanmaydi**. Bir xil `batch_id` **boshqa savollar** bilan
kelsa — `409`. Farqni so'rov mazmunining hash'i aniqlaydi.

Shuning uchun so'rov uzilib qolsa, xuddi shu so'rovni qayta yuborish **xavfsiz**.

### Avtomat testlar

Spetsifikatsiya §7 checklist'i test sifatida yozilgan:

```bash
python -m pytest tests/ -q
```

Kutilgan: `86 passed, 1 skipped`. Testlar offline generator bilan ishlaydi —
provayder kaliti kerak emas, pul ketmaydi.

---

## 7. To'liq baholash — 100 savol

Spetsifikatsiya §1 ssenariysi avtomatlashtirilgan: 100 savol, 10 batch,
ketma-ket, `429` da kutish, `202` da poll qilish.

### Test to'plami

`data/eval/testset_100.csv` — spetsifikatsiya §6 tarkibiga mos:

| Kategoriya | Soni |
|---|---|
| `faktik` | 35 |
| `raqamlar` | 15 |
| `koʻp_modda` | 15 |
| `kirill_rus` | 10 |
| `amaldagi_tahrir` | 10 |
| `javobsiz` | 10 |
| `noaniq` | 5 |

CSV sarlavhasi:

```
id,batch,category,question,language,expected_answer,expected_doc_id,expected_article,answerable
```

**`expected_answer` va `expected_article` ataylab bo'sh.** Sababi 11-bo'limda.
Bu ikki ustunni to'ldirsangiz, `score_eval.py` qolgan 3 metrikani ham hisoblaydi.

> ⚠️ **Diqqat:** `expected_*` ustunlarini to'ldirgandan keyin
> `python scripts/build_testset.py` ni **qayta yurgizmang** — u faylni
> boshidan yozadi va to'ldirganlaringiz yo'qoladi. Avval nusxa oling.

### Yurgizish

```bash
# server ishlab turgan holda, boshqa terminalda:
python scripts/run_eval.py --run-id r1
```

Natijalar `data/runs/r1/b01.json` ... `b10.json` ga yoziladi.

Foydali flaglar:

```bash
python scripts/run_eval.py --run-id r1 --batches b01,b02   # faqat ikki batch
python scripts/run_eval.py --run-id r1 --top-k 10          # boshqa top_k
python scripts/run_eval.py --run-id r1 --base-url http://localhost:8000
```

Uzilib qolsa — **xuddi shu buyruqni qayta yurgizing**, bajarilgan batch'lar
o'tkazib yuboriladi.

### Ballash

```bash
python scripts/score_eval.py --run-id r1
```

Hisobot terminalga chiqadi. JSON ko'rinishida ham saqlash mumkin:

```bash
python scripts/score_eval.py --run-id r1 --json hisobot.json
```

Agar run to'liq bo'lmasa, hisobot tepasida `PARTIAL RUN` deb yoziladi va
metrikalar **javob kelgan savollar ustidan** hisoblanadi — qamrov alohida
ko'rsatiladi.

---

## 8. Metrikalar va maqsadlar

Spetsifikatsiya §6 jadvali. Quyida **bizda o'lchangan** natijalar — 100 savol,
real HTTP endpoint orqali, generatsiya offline generator bilan (sababi
10-bo'limda):

| Metrika | O'lchangan | Maqsad | Holat |
|---|---|---|---|
| Hit@5 (hujjat) | **0.978** | ≥ 0.90 | o'tdi |
| Context precision | **0.960** | ≥ 0.60 | o'tdi |
| Faithfulness | **1.000** | ≥ 0.95 | o'tdi |
| Citation accuracy | **0.966** | ≥ 0.90 | o'tdi |
| To'qilgan manba | **0** | 0 | o'tdi |
| Eskirgan norma | **0** | 0 | o'tdi |
| Latency p50 / p95 | **0.23s / 0.27s** | ≤ 5s / ≤ 12s | o'tdi |
| Xato ulushi | **0.000** | ≤ 0.02 | o'tdi |
| No-answer aniqligi | **0.96** | — | — |
| Hit@5 (modda) | — | ≥ 0.80 | gold kerak |
| MRR | — | ≥ 0.70 | gold kerak |
| Answer correctness | — | ≥ 1.6 | gold kerak |

**Latency generatsiyani hisobga olmaydi** — u provayderga bog'liq, loyihaga
emas. O'lchangan taqsimot: mahalliy ish **272 ms** (embedding 153, Qdrant 35,
BM25 0.5), LLM kutish **21 347 ms**. Ya'ni **vaqtning 98.7% i provayderda**.

Kategoriyalar bo'yicha (to'g'ri hujjat 5 ta ichida):

```
faktik            35/35
raqamlar          15/15
koʻp_modda        15/15
kirill_rus        10/10
amaldagi_tahrir   10/10
noaniq             3/5
```

---

## 9. O'z LLM kalitingizni qo'yish

Qidiruv **hamma provayder uchun bir xil** — faqat javob yozish qismi
o'zgaradi.

### Kalitsiz sinash (eng tez yo'l)

```
RAG_LLM_BACKEND=offline
```

Bu — deterministik ekstraktiv javob beruvchi: **kalit kerak emas, pul
ketmaydi**. Qidiruv, sitata tekshiruvi, barcha status kodlari va kontrakt —
hammasi to'liq ishlaydi. Cheklovi: `no_answer` hukmini chiqarmaydi va
`answer_language` ni qo'llamaydi.

Agar maqsad **API kontraktini va qidiruvni** tekshirish bo'lsa, shu yetarli.

### O'z kalitingiz bilan

`.env` da ikki qatorni o'zgartiring:

```
RAG_LLM_BACKEND=anthropic        # yoki: gemini | groq | openai | deepseek | ...
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Qo'llanadigan provayderlar:

| Provayder | `RAG_LLM_BACKEND` | Kalit o'zgaruvchisi |
|---|---|---|
| Anthropic (Claude) | `anthropic` | `ANTHROPIC_API_KEY` |
| Google (Gemini) | `gemini` | `GEMINI_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Groq | `groq` | `GROQ_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| OpenRouter, Together, Fireworks, Mistral, xAI, Cerebras | preset nomi | mos kalit |
| Ollama, LM Studio (mahalliy) | `ollama`, `lmstudio` | kerak emas |
| Boshqa OpenAI-mos servis | `custom` | `RAG_LLM_BASE_URL` + kalit |

Kalitni **oldin sinab ko'ring** — qaysi modellarga kira olishi hisobga bog'liq:

```bash
python scripts/list_models.py anthropic --probe
```

`--probe` bitta real savolga javob yozib ko'radi. Model nomini `.env` dagi
`RAG_ANTHROPIC_MODEL` / `RAG_GEMINI_MODEL` / `RAG_GROQ_MODEL` ga yozing.
Alias (`gemini-flash-latest` kabi) emas, **aniq versiyani** qo'yish yaxshi —
aks holda model ostingizdan o'zgaradi, `pipeline_version` esa o'zgarmaydi.

> **Anthropic uchun:** Console API kaliti kerak (`sk-ant-api03-...`).
> Claude Code obuna tokeni (`sk-ant-oat01-...`) autentifikatsiyadan o'tadi,
> lekin Messages API'da `429` beradi.

---

## 10. Normal, lekin xato kabi ko'rinadigan narsalar

Bu to'rt holat **to'g'ri ishlash** — buzilish emas.

### 1. Bitta batch 10 savolgacha

20 yoki 30 savolni **bitta** so'rovda yuborish `400` qaytaradi:

```
items: List should have at most 10 items after validation, not 20
```

Bu spetsifikatsiya §2 talabi. Ko'proq savol uchun **bir nechta batch** —
baholash ssenariysi aynan shunday: 10 batch × 10 savol. `run_eval.py` buni
avtomat qiladi.

### 2. Sekin batch `202` qaytaradi

120 sekunddan oshsa:

```json
{"status": "processing", "poll": "/v1/rag/batch/test-001"}
```

**Ish yo'qolmaydi** — fonda davom etadi, `GET` bilan olinadi. O'lchangan misol:
bepul Groq kaliti bilan 10 savollik batch **285 sekund** ketdi va 33 marta poll
talab qildi. Pullik kalitda chegara ichida tugaydi va `200` qaytadi.

### 3. Sekundda bitta batch

Darhol yuborilgan ikkinchi batch `429` va `Retry-After` oladi. Bu ham spec
talabi (kamida 1 batch/sekund). `run_eval.py` `Retry-After` ni hurmat qiladi.

### 4. `LLM_ERROR` — deyarli doim kvota

Bu xato chiqsa, birinchi gumon — **provayder kvotasi**, loyiha emas. Bitta savol
~2 500 token oladi (5 modda kontekst).

O'lchangan misol: bepul Groq kaliti kunda **200 000 token** beradi. 100 savol
~250 000 token talab qiladi — ya'ni **bepul kalitga 100 savol sig'maydi**.
Bizning real run'da 40 savoldan keyin kalit tugadi va qolgani to'xtadi; xato
matni Groq'ning o'zidan:

```
tokens per day (TPD): Limit 200000, Used 197444
```

Yechimi: pullik kalit, yoki `.env` da `RAG_GEN_MIN_INTERVAL_S=20` (so'rovlar
orasini ochadi).

**Shu sababdan** 8-bo'limdagi jadval offline generator bilan o'lchangan: u
qidiruvni va sitata darvozasini **provayder kvotasidan mustaqil** o'lchash
imkonini beradi.

---

## 11. Nima o'lchangan, nima o'lchanmagan

Bu bo'lim ataylab halol yozilgan.

### O'lchangan — real ishga tushirib

| Nima | Qanday tekshirilgan |
|---|---|
| HTTP kontrakt (5 endpoint) | 86 avtomat test + qo'lda `curl` |
| Barcha status kodlari | 200/202/400/401/403/404/409/429/503 |
| Qidiruv, uch tilda | qo'lda tasdiqlangan 5 savol: 5/5 birinchi o'rinda |
| 100 savollik baholash | 10 batch ketma-ket, real HTTP orqali |
| Groq provayderi | to'liq, 60 real savol |
| Idempotentlik, 409, async 202 | test va real run'da |
| Spec §4 majburiy maydonlar | real javob qatorma-qator solishtirildi |

### O'lchanmagan — va nega

| Nima | Holat |
|---|---|
| **Claude real servisda** | so'rov shakli, structured output va token hisobi soxta endpointda tasdiqlandi; real servisda emas — mavjud kalit `429` berdi |
| **Gemini generatsiyasi** | kalit va model ro'yxati ishlaydi; javob yozish emas — sinov paytida barcha model `503` bergan |
| **OpenAI real servisda** | Groq bilan **bir xil kod yo'li** (Groq to'liq tekshirilgan), plus 4 parametrni rad etadigan soxta endpointda sinaldi |
| **Pullik kalitda 100 savol** | faqat bepul Groq kalitida o'lchangan, u yerda kvota cheklaydi |
| **Airflow DAG yurishi** | `apache-airflow` sinov mashinasida o'rnatilmagan; papka to'qnashuvi tuzatildi, lekin DAG yurgizilmagan |
| **Hit@5 (modda), MRR, answer correctness** | gold javoblar bo'sh — pastda |

### Nega `expected_article` va `expected_answer` bo'sh

Savol qaysi moddaga tushishi **kerak** ekanini yozish uchun statutlarni o'qish
kerak. Agar bu ustunlarni **tizimning o'z chiqishidan** to'ldirsak, baholash
tizimni **o'zi bilan o'zini** tekshirgan bo'lardi — ya'ni ballar ma'nosiz
bo'lardi.

Spetsifikatsiya §6 ham test to'plamini **«tester tomonida»** deb belgilagan.

Shuning uchun ular bo'sh qoldirilgan. To'ldirilsa, `score_eval.py` qolgan 3
metrikani darhol hisoblaydi — kod tayyor.

### Ma'lum cheklovlar

Bular `GET /v1/meta` ning `caveats` maydonida ham yozilgan — ya'ni o'zgarmas
qiymat «ishlayotgan tekshiruv» deb tushunilmaydi:

- `doc_status` doim `"amalda"` — 25 hujjatning hammasi kuchda, ya'ni «eskirgan
  norma» metrikasi yiqila olmaydi
- `edition_date` hujjat bo'yicha — korpus sanali snapshot, tahrir tarixi yo'q
- `as_of_date` qayd qilinadi, lekin xizmat qilinmaydi — oldingi sana `debug` ga
  ogohlantirish qo'shadi
- `temperature` / `seed` provayderga bog'liq: Groq ikkisini ham, Gemini faqat
  birinchisini, Anthropic SDK 2.x hech birini qabul qilmaydi. **Qidiruv esa
  deterministik** — teng ballda ham tartib har doim bir xil
- indeksdan **60 xom bo'lak chiqarilgan** (modda strukturasi yo'q, majburiy
  `article`/`part` ni to'ldirib bo'lmaydi); chiqarilgan son `/v1/meta` da
  ko'rinadi

### Halol xulosa

Sitata darvozasi **to'qilgan manbani strukturaviy imkonsiz** qiladi — bu
kafolat, chunki havola kodning o'zidan chiqadi, modeldan emas.

Lekin darvoza **ko'chirilgan parchani** tekshiradi, **xulosani** emas. Ya'ni
model to'g'ri moddadan to'g'ri parcha ko'chirib, undan noto'g'ri xulosa
chiqarishi nazariy jihatdan mumkin. Shuning uchun *«hech qachon xato
qilmaydi»* degan da'vo yo'q. *«Manbasiz javob bermaydi»* — bu o'lchangan.

---

## 12. Muammo chiqsa

### Birinchi qadam — har doim

```bash
python scripts/verify.py --quick
```

U muammoni **qaysi qatlamda** ekanini aytadi: muhit, Qdrant, indeks, qidiruv,
kontrakt yoki provayder.

### Tez-tez uchraydigan holatlar

| Simptom | Sabab | Yechim |
|---|---|---|
| `INDEX_UNAVAILABLE` | Qdrant ishlamayapti | `docker compose up -d` |
| Barcha savol `no_answer` | indeks bo'sh | `python ingest/ingest_corpus.py` |
| Barcha savol `LLM_ERROR` | kalit yo'q yoki kvota tugagan | `list_models.py <provayder> --probe` |
| Birinchi savol juda sekin | model yuklanishi | normal; keyingilari ~0.24s |
| `oʻ` / `gʻ` buzilib ko'rinadi | konsol kodlashi | Windows: UTF-8 (2-bo'lim) |
| `429` | sekundda bir nechta batch | `Retry-After` ni kutish |
| `409` | bir xil `batch_id`, boshqa savollar | yangi `batch_id` qo'ying |

### Hisobot yuborish

Muammo chiqsa, quyidagilar yetarli:

1. `python scripts/verify.py --quick` chiqishi
2. So'rov (`batch_id` bilan) va qaytgan javob
3. `curl -H "Authorization: Bearer test-key-lex-2026" http://localhost:8000/v1/meta`

`/v1/meta` ichida `index_version` va `pipeline_version` bor — ular bilan
qaysi holat sinalganini aniq tiklash mumkin.

---

## Qisqa xulosa

```bash
# 1. tayyormi?
python scripts/verify.py --quick               # → VERDICT: READY

# 2. kontrakt to'g'rimi?
python -m pytest tests/ -q                     # → 86 passed

# 3. qo'lda bitta so'rov
curl -X POST http://localhost:8000/v1/rag/batch \
  -H "Authorization: Bearer test-key-lex-2026" \
  -H "Content-Type: application/json" \
  -d '{"batch_id":"t1","items":[{"id":"q1","question":"Yillik taʼtil necha kun?"}]}'

# 4. 100 savollik baholash
python scripts/run_eval.py   --run-id r1
python scripts/score_eval.py --run-id r1
```

Spetsifikatsiya bilan ziddiyat topsangiz — **spetsifikatsiya ustun**, ayting,
tuzatamiz.
