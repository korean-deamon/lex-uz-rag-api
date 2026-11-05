"""
Retrieval for the /v1 RAG API: article-level hybrid search over the indexed
Uzbek legal corpus, reranked, returning the `contexts[]` the spec requires.

Three stages, in order:

  1. Candidates   Dense vector search (multilingual-e5) over the whole index,
                  plus BM25 over the same articles, plus -- when the question
                  names a specific code -- a vector search scoped to that code.
                  Scoping matters because generic articles (definitions,
                  punishment types, procedural boilerplate) otherwise outrank
                  the one specific article on vocabulary overlap alone.

  2. Rerank       Reciprocal rank fusion of the dense and lexical rankings.
                  Dense search alone misses questions whose decisive term is
                  rare ("oʻgʻrilik"), BM25 alone misses paraphrases and
                  cross-language questions; fusion recovers both, which is what
                  the spec's Hit@5 (modda) and MRR targets are measured on.

  3. Report       Every returned context carries a genuine dense cosine as
                  `retrieval_score` and its fused score as `rerank_score`, so
                  the tester can analyse whether reranking actually helps.

Determinism: both stages are pure functions of (question, index). Ties break on
chunk_id, never on dict or search-result ordering, so the same question against
the same index returns the same contexts in the same order every time -- the
spec's reproducibility requirement.
"""
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from backend import config, corpus_meta, embedding, lang

# --- fusion weights --------------------------------------------------------
# Rank-only fusion (RRF) was tried first and is wrong for this corpus: e5
# cosines here sit in a very narrow band (a question's top-40 candidates
# typically span 0.79-0.81), so dense *rank* carries almost no information
# while dense *score* still does, and BM25 margins are large and meaningful
# (an exact title hit scores 12.5 where the next article scores 7.9). Rank
# fusion discards exactly the signal that matters and buries the right article.
#
# So each retriever contributes a score mapped to 0..1 on an absolute scale:
#   dense    (cos - DENSE_FLOOR) / DENSE_SPAN, clamped -- an absolute cosine,
#            not a pool-relative one, so a pool of uniformly mediocre matches
#            cannot promote its best member to 1.0.
#   lexical  bm25 / (bm25 + BM25_HALF) -- saturating, so a strong exact match
#            counts for much more than a weak incidental word overlap.
# Dense decides when nothing matches lexically; a decisive lexical hit wins
# over a dense ranking that is statistically a coin flip.
W_DENSE = 1.0
W_LEXICAL = 0.8
# "The question names this code" is a flat membership bonus, not a third
# retriever. It was first written as an extra dense contribution for scoped
# hits, which silently doubled dense's weight whenever a code was detected and
# let a 0.845-cosine generic article ("Jinoyatning tushunchasi") outscore the
# article whose title is the question's decisive word. Flat means it cancels out
# between candidates from the same code, leaving the retrievers to decide.
W_CODE_BONUS = 0.3

# e5 similarities below ~0.75 are unrelated text on this corpus; 0.95 is a
# near-verbatim match. Mapping that band onto 0..1 keeps retrieval_score and
# rerank_score comparable between questions, which is what the spec wants them
# for (threshold analysis, "is the reranker helping").
DENSE_FLOOR = 0.75
DENSE_SPAN = 0.20
# BM25 score at which the lexical contribution reaches half its weight.
BM25_HALF = 6.0

MAX_FUSED = W_DENSE + W_LEXICAL + W_CODE_BONUS

_qdrant: QdrantClient | None = None

# The collection also holds chunks from the older raw-scrape pipeline
# (ingest/ingest.py): whole lex.uz documents split by character count, with no
# article structure in their payload. They are real lex.uz text, but a citation
# built from one cannot carry `article` or `part`, which the spec makes
# mandatory, and their chunk ids are not stable across a reindex. So /v1 serves
# the article-level corpus only, and GET /v1/meta reports how many points were
# excluded rather than letting them disappear quietly.
CORPUS_SOURCE = "uzbek_legal_corpus"
_CORPUS_CONDITION = qmodels.FieldCondition(
    key="source", match=qmodels.MatchValue(value=CORPUS_SOURCE)
)


def qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _qdrant


def _corpus_filter(codes: list[str] | None = None) -> qmodels.Filter:
    must: list = [_CORPUS_CONDITION]
    if codes:
        must.append(
            qmodels.Filter(
                should=[
                    qmodels.FieldCondition(key="code", match=qmodels.MatchValue(value=c))
                    for c in codes
                ]
            )
        )
    return qmodels.Filter(must=must)


# --------------------------------------------------------------------------
# Code ("tree") detection -- uz-Latn, uz-Cyrl and ru triggers per document.
# Ordered most-specific-first: "jinoyat-protsessual" must be tested before
# "jinoyat", otherwise every procedural question lands in the criminal code.
# --------------------------------------------------------------------------
CODE_TRIGGERS: list[tuple[str, list[str]]] = [
    (r"jinoyat[\s-]?protsessual|жиноят[\s-]?процессуал|уголовно[\s-]*процессуальн", ["jinoyat_protsessual_kodeksi"]),
    (r"jinoyat[\s-]?ijroiya|жиноят[\s-]?ижроия|уголовно[\s-]*исполнительн", ["jinoyat-ijroiya_kodeksi"]),
    (r"jinoyat kodeks|жиноят кодекс|уголовн\w* кодекс", ["jinoyat_kodeksi"]),
    (r"fuqarolik protsessual|фуқаролик процессуал|гражданск\w*\s+процессуальн", ["fuqarolik_protsessual_kodeksi"]),
    (r"iqtisodiy protsessual|иқтисодий процессуал|экономическ\w*\s+процессуальн", ["iqtisodiy_protsessual_kodeksi"]),
    (r"maʻmuriy javobgarlik|маъмурий жавобгарлик|административн\w*\s+ответственност", ["mamuriy_javobgarlik_kodeksi"]),
    (r"maʻmuriy sud|маъмурий суд|административн\w*\s+судопроизводств", ["mamuriy_sud_ishlarni_yuritish_kodeksi"]),
    (r"soliq kodeks|солиқ кодекс|налогов\w* кодекс", ["soliq_kodeksi"]),
    (r"mehnat kodeks|меҳнат кодекс|трудов\w* кодекс", ["mehnat_kodeksi"]),
    (r"oila kodeks|оила кодекс|семейн\w* кодекс", ["oila_kodeksi"]),
    (r"yer kodeks|ер кодекс|земельн\w* кодекс", ["yer_kodeksi"]),
    (r"suv kodeks|сув кодекс|водн\w* кодекс", ["suv_kodeksi"]),
    (r"havo kodeks|ҳаво кодекс|воздушн\w* кодекс", ["havo_kodeksi"]),
    (r"bojxona kodeks|божхона кодекс|таможенн\w* кодекс", ["bojxona_kodeksi"]),
    (r"budjet kodeks|бюджет кодекс|бюджетн\w* кодекс", ["budjet_kodeksi"]),
    (r"saylov kodeks|сайлов кодекс|избирательн\w* кодекс", ["saylov_kodeksi"]),
    (r"shaharsozlik kodeks|шаҳарсозлик кодекс|градостроительн\w* кодекс", ["shaharsozlik_kodeksi"]),
    (r"konstitutsiya|конституция|конституци\w*", ["konstitutsiya"]),
    (r"markaziy bank|марказий банк|центральн\w*\s+банк", ["markaziy_bank_qonuni"]),
    (r"raqobat|рақобат|конкуренци", ["raqobat_qonuni"]),
    (r"investitsiya|инвестиция|инвестици", ["investitsiyalar_faoliyati_qonuni"]),
    (r"qimmatli qog|қимматли қоғоз|ценн\w*\s+бумаг", ["qimmatli_qogozlar_bozori_qonuni"]),
    (r"tadbirkorlik|тадбиркорлик|предпринимательск", ["tadbirkorlik_faoliyati_qonuni"]),
    # The civil code is split into two documents, so a bare "fuqarolik kodeksi"
    # scopes to both parts rather than guessing one.
    (
        r"fuqarolik kodeks|фуқаролик кодекс|гражданск\w* кодекс",
        ["fuqarolik_kodeksi_1qism", "fuqarolik_kodeksi_2qism"],
    ),
]

_COMPILED_TRIGGERS = [(re.compile(p, re.IGNORECASE), codes) for p, codes in CODE_TRIGGERS]

# Words that appear in nearly every legal question and so carry no retrieval
# signal. Dropping them keeps BM25 focused on the term actually being asked
# about. Code-name roots are included: "jinoyat" matches almost every article
# title inside the criminal code, drowning out the distinctive term.
STOPWORDS = {
    # uz-Latn structural
    "kodeks", "kodeksi", "kodeksiga", "kodeksga", "kodeksda", "qonun", "qonunga",
    "qonuniga", "qonunda", "toʻgʻrisidagi", "toʻgʻrisida", "koʻra", "uchun",
    "qanday", "qancha", "necha", "nechta", "bormi", "mumkin", "mumkinmi",
    "qilinadi", "belgilangan", "belgilanadi", "nazarda", "tutilgan",
    "hisoblanadi", "beriladi", "qoʻyiladi", "amalga", "oshiriladi", "qilish",
    "qilinishi", "nima", "bilan", "yoki", "ham", "boʻladi", "boʻlishi",
    "hollarda", "holda", "deb", "modda", "moddasi", "moddasida",
    # code-name roots
    "jinoyat", "javobgarlik", "protsessual", "ijroiya", "fuqarolik", "mehnat",
    "soliq", "oila", "budjet", "bojxona", "saylov", "shaharsozlik", "markaziy",
    "investitsiya", "qimmatli", "qogozlar", "qogʻozlar", "bozori",
    "tadbirkorlik", "raqobat", "konstitutsiya", "maʻmuriy", "iqtisodiy",
    # ru structural
    "какой", "какая", "какие", "каков", "сколько", "что", "это", "для", "или",
    "при", "если", "быть", "может", "можно", "кодекс", "кодексом", "кодекса",
    "кодексе", "закон", "законом", "закона", "статья", "статьи", "статье",
    "установлен", "установлена", "установлено", "предусмотрен", "согласно",
    "соответствии", "является", "срок",
}

# Folk / compound terms expanded to the words the statute text actually uses.
# The corpus spells theft "oʻgʻrilik"; "oʻgʻirlik" is the more common everyday
# spelling, so both forms are indexed into each other.
SYNONYMS = {
    "poraxoʻrlik": ["pora"],
    "korrupsiya": ["pora", "mansab"],
    "oʻgʻirlik": ["oʻgʻrilik"],
    "oʻgʻrilik": ["oʻgʻirlik"],
    # Statutes write "har yilgi"; questions almost always say "yillik".
    "yillik": ["yilgi"],
    "yilgi": ["yillik"],
    # Tax abbreviations people actually use. The statute never writes "QQS",
    # only "qoʻshilgan qiymat soligʻi", so without this a question using the
    # abbreviation shares no token with the section that answers it.
    "qqs": ["qoʻshilgan", "qiymat"],
    "ndfl": ["jismoniy", "shaxslar", "daromad"],
    "bhm": ["bazaviy", "hisoblash", "miqdori"],
}

_TOKEN = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

# Uzbek Cyrillic -> Uzbek Latin. The corpus is written in Latin, so without this
# a Cyrillic question ("ўғрилик") shares no lexical token with the article that
# answers it ("oʻgʻrilik") and falls back to dense search alone, which on this
# corpus is not enough to pick the right article. Multi-character values are
# applied before single ones so "ю" does not become "y" + "u" out of order.
_CYRL_TO_LATN = {
    "ё": "yo", "ж": "j", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh",
    "ю": "yu", "я": "ya", "ў": "oʻ", "ғ": "gʻ", "қ": "q", "ҳ": "h",
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x",
    "ъ": "ʻ", "ь": "", "ы": "i", "э": "e",
}
_HAS_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def translit_cyrl_to_latn(token: str) -> str:
    return "".join(_CYRL_TO_LATN.get(ch, ch) for ch in token)


def tokenize(text: str) -> list[str]:
    text = lang.normalise(text).lower()
    tokens = [t for t in _TOKEN.findall(text) if len(t) > 2 and t not in STOPWORDS]
    expanded: list[str] = []
    for t in tokens:
        expanded.append(t)
        expanded.extend(SYNONYMS.get(t, []))
        if _HAS_CYRILLIC.search(t):
            # Keep both forms: the Latin one matches the corpus, the original
            # still matches should Cyrillic text ever be indexed.
            latin = translit_cyrl_to_latn(t)
            if latin != t and latin not in STOPWORDS:
                expanded.append(latin)
                expanded.extend(SYNONYMS.get(latin, []))
    return expanded


def detect_codes(question: str) -> list[str]:
    q = lang.normalise(question)
    for pattern, codes in _COMPILED_TRIGGERS:
        if pattern.search(q):
            return codes
    return []


# --------------------------------------------------------------------------
# Lexical (BM25) index, built once from the vector store's payloads.
# --------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class Article:
    point_id: str
    chunk_id: str
    doc_id: str
    code: str
    article_id: str
    article_display: str
    title: str
    text: str
    url: str
    doc_title: str
    doc_type: str
    edition_date: str
    doc_status: str
    section: str = ""
    chapter: str = ""


class LexicalIndex:
    """In-memory BM25 over the ~7.4k indexed articles.

    The whole corpus is a few tens of MB of text, so it is loaded once at
    startup rather than queried per request: exact lexical matching with no
    extra service, and no per-request Qdrant scrolling.
    """

    def __init__(self, articles: list[Article]):
        self.articles = articles
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        self.by_chunk_id: dict[str, int] = {}

        for idx, art in enumerate(articles):
            # Title terms count double: an article's title is the most
            # discriminative part of it, and it is short.
            # The section heading names the subject the article belongs to
            # ("QOʻSHILGAN QIYMAT SOLIGʻI"), which is the only thing telling
            # 13 identically-titled "Soliq stavkalari" articles apart. It is
            # indexed once, not doubled: it discriminates between sections,
            # not between articles inside one.
            tokens = (tokenize(art.title) * 2 + tokenize(art.text)
                      + tokenize(art.section) + tokenize(art.chapter))
            self.doc_len.append(len(tokens) or 1)
            tf: dict[str, int] = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            for term, count in tf.items():
                self.postings[term].append((idx, count))
            self.by_chunk_id[art.chunk_id] = idx

        self.n = len(articles) or 1
        self.avg_len = sum(self.doc_len) / self.n
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def search(self, question: str, limit: int, codes: list[str] | None = None) -> list[tuple[int, float]]:
        terms = tokenize(question)
        if not terms:
            return []
        allowed = set(codes) if codes else None

        scores: dict[int, float] = defaultdict(float)
        for term in set(terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for idx, tf in postings:
                if allowed is not None and self.articles[idx].code not in allowed:
                    continue
                dl = self.doc_len[idx]
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_len)
                scores[idx] += idf * (tf * (BM25_K1 + 1)) / denom

        # Sort by score desc, then chunk_id asc -- a total order, so the ranking
        # cannot vary between runs.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.articles[kv[0]].chunk_id))
        return ranked[:limit]


_lexical: LexicalIndex | None = None


@lru_cache(maxsize=1)
def _section_titles() -> dict[str, dict[str, str]]:
    """code -> section marker -> recovered title.

    The corpus stores some sections as a bare marker ("X BOʻLIM.") because the
    upstream parser dropped the title line after it. scripts/build_section_titles.py
    recovers those from the raw sources; without them, an article's position in
    the code is unreadable, which is how a question about the VAT rate ended up
    with a correct-but-useless no_answer (see that script's docstring).
    """
    path = config.ROOT / "data" / "section_titles.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _full_section(code: str, section: str) -> str:
    """The section as a human can read it, title included where recovered."""
    section = (section or "").strip()
    if not section:
        return ""
    title = _section_titles().get(code, {}).get(section)
    return f"{section} {title}" if title else section


def load_articles() -> list[Article]:
    """Read every indexed article's payload out of Qdrant, once."""
    client = qdrant()
    articles: list[Article] = []
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=config.COLLECTION,
            scroll_filter=_corpus_filter(),
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            code = pl.get("code") or ""
            article_id = pl.get("article_id") or ""
            if not code or not article_id:
                # Without both, the chunk cannot produce a spec-valid citation
                # and would collapse onto the degenerate id "lex:#art-".
                skipped += 1
                continue
            meta = corpus_meta.for_code(code)
            doc_id = meta["doc_id"] or f"lex:{(pl.get('lex_uz_doc') or '').lstrip('-')}"
            articles.append(
                Article(
                    point_id=str(p.id),
                    chunk_id=f"{doc_id}#art-{article_id}",
                    doc_id=doc_id,
                    code=code,
                    article_id=article_id,
                    article_display=pl.get("article_display") or "",
                    title=pl.get("title") or "",
                    text=pl.get("text") or "",
                    url=pl.get("url") or meta["url"],
                    doc_title=meta["doc_title"] or pl.get("code_title") or "",
                    doc_type=meta["doc_type"],
                    edition_date=meta["edition_date"],
                    doc_status=meta["doc_status"],
                    section=_full_section(code, pl.get("section") or ""),
                    chapter=pl.get("chapter") or "",
                )
            )
        if offset is None:
            break
    if skipped:
        _load_stats["skipped_malformed"] = skipped
    # Stable order so index positions are reproducible across restarts.
    articles.sort(key=lambda a: a.chunk_id)
    return articles


_load_stats: dict[str, int] = {}


def excluded_point_counts() -> dict[str, int]:
    """What /v1 does not serve from this collection, and why -- reported by
    GET /v1/meta so the gap between the collection and the index is visible."""
    client = qdrant()
    try:
        total = client.count(config.COLLECTION, exact=True).count
        corpus = client.count(
            config.COLLECTION, count_filter=_corpus_filter(), exact=True
        ).count
    except Exception:
        return {}
    return {
        "points_in_collection": total,
        "non_article_points_excluded": total - corpus,
        "malformed_article_points_excluded": _load_stats.get("skipped_malformed", 0),
    }


def lexical_index() -> LexicalIndex:
    global _lexical
    if _lexical is None:
        _lexical = LexicalIndex(load_articles())
    return _lexical


def index_size() -> int:
    return len(lexical_index().articles)


# --------------------------------------------------------------------------
# Hybrid retrieval
# --------------------------------------------------------------------------
@dataclass
class Context:
    rank: int
    chunk_id: str
    doc_id: str
    article: str
    text: str
    retrieval_score: float
    rerank_score: float
    used_in_answer: bool = False
    # carried for citation building, not part of the spec's contexts[] shape
    article_meta: Article | None = field(default=None, repr=False)

    def to_spec_dict(self) -> dict:
        d = asdict(self)
        d.pop("article_meta", None)
        return d


def embed_query(text: str) -> list[float]:
    return embedding.embed_query(lang.normalise(text))


def _dense_search(vec: list[float], limit: int, codes: list[str] | None) -> list[tuple[str, float]]:
    """-> [(chunk_id, cosine)] ; chunk_id rebuilt from the payload."""
    hits = qdrant().query_points(
        collection_name=config.COLLECTION,
        query=vec,
        limit=limit,
        query_filter=_corpus_filter(codes),
        with_payload=["code", "article_id", "lex_uz_doc"],
    ).points

    out: list[tuple[str, float]] = []
    for h in hits:
        pl = h.payload or {}
        code = pl.get("code") or ""
        article_id = pl.get("article_id") or ""
        if not code or not article_id:
            continue
        meta = corpus_meta.for_code(code)
        doc_id = meta["doc_id"] or f"lex:{(pl.get('lex_uz_doc') or '').lstrip('-')}"
        out.append((f"{doc_id}#art-{article_id}", h.score))
    return out


def retrieve(question: str, top_k: int = config.DEFAULT_TOP_K) -> tuple[list[Context], dict]:
    """Hybrid retrieve + rerank. Returns (contexts, debug_info)."""
    index = lexical_index()
    candidates = config.RETRIEVAL_CANDIDATES
    codes = detect_codes(question)

    vec = embed_query(question)
    dense = _dense_search(vec, candidates, None)
    scoped = _dense_search(vec, candidates, codes) if codes else []
    lexical = index.search(question, candidates, codes or None)

    # Dense cosine per chunk, used as the reported retrieval_score.
    cosine: dict[str, float] = {}
    for chunk_id, score in dense + scoped:
        cosine[chunk_id] = max(cosine.get(chunk_id, 0.0), score)

    def dense_component(cos: float) -> float:
        return min(1.0, max(0.0, (cos - DENSE_FLOOR) / DENSE_SPAN))

    # --- score fusion (see the weight block at the top of this module) ---
    # The code-scoped dense search is a candidate *source* -- it pulls in-code
    # articles the global top-N missed -- and contributes its cosine through the
    # same dense term as the global search, not a second one.
    fused: dict[str, float] = defaultdict(float)
    for chunk_id, cos in cosine.items():
        fused[chunk_id] += W_DENSE * dense_component(cos)
    for idx, bm25 in lexical:
        chunk_id = index.articles[idx].chunk_id
        fused[chunk_id] += W_LEXICAL * (bm25 / (bm25 + BM25_HALF))
    if codes:
        wanted = set(codes)
        for chunk_id in list(fused):
            art_idx = index.by_chunk_id.get(chunk_id)
            if art_idx is not None and index.articles[art_idx].code in wanted:
                fused[chunk_id] += W_CODE_BONUS

    # Drop anything the index doesn't actually hold (defensive: a stale doc_id
    # mapping would otherwise produce a context with no text).
    ordered = [cid for cid in fused if cid in index.by_chunk_id]
    ordered.sort(key=lambda cid: (-fused[cid], cid))

    # Lexical-only candidates have no cosine yet. Fetch their vectors and compute
    # it, so every context reports a real retrieval_score rather than a zero.
    missing = [cid for cid in ordered[:top_k] if cid not in cosine]
    if missing:
        point_ids = [index.articles[index.by_chunk_id[cid]].point_id for cid in missing]
        for point in qdrant().retrieve(
            collection_name=config.COLLECTION, ids=point_ids, with_vectors=True, with_payload=False
        ):
            pv = point.vector
            if isinstance(pv, dict):  # named vectors
                pv = next(iter(pv.values()))
            if pv:
                # Both sides are L2-normalised, so dot product == cosine.
                dot = sum(a * b for a, b in zip(vec, pv))
                for cid in missing:
                    if index.articles[index.by_chunk_id[cid]].point_id == str(point.id):
                        cosine[cid] = dot

    contexts: list[Context] = []
    for rank, chunk_id in enumerate(ordered[:top_k], start=1):
        art = index.articles[index.by_chunk_id[chunk_id]]
        contexts.append(
            Context(
                rank=rank,
                chunk_id=chunk_id,
                doc_id=art.doc_id,
                article=art.article_display,
                text=art.text,
                retrieval_score=round(cosine.get(chunk_id, 0.0), 6),
                rerank_score=round(fused[chunk_id] / MAX_FUSED, 6),
                article_meta=art,
            )
        )

    debug = {
        "rewritten_query": " ".join(tokenize(question)),
        "detected_codes": codes,
        "retrieved_total": len(fused),
        "dense_candidates": len(dense),
        "lexical_candidates": len(lexical),
        "scoped_candidates": len(scoped),
    }
    return contexts, debug
