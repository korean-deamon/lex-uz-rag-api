"""
RAG pipeline: embed the user's question (local model), search Qdrant for matching
chunks, hand them to Claude, return an answer grounded only in retrieved context.
"""
import os
import re
from dotenv import load_dotenv
from anthropic import Anthropic
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ingest.ingest import embed_model  # reuse the same local embedder as ingestion

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION = os.getenv("QDRANT_COLLECTION", "lex_uz")
CHAT_MODEL = "claude-haiku-4-5-20251001"
TOP_K = 14

# "Tree" layer: when the question names a specific kodeks/qonun, scope retrieval to
# that code first -- otherwise generic articles (crime definitions, punishment types,
# procedural boilerplate) out-compete the actually-relevant specific article purely on
# word overlap. Longer/more specific phrases must be checked before shorter ones
# (e.g. "jinoyat-protsessual" before "jinoyat").
CODE_KEYWORDS = [
    (r"jinoyat[\s-]?protsessual", "jinoyat_protsessual_kodeksi"),
    (r"jinoyat[\s-]?ijroiya", "jinoyat-ijroiya_kodeksi"),
    (r"jinoyat kodeks", "jinoyat_kodeksi"),
    (r"fuqarolik protsessual", "fuqarolik_protsessual_kodeksi"),
    (r"ma\u02bbmuriy javobgarlik", "mamuriy_javobgarlik_kodeksi"),
    (r"ma\u02bbmuriy sud", "mamuriy_sud_ishlarni_yuritish_kodeksi"),
    (r"iqtisodiy protsessual", "iqtisodiy_protsessual_kodeksi"),
    (r"soliq kodeks", "soliq_kodeksi"),
    (r"mehnat kodeks", "mehnat_kodeksi"),
    (r"oila kodeks", "oila_kodeksi"),
    (r"yer kodeks", "yer_kodeksi"),
    (r"suv kodeks", "suv_kodeksi"),
    (r"havo kodeks", "havo_kodeksi"),
    (r"bojxona kodeks", "bojxona_kodeksi"),
    (r"budjet kodeks", "budjet_kodeksi"),
    (r"saylov kodeks", "saylov_kodeksi"),
    (r"shaharsozlik kodeks", "shaharsozlik_kodeksi"),
    (r"konstitutsiya", "konstitutsiya"),
    (r"markaziy bank", "markaziy_bank_qonuni"),
    (r"raqobat", "raqobat_qonuni"),
    (r"investitsiya", "investitsiyalar_faoliyati_qonuni"),
    (r"qimmatli qog[o\u02bb]zlar", "qimmatli_qogozlar_bozori_qonuni"),
    (r"tadbirkorlik faoliyati", "tadbirkorlik_faoliyati_qonuni"),
    # Fuqarolik kodeksi is split 1qism/2qism in the corpus -- ambiguous on its own,
    # intentionally not scoped, both parts stay in the unfiltered pool.
]

_anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Uzbek stopwords / boilerplate that shows up in nearly every question -- stripped out
# before keyword matching so we're left with the actual legal term being asked about
# (e.g. "oʻgʻrilik", "poraxoʻrlik") instead of generic words like "kodeksiga".
_STOPWORDS = {
    "kodeks", "kodeksi", "kodeksiga", "kodeksga", "qonun", "qonunga", "qonuniga",
    "toʻgʻrisidagi", "togrisidagi", "koʻra", "kora", "uchun", "qanday", "qancha",
    "necha", "bor", "bormi", "mumkin", "mumkinmi", "qilinadi", "belgilangan",
    "nazarda", "tutilgan", "hisoblanadi", "beriladi", "qoʻyiladi", "qoyiladi",
    "amalga", "oshiriladi", "qilish", "qilinishi", "kim", "nima", "deb", "bilan",
    "yoki", "va", "ham", "bu", "shu", "u", "biz", "siz",
    # code-name roots and generic legal filler -- too common within their own code to
    # discriminate between articles (e.g. "jinoyat" matches ~every article title in
    # jinoyat_kodeksi, drowning out the actually distinctive term like "oʻgʻrilik")
    "jinoyat", "javobgarlik", "jazo", "protsessual", "ijroiya", "fuqarolik",
    "mehnat", "soliq", "oila", "budjet", "bojxona", "saylov", "shaharsozlik",
    "markaziy", "investitsiya", "qimmatli", "qogozlar", "qogʻozlar", "bozori",
    "tadbirkorlik", "raqobat", "konstitutsiya", "maʼmuriy", "mamuriy", "iqtisodiy",
    "yillik", "umumiy", "aloqador", "tegishli",
}

_code_articles_cache: dict = {}

# common folk/compound legal terms -> the actual words the statute text uses
_SYNONYMS = {
    "poraxoʻrlik": ["pora"],
    "poraxorlik": ["pora"],
    "korrupsiya": ["pora", "mansab"],
    "korrupsiyaga": ["pora", "mansab"],
}


def _load_code_articles(code: str) -> list[dict]:
    """Fetch all articles for one code once, cached, used for keyword matching."""
    if code in _code_articles_cache:
        return _code_articles_cache[code]
    points, _ = _qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=qmodels.Filter(
            must=[qmodels.FieldCondition(key="code", match=qmodels.MatchValue(value=code))]
        ),
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    articles = [p.payload for p in points]
    _code_articles_cache[code] = articles
    return articles


def extract_keywords(query: str) -> list[str]:
    words = re.findall(r"[a-zA-Z\u02bb\u02bc\u0100-\u017f]+", query.lower())
    keywords = [w for w in words if len(w) >= 5 and w not in _STOPWORDS]
    # expand a few common folk/compound terms to the words the statute actually uses
    expanded = []
    for w in keywords:
        expanded.append(w)
        expanded.extend(_SYNONYMS.get(w, []))
    return expanded


def keyword_matches(code: str, keywords: list[str], limit: int = 5) -> list[dict]:
    """Exact-substring keyword search over one code's articles -- catches cases the
    embedding model misses because of generic-word overlap (e.g. a question about
    'oʻgʻrilik' losing to generic 'javobgarlik' articles on pure vector similarity)."""
    if not keywords:
        return []
    articles = _load_code_articles(code)
    scored = []
    for a in articles:
        title = (a.get("title") or "").lower()
        text = (a.get("text") or "").lower()
        score = sum(3 for k in keywords if k in title) + sum(2 for k in keywords if k in text)
        if score > 0:
            scored.append((score, a))
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "score": 1.0,  # keyword hits are treated as max-confidence, listed first
            "text": a.get("text", ""),
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "doc_id": None,
            "code": a.get("code"),
            "code_title": a.get("code_title"),
            "article_display": a.get("article_display"),
        }
        for _, a in scored[:limit]
    ]

SYSTEM_PROMPT = (
    "You are a legal information assistant for Uzbekistan legislation, grounded in "
    "lex.uz and the Uzbek Legal Corpus (Konstitutsiya + 20 kodekslar + 4 qonun). "
    "Answer ONLY using the CONTEXT provided below. If the context does not contain "
    "enough information to answer, say so plainly instead of guessing -- never invent "
    "an article number or a code name that isn't in the context. "
    "Every answer MUST explicitly cite, for each fact used: the exact kodeks/qonun name "
    "(code_title) AND the exact modda (article_display) it came from -- e.g. "
    "'(Soliq kodeksi, 480\u00b9-modda)'. If multiple articles/codes are relevant, cite each "
    "one next to the fact it supports. Respond in the same language the user asked in."
)


def detect_code(query: str) -> str | None:
    q = query.lower()
    for pattern, code in CODE_KEYWORDS:
        if re.search(pattern, q):
            return code
    return None


def embed_query(text: str) -> list[float]:
    # e5 models need a "query: " prefix on search queries (vs "passage: " for indexed text)
    vec = embed_model().encode([f"query: {text}"], normalize_embeddings=True)[0]
    return vec.tolist()


def _hits_to_dicts(hits) -> list[dict]:
    return [
        {
            "score": h.score,
            "text": h.payload.get("text", ""),
            "title": h.payload.get("title", ""),
            "url": h.payload.get("url", ""),
            "doc_id": h.payload.get("doc_id", ""),
            "code": h.payload.get("code"),
            "code_title": h.payload.get("code_title"),
            "article_display": h.payload.get("article_display"),
        }
        for h in hits
    ]


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    vec = embed_query(query)
    code = detect_code(query)

    if code:
        keyword_hits = keyword_matches(code, extract_keywords(query), limit=8)

        scoped = _qdrant.query_points(
            collection_name=COLLECTION,
            query=vec,
            limit=top_k,
            query_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="code", match=qmodels.MatchValue(value=code))]
            ),
        ).points
        vector_hits = _hits_to_dicts(scoped)

        # merge: keyword hits first (exact term match = high confidence), then vector
        # hits, deduped by article_display so we don't repeat the same modda twice
        seen = set()
        merged = []
        for item in keyword_hits + vector_hits:
            key = (item.get("code"), item.get("article_display"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        if merged:
            return merged[:top_k]
        # fall through to unfiltered search if the code had no matches at all

    hits = _qdrant.query_points(collection_name=COLLECTION, query=vec, limit=top_k).points
    return _hits_to_dicts(hits)


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        if c.get("code_title"):
            tag = f"[Kodeks: {c['code_title']} | Modda: {c.get('article_display','')} | {c['url']}]"
        else:
            tag = f"[Source: {c['title']} | {c['url']}]"
        parts.append(f"{tag}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer(question: str) -> dict:
    chunks = retrieve(question)
    if not chunks:
        return {
            "answer": "I don't have any indexed data yet to answer that from. Run the scraper/ingest pipeline first.",
            "sources": [],
        }

    context = build_context_block(chunks)
    resp = _anthropic.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
        ],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")

    sources = [{"title": c["title"], "url": c["url"]} for c in chunks]
    seen = set()
    uniq_sources = []
    for s in sources:
        if s["url"] not in seen:
            seen.add(s["url"])
            uniq_sources.append(s)

    return {"answer": text, "sources": uniq_sources}
