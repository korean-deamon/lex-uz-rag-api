"""
Scraper for lex.uz (Uzbekistan legislation portal).

lex.uz renders document body text client-side via JS, so plain requests+BeautifulSoup
only sees an empty shell. We use Playwright (headless Chromium) to actually render
the page before pulling text out.

Output: one JSON file per scraped document under data/raw/, named by doc id, e.g:
  {
    "url": "...",
    "doc_id": "8563549",
    "title": "...",
    "text": "...",
    "content_hash": "sha256...",
    "scraped_at": "2026-10-30T12:00:00Z"
  }

State tracking: scraper/state.json maps doc_id -> content_hash of the last successful
scrape. On each run we only re-save (and mark for ingestion) docs that are new or whose
hash changed since last time -- this is what lets the daily Airflow job know "is there
new data or not".
"""
import json
import re
import hashlib
import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://lex.uz"
LISTING_URLS = [
    "https://lex.uz/uz/",   # homepage: "what's new" list of recently published docs
]
DOC_LINK_RE = re.compile(r"/uz/docs/-?\d+$")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
STATE_PATH = ROOT / "scraper" / "state.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def doc_id_from_url(url: str) -> str:
    m = re.search(r"docs/(-?\d+)", url)
    return m.group(1).lstrip("-") if m else hashlib.sha1(url.encode()).hexdigest()[:10]


def clean_text(raw: str) -> str:
    # drop the repeated "send a suggestion / listen to audio / get link" icon captions
    # that lex.uz interleaves between every paragraph
    noise = ["Hujjatga taklif yuborish", "Audioni tinglash", "Hujjat elementidan havola olish"]
    lines = [l.strip() for l in raw.splitlines()]
    lines = [l for l in lines if l and l not in noise]
    # collapse accidental duplicate blank-ish artifacts
    return "\n".join(lines)


def discover_doc_links(page, listing_url: str) -> set[str]:
    page.goto(listing_url, wait_until="networkidle", timeout=60000)
    hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.getAttribute('href'))")
    found = set()
    for h in hrefs:
        if not h:
            continue
        full = h if h.startswith("http") else BASE + h
        full = full.split("?")[0]
        if DOC_LINK_RE.search(full):
            found.add(full)
    return found


def scrape_doc(page, url: str) -> dict:
    page.goto(url, wait_until="networkidle", timeout=60000)
    title = page.title()
    body_text = page.eval_on_selector("body", "el => el.innerText")
    text = clean_text(body_text)
    doc_id = doc_id_from_url(url)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "url": url,
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "content_hash": content_hash,
        "scraped_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


def run(max_docs: int | None = None) -> list[dict]:
    """Scrape, compare against state, save only new/changed docs.
    Returns the list of doc dicts that were NEW or CHANGED this run."""
    state = load_state()
    changed_docs = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(user_agent="Mozilla/5.0 (LexRagBot/1.0)")

        all_links = set()
        for listing in LISTING_URLS:
            all_links |= discover_doc_links(page, listing)

        if max_docs:
            all_links = set(list(all_links)[:max_docs])

        print(f"[scraper] discovered {len(all_links)} candidate doc links")

        for url in sorted(all_links):
            try:
                doc = scrape_doc(page, url)
            except Exception as e:
                print(f"[scraper] FAILED {url}: {e}")
                continue

            prev_hash = state.get(doc["doc_id"])
            if prev_hash == doc["content_hash"]:
                continue  # unchanged, skip

            out_path = RAW_DIR / f"{doc['doc_id']}.json"
            out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
            state[doc["doc_id"]] = doc["content_hash"]
            changed_docs.append(doc)
            status = "NEW" if prev_hash is None else "CHANGED"
            print(f"[scraper] {status} {doc['doc_id']}: {doc['title'][:60]}")

        browser.close()

    save_state(state)
    print(f"[scraper] done. {len(changed_docs)} new/changed docs this run.")
    return changed_docs


if __name__ == "__main__":
    run()
