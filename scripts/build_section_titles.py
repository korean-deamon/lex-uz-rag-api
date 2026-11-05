"""
Recover the section (boʻlim) titles the corpus parser dropped.

The corpus's article records carry `section` as a bare marker -- "X BOʻLIM." --
with no title, because upstream captured the marker line and not the line after
it. In the raw text the title is right there:

    X BOʻLIM.
    QOʻSHILGAN QIYMAT SOLIGʻI          <- dropped
    31-bob. Soliq toʻlovchilar va soliq solish obyekti

That loss has a real cost. Thirteen articles of the Tax Code are titled "Soliq
stavkalari" and each states a rate without naming its tax: article 258 says
"soliq stavkasi 12 foiz miqdorida belgilanadi" and is the VAT rate *only*
because it sits under "QOʻSHILGAN QIYMAT SOLIGʻI". Asked for the VAT rate, the
pipeline retrieved article 258 at rank 1 and then correctly answered no_answer,
because nothing in the context said which tax the 12% was. The article was
found; the heading that gives it meaning was missing.

This script parses the raw .txt sources and writes the recovered titles to
data/section_titles.json, which retrieval.py loads to complete each article's
`section`.

Provenance note: the raw .txt sources are not committed (8.8 MB, and nothing at
runtime reads them) -- see data/corpus/CORPUS_SOURCE.md. The generated JSON is
committed instead, so a fresh clone needs neither the raw text nor this script.
To regenerate, fetch the corpus at the commit recorded in CORPUS_SOURCE.md so
the raw text matches the committed articles.

Run: python scripts/build_section_titles.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "corpus" / "uzbek-legal-corpus" / "data" / "raw"
OUT = ROOT / "data" / "section_titles.json"

# "X BOʻLIM." / "I BOB." style marker, exactly as the article records store it.
SECTION_MARKER = re.compile(r"^\s*([IVXL]+)\s+BO[ʻʼ'`’]LIM\.?\s*$", re.IGNORECASE)
# Lines that are structure, not a title.
CHAPTER_LINE = re.compile(r"^\s*\d+[¹²³⁴⁵⁶⁷⁸⁹]?\s*-\s*bob\.", re.IGNORECASE)
ARTICLE_LINE = re.compile(r"^\s*\d+[¹²³⁴⁵⁶⁷⁸⁹]?(\.\d+)?\s*-\s*modda\.", re.IGNORECASE)


def section_titles(text: str) -> dict[str, str]:
    """marker line -> recovered title, for one document."""
    lines = text.splitlines()
    found: dict[str, str] = {}

    for i, line in enumerate(lines):
        match = SECTION_MARKER.match(line)
        if not match:
            continue
        marker = line.strip()

        # The title is the next non-empty line, unless that line is already a
        # chapter or article heading -- some sections genuinely have no title.
        for candidate in lines[i + 1: i + 4]:
            candidate = candidate.strip()
            if not candidate:
                continue
            if CHAPTER_LINE.match(candidate) or ARTICLE_LINE.match(candidate):
                break
            # Titles are set in capitals in these documents; a lowercase line is
            # body text, which means this section has no heading of its own.
            letters = [c for c in candidate if c.isalpha()]
            if letters and sum(c.isupper() for c in letters) / len(letters) < 0.6:
                break
            found[marker] = candidate
            break

    return found


def main() -> int:
    if not RAW_DIR.exists():
        print(f"raw sources not found at {RAW_DIR}", file=sys.stderr)
        print("They are not committed on purpose -- see data/corpus/CORPUS_SOURCE.md.",
              file=sys.stderr)
        print(f"{OUT.name} is committed, so you only need this to regenerate it.",
              file=sys.stderr)
        return 1

    result: dict[str, dict[str, str]] = {}
    for path in sorted(RAW_DIR.glob("*.txt")):
        code = path.stem
        titles = section_titles(path.read_text(encoding="utf-8"))
        if titles:
            result[code] = titles

    OUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total = sum(len(v) for v in result.values())
    print(f"wrote {total} section titles across {len(result)} documents -> {OUT}")
    for code, titles in sorted(result.items()):
        print(f"  {code:42s} {len(titles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
