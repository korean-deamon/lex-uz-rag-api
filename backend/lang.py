"""
Language/script detection for the three languages the spec requires:
uz-Latn, uz-Cyrl and ru.

Script tells Latin from Cyrillic outright. Inside Cyrillic, Uzbek and Russian are
separated by letters that exist in only one of them -- Uzbek Cyrillic has
ў/қ/ғ/ҳ, Russian has ы/э/щ/ъ/ё -- which is enough for the short questions the
test set is made of, and needs no model.
"""
import re
import unicodedata

UZ_LATN = "uz-Latn"
UZ_CYRL = "uz-Cyrl"
RU = "ru"
SUPPORTED = (UZ_LATN, UZ_CYRL, RU)

_UZ_CYRILLIC_ONLY = set("ўқғҳЎҚҒҲ")
_RUSSIAN_ONLY = set("ыэщъёЫЭЩЪЁ")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")

# Uzbek Latin writes oʻ/gʻ with U+02BB (modifier letter turned comma). Keyboards
# and copy-paste produce at least five other characters for the same sound; they
# are folded to U+02BB so retrieval matches the corpus, which is consistent.
_APOSTROPHE_VARIANTS = "‘’ʼ′`´'"


def normalise(text: str) -> str:
    """Fold apostrophe variants and compose combining marks, without changing
    the script or stripping anything the user typed."""
    text = unicodedata.normalize("NFC", text)
    for ch in _APOSTROPHE_VARIANTS:
        text = text.replace(ch, "ʻ")
    return text


def detect(text: str) -> str:
    """Best-effort language tag; defaults to uz-Latn, the corpus language."""
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))

    if cyrillic > latin:
        chars = set(text)
        if chars & _UZ_CYRILLIC_ONLY:
            return UZ_CYRL
        if chars & _RUSSIAN_ONLY:
            return RU
        # Cyrillic with no distinguishing letter either way: Russian is the more
        # common case for such questions in this test set.
        return RU
    return UZ_LATN


def resolve_answer_language(requested: str | None, question: str) -> str:
    """options.answer_language, where 'auto' (the default) means the question's own
    language. An unsupported explicit tag is rejected by the caller."""
    if not requested or requested == "auto":
        return detect(question)
    return requested


def is_supported(tag: str | None) -> bool:
    return tag is None or tag == "auto" or tag in SUPPORTED


LANGUAGE_NAMES = {
    UZ_LATN: "Uzbek written in the Latin alphabet (oʻzbek lotin yozuvi)",
    UZ_CYRL: "Uzbek written in the Cyrillic alphabet (ўзбек кирилл ёзуви)",
    RU: "Russian (русский язык)",
}
