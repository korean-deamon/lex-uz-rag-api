"""
Build the 100-question test set in the format spec section 6 defines.

Composition is the spec's table, exactly:

  faktik              35   single-article factual questions
  raqamlar            15   a figure must be copied without error
  koʻp_modda          15   the answer needs more than one article or document
  amaldagi_tahrir     10   the current consolidated edition must be used
  kirill_rus          10   5 Russian + 5 Uzbek Cyrillic
  javobsiz            10   the corpus does not answer it -> no_answer expected
  noaniq               5   vague or colloquial -> query rewriting

Where the questions come from:
  * faktik / raqamlar are drawn from data/stress_test_100_uz.json, which already
    holds real questions tagged with the code that answers them.
  * the other 55 are written here. They are test *inputs*, so writing them is
    legitimate; what is not legitimate is inventing the ground truth, so see
    below.

What this script fills in, and what it deliberately leaves blank:
  expected_doc_id    filled. It is derived from each question's code tag via the
                     corpus manifest, so it is a real lex.uz document number,
                     not a guess. Hit@5 (document), latency, error share and
                     no_answer accuracy are all computable from this alone.
  expected_article   LEFT BLANK, except for the javobsiz rows where it is
                     correctly empty. Naming the article a question "should"
                     land on requires reading the statute; filling it from a
                     model's own output would make the eval grade the pipeline
                     against itself. Hit@5 (modda) and MRR stay unmeasured until
                     a human fills this column -- which is what the spec means by
                     "test toʻplami (tester tomonida)".
  expected_answer    LEFT BLANK for the same reason. Answer correctness is
                     scored by a human or an LLM judge against this column.

Run: python scripts/build_testset.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "stress_test_100_uz.json"
OUT = ROOT / "data" / "eval" / "testset_100.csv"
MANIFEST = ROOT / "data" / "corpus" / "uzbek-legal-corpus" / "manifest.csv"

TOTAL_BATCHES = 10
PER_BATCH = 10

# A question asking for a figure -- duration, amount, percentage, BHM multiple.
NUMERIC = re.compile(
    r"necha|qancha|nechta|stavka|foiz|miqdor|muddat|BHM|bazaviy|summa|yosh[i]?\b",
    re.IGNORECASE,
)

# --- authored questions -----------------------------------------------------
# (code tag, question). The code tag is the document the question is about; for
# javobsiz rows it is empty, because no document answers them.

KOP_MODDA = [
    ("jinoyat_kodeksi", "Jinoyat kodeksiga koʻra qasddan odam oʻldirish va ehtiyotsizlik oqibatida odam oʻlimiga sabab boʻlish uchun javobgarlik qanday farqlanadi?"),
    ("mehnat_kodeksi", "Mehnat kodeksiga koʻra mehnat shartnomasini xodim va ish beruvchi tashabbusi bilan bekor qilish tartibi qanday farqlanadi?"),
    ("soliq_kodeksi", "Soliq kodeksiga koʻra soliq toʻlovchining huquqlari va majburiyatlari nimalardan iborat?"),
    ("oila_kodeksi", "Oila kodeksiga koʻra nikohni bekor qilish va nikohni haqiqiy emas deb topish oʻrtasida qanday farq bor?"),
    ("fuqarolik_kodeksi_1qism", "Fuqarolik kodeksiga koʻra bitimning haqiqiy emasligi va bitimni bekor qilish qanday tartibda belgilanadi?"),
    ("yer_kodeksi", "Yer kodeksiga koʻra yer uchastkasini doimiy va muddatli egalik qilish huquqlari qanday farqlanadi?"),
    ("mamuriy_javobgarlik_kodeksi", "Maʼmuriy javobgarlik toʻgʻrisidagi kodeksga koʻra maʼmuriy jazo turlari va ularni qoʻllash tartibi qanday?"),
    ("jinoyat_protsessual_kodeksi", "Jinoyat-protsessual kodeksiga koʻra guvoh va jabrlanuvchining huquqlari qanday farqlanadi?"),
    ("konstitutsiya", "Konstitutsiyaga koʻra Oliy Majlis Qonunchilik palatasi va Senatning vakolatlari qanday taqsimlangan?"),
    ("budjet_kodeksi", "Budjet kodeksiga koʻra davlat budjeti daromadlari va xarajatlari qanday tarkibdan iborat?"),
    ("fuqarolik_protsessual_kodeksi", "Fuqarolik protsessual kodeksiga koʻra apellyatsiya va kassatsiya tartibida shikoyat qilish qanday farqlanadi?"),
    ("saylov_kodeksi", "Saylov kodeksiga koʻra Prezident va Oliy Majlis deputatlari saylovida nomzodlarga qoʻyiladigan talablar qanday farqlanadi?"),
    ("bojxona_kodeksi", "Bojxona kodeksiga koʻra tovarlarni vaqtincha olib kirish va erkin muomala uchun chiqarish rejimlari qanday farqlanadi?"),
    ("mehnat_kodeksi", "Mehnat kodeksiga koʻra har yilgi asosiy va qoʻshimcha taʼtillar qanday asoslarda beriladi?"),
    ("jinoyat_kodeksi", "Jinoyat kodeksiga koʻra oʻgʻrilik, talonchilik va bosqinchilik bir-biridan nimasi bilan farq qiladi?"),
]

AMALDAGI_TAHRIR = [
    ("mehnat_kodeksi", "Hozirda amalda boʻlgan Mehnat kodeksi qachon qabul qilingan va undan oldingi kodeks oʻrniga kelganmi?"),
    ("konstitutsiya", "Amaldagi tahrirdagi Konstitutsiyaga koʻra Prezident vakolat muddati necha yil?"),
    ("soliq_kodeksi", "Amaldagi Soliq kodeksiga koʻra aylanmadan olinadigan soliq toʻlovchilari kimlar?"),
    ("suv_kodeksi", "Amaldagi tahrirdagi Suv kodeksiga koʻra suvdan foydalanish huquqi qanday yuzaga keladi?"),
    ("yer_kodeksi", "Amaldagi Yer kodeksiga koʻra yer uchastkasini xususiy mulk qilib berish mumkinmi?"),
    ("raqobat_qonuni", "Amaldagi “Raqobat toʻgʻrisida”gi Qonunga koʻra ustun mavqe qanday aniqlanadi?"),
    ("markaziy_bank_qonuni", "Amaldagi Qonunga koʻra Markaziy bankning asosiy maqsadi nima?"),
    ("shaharsozlik_kodeksi", "Amaldagi Shaharsozlik kodeksiga koʻra shaharsozlik hujjatlari qanday turlarga boʻlinadi?"),
    ("oila_kodeksi", "Amaldagi Oila kodeksiga koʻra nikoh yoshi oʻzgartirilganmi va hozir necha yosh?"),
    ("jinoyat_kodeksi", "Amaldagi Jinoyat kodeksiga koʻra oʻlim jazosi saqlanganmi?"),
]

RUS_KIRILL = [
    # ru
    ("mehnat_kodeksi", "Какова минимальная продолжительность ежегодного основного трудового отпуска по Трудовому кодексу?", "ru"),
    ("fuqarolik_kodeksi_1qism", "Какой общий срок исковой давности установлен Гражданским кодексом Узбекистана?", "ru"),
    ("soliq_kodeksi", "Какая ставка налога на добавленную стоимость установлена Налоговым кодексом?", "ru"),
    ("konstitutsiya", "Какой язык является государственным согласно Конституции Республики Узбекистан?", "ru"),
    ("oila_kodeksi", "С какого возраста допускается заключение брака по Семейному кодексу?", "ru"),
    # uz-Cyrl
    ("jinoyat_kodeksi", "Жиноят кодексига кўра ўғрилик учун қанday жавобгарлик белгиланган?", "uz-Cyrl"),
    ("mehnat_kodeksi", "Меҳнат кодексига кўра иш вақтининг нормал давомийлиги қанча?", "uz-Cyrl"),
    ("konstitutsiya", "Конституцияга кўра Ўзбекистон Республикасининг пойтахти қайси шаҳар?", "uz-Cyrl"),
    ("soliq_kodeksi", "Солиқ кодексига кўра жисмоний шахслардан олинадиган даромад солиғи ставкаси қанча?", "uz-Cyrl"),
    ("oila_kodeksi", "Оила кодексига кўра никоҳ тузиш учун қандай шартлар талаб қилинади?", "uz-Cyrl"),
]

# Questions no document in this corpus answers. The expected result is
# no_answer: true -- these are what the hallucination metric is measured on.
JAVOBSIZ = [
    "Marsda yer uchastkasini qanday roʻyxatdan oʻtkazish kerak?",
    "Oʻzbekistonda kriptovalyuta mayningi uchun litsenziya Jinoyat kodeksining qaysi moddasida belgilangan?",
    "Qozogʻiston Respublikasi Mehnat kodeksiga koʻra taʼtil muddati qancha?",
    "Toshkentdagi eng yaxshi advokat kim?",
    "Oʻzbekiston Fuqarolik kodeksiga koʻra sunʼiy intellekt yaratgan asarning muallifi kim hisoblanadi?",
    "Yevropa Ittifoqi direktivasiga koʻra shaxsiy maʼlumotlarni himoya qilish tartibi qanday?",
    "Oʻzbekiston Respublikasi Konstitutsiyasining 300-moddasida nima deyilgan?",
    "Mehnat kodeksiga koʻra kosmonavtlarning ish vaqti qanday hisoblanadi?",
    "Soliq kodeksiga koʻra 2050 yilda QQS stavkasi qancha boʻladi?",
    "Oila kodeksiga koʻra uy hayvonini farzandlikka olish tartibi qanday?",
]

NOANIQ = [
    ("mehnat_kodeksi", "ishdan boʻshatishsa nima qilish kerak"),
    ("oila_kodeksi", "ajrashish haqida gapirib bering"),
    ("soliq_kodeksi", "soliq toʻlamasam nima boʻladi"),
    ("fuqarolik_kodeksi_1qism", "pul qarz berganimni qaytarmasa-chi"),
    ("yer_kodeksi", "yer olish uchun nima qilaman"),
]


def doc_ids_by_code() -> dict[str, str]:
    out: dict[str, str] = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["filename"].removesuffix(".txt")
            out[code] = f"lex:{row['lex_uz_doc'].lstrip('-')}"
    return out


def build_rows() -> list[dict]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    doc_ids = doc_ids_by_code()

    numeric = [q for q in source if NUMERIC.search(q["q"])]
    plain = [q for q in source if not NUMERIC.search(q["q"])]
    # Spread the picks across codes rather than taking the first N, which would
    # over-sample whichever code happens to come first in the file.
    numeric_pick = _spread(numeric, 15)
    taken = {q["q"] for q in numeric_pick}
    plain_pool = plain + [q for q in numeric if q["q"] not in taken]
    faktik_pick = _spread([q for q in plain_pool if q["q"] not in taken], 35)

    rows: list[dict] = []

    def add(category: str, code: str, question: str, language: str = "uz-Latn",
            answerable: bool = True):
        rows.append(
            {
                "category": category,
                "question": question,
                "language": language,
                "expected_doc_id": doc_ids.get(code, "") if answerable else "",
                "expected_article": "",
                "expected_answer": "",
                "answerable": "true" if answerable else "false",
            }
        )

    for q in faktik_pick:
        add("faktik", q["code"], q["q"])
    for q in numeric_pick:
        add("raqamlar", q["code"], q["q"])
    for code, question in KOP_MODDA:
        add("koʻp_modda", code, question)
    for code, question in AMALDAGI_TAHRIR:
        add("amaldagi_tahrir", code, question)
    for code, question, language in RUS_KIRILL:
        add("kirill_rus", code, question, language)
    for question in JAVOBSIZ:
        add("javobsiz", "", question, answerable=False)
    for code, question in NOANIQ:
        add("noaniq", code, question)

    assert len(rows) == TOTAL_BATCHES * PER_BATCH, f"expected 100 rows, built {len(rows)}"
    return _interleave_into_batches(rows)


def _spread(questions: list[dict], n: int) -> list[dict]:
    """Round-robin over codes so one code cannot dominate the sample."""
    by_code: dict[str, list[dict]] = {}
    for q in questions:
        by_code.setdefault(q["code"], []).append(q)
    picked: list[dict] = []
    while len(picked) < n:
        progressed = False
        for code in sorted(by_code):
            if by_code[code] and len(picked) < n:
                picked.append(by_code[code].pop(0))
                progressed = True
        if not progressed:
            break
    return picked[:n]


def _interleave_into_batches(rows: list[dict]) -> list[dict]:
    """Spec: 'each batch gets a mix of every category, so one batch is itself a
    small sample.' Dealing the categories round-robin into 10 batches does that
    and keeps the assignment reproducible."""
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    batches: list[list[dict]] = [[] for _ in range(TOTAL_BATCHES)]
    cursor = 0
    for category in sorted(by_category):
        for row in by_category[category]:
            # Find the next batch with room, starting where the last one landed.
            for _ in range(TOTAL_BATCHES):
                target = cursor % TOTAL_BATCHES
                cursor += 1
                if len(batches[target]) < PER_BATCH:
                    batches[target].append(row)
                    break

    ordered: list[dict] = []
    number = 1
    for index, batch in enumerate(batches, start=1):
        for row in batch:
            row["id"] = f"q{number:03d}"
            row["batch"] = f"b{index:02d}"
            number += 1
            ordered.append(row)
    return ordered


FIELDS = ["id", "batch", "category", "question", "language",
          "expected_answer", "expected_doc_id", "expected_article", "answerable"]


def main() -> None:
    rows = build_rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(f"wrote {len(rows)} questions -> {OUT}")
    for category, count in sorted(counts.items()):
        print(f"  {category:18s} {count}")
    print(f"  batches: {TOTAL_BATCHES} x {PER_BATCH}")


if __name__ == "__main__":
    main()
