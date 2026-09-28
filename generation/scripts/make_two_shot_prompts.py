"""Generate a two-example variant of the few_shot_cot prompt templates.

The localized few_shot_cot headers say "study the six solved examples"; the
2-shot runtime truncation shows only two. This script writes
scripts/prompts/few_shot_cot_2shot.jsonl with the example-count wording REMOVED (count-agnostic) for
the languages listed in REPLACEMENTS (phrase-level, because numerals govern
noun case/agreement in several languages). All other languages are copied
unchanged and reported, so the fix's coverage is always explicit.

Usage:  python scripts/make_two_shot_prompts.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).with_name("prompts")
SOURCE = PROMPTS_DIR / "few_shot_cot.jsonl"
TARGET = PROMPTS_DIR / "few_shot_cot_2shot.jsonl"

# Exact phrase -> grammatically correct two-example phrase.
# Phrase-level on purpose: e.g. Russian "шесть решённых примеров" (gen. pl.)
# becomes "два решённых примера" (gen. sg.); Arabic switches to the dual;
# Swahili needs mi-class agreement (sita -> miwili).
REPLACEMENTS: dict[str, tuple[str, str]] = {
    # ---- original 20 (count removed; grammar repaired where the numeral governed it)
    "en": ("Study the six solved examples below.",
           "Study the solved examples below."),
    "cs": ("Prostudujte si šest níže vyřešených příkladů.",
           "Prostudujte si níže vyřešené příklady."),
    "af": ("Bestudeer die ses opgeloste voorbeelde hieronder.",
           "Bestudeer die opgeloste voorbeelde hieronder."),
    "hi": ("नीचे दिए गए छह हल किए गए उदाहरणों का अध्ययन करें।",
           "नीचे दिए गए हल किए गए उदाहरणों का अध्ययन करें।"),
    "am": ("ከታች ያሉትን ስድስት የተፈቱ ምሳሌዎች አጥና።",
           "ከታች ያሉትን የተፈቱ ምሳሌዎች አጥና።"),
    "vi": ("Hãy nghiên cứu sáu ví dụ đã được giải dưới đây.",
           "Hãy nghiên cứu các ví dụ đã được giải dưới đây."),
    "de": ("Studiere die sechs gelösten Beispiele unten.",
           "Studiere die gelösten Beispiele unten."),
    "zh-tw": ("研究以下六個已解答的範例。",
              "研究以下已解答的範例。"),
    "tr": ("Aşağıdaki altı çözülmüş örneği inceleyin.",
           "Aşağıdaki çözülmüş örnekleri inceleyin."),
    "el": ("Μελέτησε τα έξι λυμένα παραδείγματα παρακάτω.",
           "Μελέτησε τα λυμένα παραδείγματα παρακάτω."),
    "th": ("ศึกษาตัวอย่างที่แก้แล้วทั้งหกข้อด้านล่าง",
           "ศึกษาตัวอย่างที่แก้แล้วด้านล่าง"),
    "bn": ("নিচের ছয়টি সমাধান করা উদাহরণ অধ্যয়ন করুন।",
           "নিচের সমাধান করা উদাহরণগুলি অধ্যয়ন করুন।"),
    "ru": ("Изучите шесть решённых примеров ниже.",
           "Изучите решённые примеры ниже."),
    "ar": ("ادرس الأمثلة الستة المحلولة أدناه.",
           "ادرس الأمثلة المحلولة أدناه."),
    "ko": ("아래의 풀이된 6개의 예시를 학습하세요.",
           "아래의 풀이된 예시를 학습하세요."),
    "fa": ("شش مثال حل‌شده زیر را مطالعه کنید.",
           "مثال‌های حل‌شده زیر را مطالعه کنید."),
    "kn": ("ಕೆಳಗಿನ ಆರು ಪರಿಹರಿಸಿದ ಉದಾಹರಣೆಗಳನ್ನು ಅಧ್ಯಯನ ಮಾಡಿ.",
           "ಕೆಳಗಿನ ಪರಿಹರಿಸಿದ ಉದಾಹರಣೆಗಳನ್ನು ಅಧ್ಯಯನ ಮಾಡಿ."),
    "sw": ("Soma mifano sita iliyotatuliwa hapa chini.",
           "Soma mifano iliyotatuliwa hapa chini."),
    "eu": ("Aztertu beheko sei ebatzitako adibideak.",
           "Aztertu beheko ebatzitako adibideak."),
    "uk": ("Вивчіть шість розв'язаних прикладів нижче.",
           "Вивчіть розв'язані приклади нижче."),
    # ---- 17 scale-up languages
    "es": ("Estudia los seis ejemplos resueltos a continuación.",
           "Estudia los ejemplos resueltos a continuación."),
    "fr": ("Étudiez les six exemples résolus ci-dessous.",
           "Étudiez les exemples résolus ci-dessous."),
    "pt": ("Estude os seis exemplos resolvidos abaixo.",
           "Estude os exemplos resolvidos abaixo."),
    "it": ("Studia i sei esempi risolti qui sotto.",
           "Studia gli esempi risolti qui sotto."),
    "nl": ("Bestudeer de zes opgeloste voorbeelden hieronder.",
           "Bestudeer de opgeloste voorbeelden hieronder."),
    "pl": ("Przeanalizuj sześć rozwiązanych przykładów poniżej.",
           "Przeanalizuj rozwiązane przykłady poniżej."),
    "bg": ("Изучете шестте решени примера по-долу.",
           "Изучете решените примери по-долу."),
    "id": ("Pelajari enam contoh terselesaikan di bawah ini.",
           "Pelajari contoh-contoh terselesaikan di bawah ini."),
    "ms": ("Teliti enam contoh yang telah diselesaikan di bawah.",
           "Teliti contoh-contoh yang telah diselesaikan di bawah."),
    "ur": ("ذیل میں دی گئی چھ حل شدہ مثالوں کا مطالعہ کریں۔",
           "ذیل میں دی گئی حل شدہ مثالوں کا مطالعہ کریں۔"),
    "mr": ("खालील सहा सोडवलेल्या उदाहरणांचा अभ्यास करा.",
           "खालील सोडवलेल्या उदाहरणांचा अभ्यास करा."),
    "kk": ("Төмендегі алты шығарылған мысалды зерттеңіз.",
           "Төмендегі шығарылған мысалдарды зерттеңіз."),
    "az": ("Aşağıdakı altı həll edilmiş nümunəni öyrənin.",
           "Aşağıdakı həll edilmiş nümunələri öyrənin."),
    "ml": ("താഴെയുള്ള പരിഹരിച്ച ആറ് ഉദാഹരണങ്ങൾ പഠിക്കുക.",
           "താഴെയുള്ള പരിഹരിച്ച ഉദാഹരണങ്ങൾ പഠിക്കുക."),
    "is": ("Skoðaðu sex leystu dæmin hér að neðan.",
           "Skoðaðu leystu dæmin hér að neðan."),
    "ky": ("Төмөндөгү алты чыгарылган мисалды изилде.",
           "Төмөндөгү чыгарылган мисалдарды изилде."),
    "ceb": ("Tun-i ang unom ka nasulbad nga mga pananglitan sa ubos.",
            "Tun-i ang nasulbad nga mga pananglitan sa ubos."),
}


def main() -> None:
    fixed: list[str] = []
    unchanged: list[str] = []
    out_rows: list[dict] = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        code = row["language_code"]
        if code in REPLACEMENTS:
            old, new = REPLACEMENTS[code]
            count = row["prompt"].count(old)
            if count != 1:
                raise SystemExit(
                    f"{code}: expected exactly 1 occurrence of the source "
                    f"phrase, found {count} — refusing to write anything."
                )
            row["prompt"] = row["prompt"].replace(old, new)
            fixed.append(code)
        else:
            unchanged.append(code)
        row["prompt_name"] = "few_shot_cot_2shot"
        out_rows.append(row)

    with TARGET.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {TARGET} ({len(out_rows)} languages)")
    print(f"count wording fixed ({len(fixed)}): {' '.join(sorted(fixed))}")
    print(
        f"NOT fixed, copied verbatim ({len(unchanged)}): still say 'six' — "
        "extend REPLACEMENTS before using these languages in 2-shot runs."
    )


if __name__ == "__main__":
    main()
