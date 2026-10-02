"""H6: translated gold solutions carry translation artefacts that native chains do not.

The cleanest artefact to measure without a judge: letters from a THIRD script,
i.e. neither the language's own script nor Latin (e.g. Korean "하였다" inside a
Burmese solution). Latin is excluded because variable names, units and English
loanwords are normal in maths text in both sets.

Same (language, problem) pairs in both sets: the q09 training files
(translated vs method, identical ids). Only languages with a non-Latin script.

PRE-REGISTERED: the share of solutions containing third-script letters is
higher in the translated set than in the method set in >= 2/3 of these
languages, and higher overall (pooled two-proportion z-test, p < 0.01).

    python -m icm.experiments.q10_pipeline_hypotheses.h6_translationese
"""
import json
import math
import sys
import unicodedata
from collections import defaultdict

from ...paths import RUN_DIR
from .common import write

# the script word(s) in unicodedata names that count as "own" per language
OWN = {"am": ("ETHIOPIC",), "ar": ("ARABIC",), "bg": ("CYRILLIC",), "bn": ("BENGALI",),
       "el": ("GREEK",), "fa": ("ARABIC",), "hi": ("DEVANAGARI",), "kk": ("CYRILLIC",),
       "kn": ("KANNADA",), "ko": ("HANGUL", "CJK"), "ky": ("CYRILLIC",), "ml": ("MALAYALAM",),
       "mr": ("DEVANAGARI",), "ru": ("CYRILLIC",), "th": ("THAI",), "uk": ("CYRILLIC",),
       "ur": ("ARABIC",), "zh-tw": ("CJK", "HALFWIDTH", "FULLWIDTH", "IDEOGRAPHIC")}


def third_script_letters(text, lang):
    """Letters whose script is neither the language's own nor Latin."""
    own = OWN[lang]
    out = []
    for ch in text:
        if not unicodedata.category(ch).startswith("L"):
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("LATIN") or any(name.startswith(s) for s in own):
            continue
        out.append(ch)
    return out


def self_test():
    ok = (third_script_letters("ছয়টি x = 5 하였다", "bn") == list("하였다")
          and third_script_letters("Привет, John 5+3", "ru") == []
          and third_script_letters("研究以下 x", "zh-tw") == [])
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed")
    sets = {}
    for arm in ("translated", "method"):
        sets[arm] = {json.loads(l)["id"]: json.loads(l) for l in open(RUN_DIR / f"q09_sft_{arm}.jsonl", encoding="utf-8")}
    assert sets["translated"].keys() == sets["method"].keys(), "arms are not paired"
    counts = defaultdict(lambda: {"n": 0, "translated": 0, "method": 0, "examples": []})
    for iid, row in sets["translated"].items():
        lang = row["lang"]
        if lang not in OWN:
            continue
        c = counts[lang]; c["n"] += 1
        for arm in ("translated", "method"):
            bad = third_script_letters(sets[arm][iid]["completion"], lang)
            if bad:
                c[arm] += 1
                if arm == "translated" and len(c["examples"]) < 2:
                    c["examples"].append({"id": iid, "letters": "".join(bad)[:20]})
    print(f"{'lang':<6}{'pairs':>7}{'translated %':>14}{'method %':>10}")
    wins = 0; T = M = N = 0
    for lang in sorted(counts):
        c = counts[lang]; t, m = c["translated"] / c["n"], c["method"] / c["n"]
        wins += t > m; T += c["translated"]; M += c["method"]; N += c["n"]
        print(f"{lang:<6}{c['n']:>7}{100 * t:>14.2f}{100 * m:>10.2f}   e.g. {c['examples'][:1]}")
    pt, pm = T / N, M / N; p = (T + M) / (2 * N)
    z = (pt - pm) / math.sqrt(max(2 * p * (1 - p) / N, 1e-12))
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    holds = wins >= math.ceil(2 / 3 * len(counts)) and pval < 0.01 and pt > pm
    print(f"\nALL   translated {100 * pt:.2f}%  method {100 * pm:.2f}%  z = {z:.1f}  one-sided p = {pval:.2g}; "
          f"translated worse in {wins}/{len(counts)} languages -> H6 {'HOLDS' if holds else 'FAILS'}")
    write("h6_translationese.json", {"per_language": {k: {kk: v for kk, v in c.items()} for k, c in counts.items()},
                                     "pooled": {"translated": pt, "method": pm, "z": z, "p": pval},
                                     "languages_translated_worse": wins, "H6_holds": bool(holds)})


if __name__ == "__main__":
    main()
