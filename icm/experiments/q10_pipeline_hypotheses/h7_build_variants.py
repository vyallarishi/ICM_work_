"""H7 step 1 (CPU): a 37-language dataset of number-perturbed problems with derived answers.

Source: 2,000 English variants (probe_work_ q05 problem_variants.py): each
replaces some numbers in a GSM8K question and re-executes the solution's
<<a op b=c>> calculations to DERIVE the new answer. No human writes a label.

To get the same variant in every language WITHOUT a translation step, the same
number substitution is applied to the dataset's translated question. This is
only done when it is unambiguous: the translated question must contain exactly
the same ASCII-digit number tokens as the English one (same values, same
counts). Otherwise that (variant, language) is skipped and counted, never
guessed. Languages that write numbers as words or in native digits are
therefore under-represented, and the per-language skip rate is reported.

Variants are taken from source problems inside the 37 x 1,500 record (ids <
1,500), first N by source id, so H7 can compare precision on a variant with
precision on its own original problem.

Writes a dataset the generator reads like gsm8k-106:
    <OUT>/train/<lang>  (variants)   <OUT>/test/<lang> -> original test split (few-shot examples)
    <OUT>/offline_manifest.json
and runs/h7_variants_index.json.

    python -m icm.experiments.q10_pipeline_hypotheses.h7_build_variants [--self-test] [--n 500]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from ...method import data
from ...paths import RUN_DIR

SRC_DATASET = Path("/scratch/rvyalla/multilingual-icm/assets/datasets/gsm8k-106")
OUT = Path("/scratch/rvyalla/multilingual-icm/assets/datasets/gsm8k-106-variants500")
VARIANTS = Path("/home/rvyalla/multilingual_icm/data/probe_work_/runs/stage93_variants.jsonl")
NUM = re.compile(r"(?<![0-9.])[0-9]+(?:\.[0-9]+)?(?![0-9]|\.[0-9])")


def number_tokens(text):
    return Counter(NUM.findall(text))


def substitute(english, translated, mapping):
    """Apply mapping to `translated` iff its number tokens equal the English question's.
    Returns the new text or None (ambiguous: skip)."""
    en, tr = number_tokens(english), number_tokens(translated)
    if en != tr or not all(en[k] >= 1 for k in mapping):
        return None
    return NUM.sub(lambda m: mapping.get(m.group(0), m.group(0)), translated)


def self_test():
    ok = True
    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False; print(f"FAIL {name}: {got!r} vs {want!r}")
    en = "Natalia sold clips to 48 of her friends in April, and half as many in May."
    check("same numbers -> substituted", substitute(en, "Natalia a vendu 48 clips en avril.", {"48": "26"}),
          "Natalia a vendu 26 clips en avril.")
    check("number written differently -> skip", substitute("costs 1200 dollars", "kostet 1.200 Dollar", {"1200": "900"}), None)
    check("extra number -> skip", substitute(en, "48 clips, 2 mois", {"48": "26"}), None)
    check("native digits -> skip", substitute(en, "Natalia ৪৮ clips", {"48": "26"}), None)
    check("simultaneous swap", substitute("a 3 b 5", "x 3 y 5", {"3": "5", "5": "3"}), "x 5 y 3")
    check("decimal kept whole", substitute("rate 0.5 and 5", "taux 0.5 et 5", {"5": "7"}), "taux 0.5 et 7")
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed")
    from datasets import Dataset, load_from_disk
    if OUT.exists():
        sys.exit(f"{OUT} exists; refusing to overwrite")

    variants = [json.loads(l) for l in open(VARIANTS, encoding="utf-8")]
    variants = sorted([v for v in variants if v["source_original_id"] < 1500],
                      key=lambda v: v["source_original_id"])[:a.n]
    langs = sorted(data.TIER)
    english = {int(r["original_id"]): r for r in load_from_disk(str(SRC_DATASET / "train" / "en"))}
    index, skipped = {}, Counter()
    for lang in langs:
        rows_src = {int(r["original_id"]): r for r in load_from_disk(str(SRC_DATASET / "train" / lang))}
        out_rows = []
        for vi, v in enumerate(variants):
            src = rows_src[v["source_original_id"]]
            q = substitute(english[v["source_original_id"]]["question"], src["question"], v["mapping"])
            if q is None:
                skipped[lang] += 1
                continue
            out_rows.append({"id": f"var_{lang}_{vi}", "original_id": vi, "split": "train", "language": lang,
                             "language_name": src["language_name"], "question": q, "answer": "",
                             "answer_number": v["answer"], "original_question": v["question"],
                             "original_answer": "", "original_answer_number": v["answer"]})
            index.setdefault(vi, {"source_original_id": v["source_original_id"], "answer": v["answer"],
                                  "source_answer": v["source_answer"], "langs": []})["langs"].append(lang)
        Dataset.from_list(out_rows).save_to_disk(str(OUT / "train" / lang))
        print(f"{lang:<6} variants {len(out_rows):>4} / {len(variants)}  skipped {skipped[lang]}", flush=True)
    (OUT / "test").mkdir(parents=True)
    for lang in langs:
        os.symlink(SRC_DATASET / "test" / lang, OUT / "test" / lang)
    manifest = json.loads((SRC_DATASET / "offline_manifest.json").read_text())
    manifest.update({"dataset_id": "gsm8k-106-variants500", "dataset_path": str(OUT), "languages": langs})
    (OUT / "offline_manifest.json").write_text(json.dumps(manifest, indent=2))
    (RUN_DIR / "h7_variants_index.json").write_text(json.dumps(index, indent=1))
    total = sum(len(v["langs"]) for v in index.values())
    print(f"\nwrote {OUT}: {total} (variant, language) problems from {len(variants)} variants; "
          f"skip rate {100 * sum(skipped.values()) / (len(variants) * len(langs)):.1f}%")


if __name__ == "__main__":
    main()
