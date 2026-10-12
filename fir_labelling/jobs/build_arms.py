"""Training arms built from our own chains, for a language layer (core24 | all67), with the q09 rules
and seeds (functions imported unchanged from q09):

  own_unlabelled   one own chain, picked at random, no label                 (needs generation only)
  gold_filtered    one own chain whose answer equals the gold answer         (needs generation only)
  method           one own chain agreeing with the method's committed label  (needs labels)
  method_lenmatch  the method's label, the chain closest in length to the translated solution

Chains are picked from the FULL pool (every usable chain), as in q09; the label comes from the
method's diverse-8 vote (label_run.py). Every arm covers every (language, problem) pool for which it
has a usable completion (a chain with a '#### N' line); coverage per arm and language is printed and
written to results/arms_<layer>_coverage.json. The translated arm covers every pair.

    python build_arms.py --layer core24 --arms own_unlabelled gold_filtered
    python build_arms.py --layer core24 --arms method method_lenmatch
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/rvyalla/multilingual_icm/ICM_work_")
sys.path.insert(0, "/home/rvyalla/multilingual_icm/data/findings/jobs")
from icm.method import data  # noqa: E402
from icm.experiments.q09_does_the_dataset_beat_translation.build_sft_sets import (  # noqa: E402
    SEED, completion_of, pick_method_chain, prompt_of)
from icm.experiments.q09_does_the_dataset_beat_translation.build_gold_arm import gold_chain  # noqa: E402
from label_run import language_set, work_dir  # noqa: E402

F = Path("/home/rvyalla/multilingual_icm/data/findings")
LABEL_SET = {"core24": "core25", "all67": "all67"}


def own_chain(iid, chains):
    cands = [c for c in chains if completion_of(c["text"])]
    return random.Random(f"{SEED}-{iid}-own").choice(cands) if cands else None


def lenmatch_chain(iid, chains, committed, ref_len):
    cands = [c for c in chains if data.match(c["answer"], committed) and completion_of(c["text"])]
    if not cands:
        return None
    random.Random(f"{SEED}-{iid}-len").shuffle(cands)
    return min(cands, key=lambda c: abs(len(completion_of(c["text"])) - ref_len))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", required=True, choices=list(LABEL_SET))
    ap.add_argument("--arms", nargs="+", required=True,
                    choices=["own_unlabelled", "gold_filtered", "method", "method_lenmatch"])
    a = ap.parse_args()
    lset = LABEL_SET[a.layer]
    langs = set(language_set(lset))
    full = data.load_pools(str(work_dir(lset)), langs)
    assert {p["lang"] for p in full.values()} == langs
    out_dir = F / f"runs_{a.layer}"
    labels = None
    if {"method", "method_lenmatch"} & set(a.arms):
        labels = json.load(open(work_dir(lset) / "labels.json"))["labels"]
    ref_len = {}
    if "method_lenmatch" in a.arms:
        for line in open(out_dir / "sft_translated.jsonl", encoding="utf-8"):
            r = json.loads(line)
            ref_len[r["id"]] = len(r["completion"])

    cov_path = F / f"results/arms_{a.layer}_coverage.json"
    coverage = json.load(open(cov_path)) if cov_path.exists() else {}
    for arm in a.arms:
        st = defaultdict(Counter)
        with open(out_dir / f"sft_{arm}.jsonl", "w", encoding="utf-8") as h:
            for iid in sorted(full):
                p = full[iid]
                st[p["lang"]]["pools"] += 1
                if arm == "own_unlabelled":
                    c = own_chain(iid, p["chains"])
                elif arm == "gold_filtered":
                    c = gold_chain(iid, p["chains"], p["gold"])
                else:
                    lab = labels.get(iid)
                    if lab is None:
                        st[p["lang"]]["abstained"] += 1
                        continue
                    c = (pick_method_chain(iid, p["chains"], lab) if arm == "method"
                         else lenmatch_chain(iid, p["chains"], lab, ref_len.get(iid, 0)))
                if c is None:
                    st[p["lang"]]["no_usable_chain"] += 1
                    continue
                h.write(json.dumps({"id": iid, "lang": p["lang"], "prompt": prompt_of(p["question"]),
                                    "completion": completion_of(c["text"])}, ensure_ascii=False) + "\n")
                st[p["lang"]]["kept"] += 1
                st[p["lang"]]["correct"] += data.match(c["answer"], p["gold"])   # audit only, after selection
        tot = sum((v for v in st.values()), Counter())
        coverage[arm] = {l: dict(v) for l, v in sorted(st.items())}
        print(f"{arm}: {tot['kept']}/{tot['pools']} pools kept ({tot['kept']/tot['pools']:.4f}); "
              f"abstained {tot['abstained']}, no usable chain {tot['no_usable_chain']}; "
              f"training chains correct (audit) {tot['correct']/max(1, tot['kept']):.4f}", flush=True)
        low = sorted((v["kept"] / v["pools"], l) for l, v in st.items())[:5]
        print("   lowest coverage:", ", ".join(f"{l} {r:.3f}" for r, l in low))
    json.dump(coverage, open(cov_path, "w"), indent=1)


if __name__ == "__main__":
    main()
