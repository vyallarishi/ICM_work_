"""Q09 step 1b (CPU): the gold-selected arm, the ICM-style ceiling.

`gold_filtered`: for the SAME (language, problem) pairs as the other arms, one of
the model's OWN chains whose answer equals the GOLD answer, chosen uniformly at
random with a per-pool seed. Same chains pool as `method`; only who picks the
label differs (gold instead of the vote). This is the comparison ICM made
(its labels vs gold labels on the same inputs), so self-distillation is not a
confound here:

    method - gold_filtered      how much the label-free labels lose to gold
    gold_filtered - translated  own chains with PERFECT labels vs translated gold

Where the pool has no gold-correct chain with an answer line (the vote cannot be
right there either), the `method` chain is kept, so every arm has the same pairs;
the count is printed and is part of the result.

    python -m icm.experiments.q09_does_the_dataset_beat_translation.build_gold_arm [--self-test]
"""
import argparse
import json
import random
import sys
from collections import Counter

from ...method import data
from ...paths import RUN_DIR
from .build_sft_sets import ROOT, SEED, completion_of

SOURCE_ARM = RUN_DIR / "q09_sft_method.jsonl"
OUT = RUN_DIR / "q09_sft_gold_filtered.jsonl"


def gold_chain(iid, chains, gold):
    """gold: the normalised gold string (data.norm)."""
    cands = [c for c in chains if data.norm(c["answer"]) is not None and data.norm(c["answer"]) == gold
             and completion_of(c["text"])]
    return random.Random(f"{SEED}-{iid}-gold").choice(cands) if cands else None


def self_test():
    ok = True
    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print(f"FAIL {name}: {got!r} vs {want!r}")
    C = lambda a, t: {"answer": a, "text": t}
    chains = [C("5", "a\n#### 5"), C("6", "b\n#### 6"), C("6", "no line")]
    check("picks a gold chain", gold_chain("p", chains, "6"), C("6", "b\n#### 6"))
    check("none correct -> None", gold_chain("p", chains, "7"), None)
    check("deterministic", gold_chain("q", chains * 2, "5"), gold_chain("q", chains * 2, "5"))
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if not self_test():
        sys.exit(1)
    if a.self_test:
        return
    full = data.load_pools(ROOT)
    rows = [json.loads(l) for l in open(SOURCE_ARM, encoding="utf-8")]
    st = Counter()
    with open(OUT, "w", encoding="utf-8") as h:
        for r in rows:
            gold = data.norm(full[r["id"]]["gold"])
            c = gold_chain(r["id"], full[r["id"]]["chains"], gold)
            if c is None:
                st["no_gold_chain_kept_method_chain"] += 1
                comp = r["completion"]
            else:
                comp = completion_of(c["text"])
                st["same_as_method_chain" if comp == r["completion"] else "different_chain"] += 1
            h.write(json.dumps({**r, "completion": comp}, ensure_ascii=False) + "\n")
    tiers = Counter(data.TIER[full[r["id"]]["lang"]] for r in rows)
    print(f"{len(rows)} pairs (same as every other arm); tiers {dict(tiers)}")
    print(json.dumps(dict(st), indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
