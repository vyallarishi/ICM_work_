"""Stage 30 — IS RE-DERIVATION AN EXPLOITABLE ENERGY? (CPU, cached scores)

Stage 29 killed energies built from STATIC text overlap: E(gold) was HIGHER
than E(converged) (Q3) and the downhill single flip found gold 0.9% of the
time (Q4). Structural, not tunable: consensus energies are minimised by
agreement, and the residual errors ARE agreement.

Stage 28 measured a DIFFERENT quantity on ~11k pools:
    S(a) = fraction of m=8 samples re-deriving a from truncated peers
    S(settled)=0.483 vs S(wrong)=0.266, gap +0.217 (leak-free +0.213),
    gap WIDENS as leak vectors are removed.
That gradient points toward correctness where static overlap did not. This
probe re-runs stage29's two fatal tests against the re-derivation energy,
using only cached stage28 scores (trunc2 arm, leak-free preferred).

    E(L) = - SUM_pools S(L_i | truncated peers supporting L_i)

R1 (=Q3): E(gold*) < E(converged)?  If not, dead the same way.
R2 (=Q4): on pools the loop got WRONG where gold is available AND scored,
     does argmax_a S(a) land on gold?  Stage28 P1=0.46 (per-pool discrimination
     at chance) predicts failure; this measures it directly on the wrong-pool
     subset, which is the only place a search would need to act.
R3: coverage sanity -- how many pools even have >=2 scored candidates
     (a search cannot move where only one answer is scored).
R4: calibration of the margin S(top1)-S(top2) as an abstention signal, the
     fallback use if R1/R2 fail.

PRE-REGISTERED: build an annealer ONLY IF R1 passes AND R2 > 0.55.
Otherwise report re-derivation as an aggregate-only signal, unusable as a
search objective, and close the annealing line with evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load

from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
SCORES = OUT_DIR / "stage28_scores.jsonl"
ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
ARM = "trunc2"          # lowest leak rate (11.8%), widest stage28 gap


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- energy arithmetic ---")
    S = {("p1", "5"): 0.8, ("p1", "40"): 0.2, ("p2", "7"): 0.5}
    def E(labels):
        return -sum(S.get((i, a), 0.0) for i, a in labels.items())
    check("lower energy = better re-derivation",
          E({"p1": "5"}) < E({"p1": "40"}), True)
    check("missing score contributes 0", E({"p9": "1"}), 0.0)
    check("sums across pools", round(E({"p1": "5", "p2": "7"}), 4), -1.3)

    print("--- argmax over scored candidates ---")
    per_pool = {"p1": {"5": 0.8, "40": 0.2}}
    best = max(per_pool["p1"], key=lambda a: (per_pool["p1"][a], a))
    check("picks higher S", best, "5")
    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--arm", default=ARM)
    parser.add_argument("--leak-free", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    S, leaked = {}, {}
    for line in SCORES.open(encoding="utf-8"):
        r = json.loads(line)
        if r["arm"] != args.arm:
            continue
        if args.leak_free and r["leaked"]:
            continue
        S[(r["iid"], r["answer"])] = r["S"]
        leaked[(r["iid"], r["answer"])] = r["leaked"]
    per_pool = defaultdict(dict)
    for (iid, ans), s in S.items():
        per_pool[iid][ans] = s
    print(f"arm={args.arm} leak_free={args.leak_free}: "
          f"{len(S)} scored (pool, answer) pairs over {len(per_pool)} pools")

    pools = load(ROOT)
    world = World(pools)
    state, contested, _ = converge(pools, world)
    conv = {i: state[i]["answer"] for i in pools}
    results = {"arm": args.arm, "leak_free": bool(args.leak_free),
               "n_scored_pairs": len(S), "n_pools_scored": len(per_pool)}

    # ---- R3 coverage first: a search can only move where >=2 candidates scored
    multi = {i: d for i, d in per_pool.items() if len(d) >= 2}
    print(f"\n=== R3 coverage ===")
    print(f"  pools with >=2 scored candidates: {len(multi)} "
          f"({100*len(multi)/max(len(per_pool),1):.1f}% of scored pools, "
          f"{100*len(multi)/max(len(contested),1):.1f}% of contested)")
    results["R3"] = {"pools_multi_candidate": len(multi),
                     "pct_of_scored": round(100*len(multi)/max(len(per_pool),1), 2)}

    # ---- R1: E(gold*) vs E(converged), restricted to scored pools
    def energy(labels):
        return -sum(S.get((i, a), 0.0) for i, a in labels.items()
                    if a is not None)
    scored_ids = list(per_pool)
    conv_lab = {i: conv[i] for i in scored_ids}
    gold_lab = {}
    for i in scored_ids:
        g = pools[i]["gold"]
        has_chain = any(match(c["answer"], g) for c in pools[i]["chains"])
        gold_lab[i] = g if has_chain else conv[i]
    e_conv, e_gold = energy(conv_lab), energy(gold_lab)
    # how many gold labels are even scored (else E is blind to them)
    gold_scored = sum(1 for i in scored_ids
                      if (i, gold_lab[i]) in S)
    print(f"\n=== R1 landscape ===")
    print(f"  E(converged) = {e_conv:.2f}")
    print(f"  E(gold*)     = {e_gold:.2f}")
    print(f"  gold labels with a score: {gold_scored}/{len(scored_ids)}")
    print(f"  R1 {'PASS (gold is lower)' if e_gold < e_conv else 'FAIL (converged already lower)'}")
    results["R1"] = {"E_converged": round(e_conv, 2), "E_gold": round(e_gold, 2),
                     "gold_labels_scored": gold_scored,
                     "pass": bool(e_gold < e_conv)}

    # ---- R2: on WRONG pools, does argmax S land on gold?
    wrong = [i for i in multi
             if conv[i] is not None and not match(conv[i], pools[i]["gold"])
             and any(match(a, pools[i]["gold"]) for a in per_pool[i])]
    hit = 0
    for i in wrong:
        best = max(per_pool[i], key=lambda a: (per_pool[i][a], a))
        hit += match(best, pools[i]["gold"])
    rate = hit / max(len(wrong), 1)
    print(f"\n=== R2 gradient on wrong pools ===")
    print(f"  wrong pools with gold among scored candidates: {len(wrong)}")
    print(f"  argmax S lands on gold: {hit} ({100*rate:.1f}%)  [bar 55%] "
          f"{'PASS' if rate > 0.55 else 'FAIL'}")
    results["R2"] = {"n_wrong_with_gold_scored": len(wrong), "hits": hit,
                     "rate": round(rate, 4), "pass": bool(rate > 0.55)}
    # control: how often does argmax S land on gold among ALL multi pools
    allhit = alltot = 0
    for i in multi:
        if not any(match(a, pools[i]["gold"]) for a in per_pool[i]):
            continue
        alltot += 1
        best = max(per_pool[i], key=lambda a: (per_pool[i][a], a))
        allhit += match(best, pools[i]["gold"])
    print(f"  (control) argmax S = gold across ALL multi-candidate pools "
          f"where gold is scored: {allhit}/{alltot} "
          f"= {100*allhit/max(alltot,1):.1f}%")
    results["R2_control_all_pools"] = {"hits": allhit, "total": alltot,
                                       "rate": round(allhit/max(alltot,1), 4)}

    # ---- R4: margin as an abstention/confidence signal (fallback use)
    pos, neg, rows = [], [], []
    for i in multi:
        ranked = sorted(per_pool[i].items(), key=lambda kv: -kv[1])
        margin = ranked[0][1] - ranked[1][1]
        ok = conv[i] is not None and match(conv[i], pools[i]["gold"])
        (pos if ok else neg).append(margin)
        rows.append((margin, ok))
    a = auc(pos, neg)
    rows.sort(reverse=True)
    print(f"\n=== R4 margin as confidence ===")
    print(f"  AUC(margin separates correct vs wrong commits) = "
          f"{a if a else float('nan'):.4f}  (n_correct {len(pos)}, n_wrong {len(neg)})")
    print(f"  {'thresh':>8}{'kept':>8}{'coverage':>10}{'accuracy':>10}")
    for frac in (1.0, 0.9, 0.75, 0.5):
        k = max(1, int(frac * len(rows)))
        acc = sum(ok for _, ok in rows[:k]) / k
        print(f"{frac:>8.2f}{k:>8}{frac:>10.2f}{100*acc:>9.1f}%")
    results["R4"] = {"auc": round(a, 4) if a else None,
                     "n_correct": len(pos), "n_wrong": len(neg)}

    verdict = ("BUILD THE ANNEALER" if (results["R1"]["pass"]
               and results["R2"]["pass"]) else
               "RE-DERIVATION IS AGGREGATE-ONLY -- not a search objective")
    results["VERDICT"] = verdict
    print(f"\nVERDICT: {verdict}")
    tag = f"_{args.arm}{'_leakfree' if args.leak_free else ''}"
    (OUT_DIR / f"stage30_results{tag}.json").write_text(json.dumps(results, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
