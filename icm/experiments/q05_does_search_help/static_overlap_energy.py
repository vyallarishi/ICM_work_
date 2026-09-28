"""Stage 29 — DOES A DERIVATION-COHERENCE ENERGY LANDSCAPE EXIST? (CPU)

WHY NOW: stage28 (n~11k, adversarial control, leak ablation) showed
re-derivation IS selective in aggregate -- S(settled)=0.48 vs S(wrong)=0.27,
gap +0.21, widening as leaks are removed -- but P1 (per-pool discrimination)
came in at 0.46, i.e. chance. Signal exists globally, not pool-locally.
That is the profile of a term in a GLOBAL OBJECTIVE, not a greedy rule.

THE PROPOSED ENERGY (all computable from cached chains, no model calls):
    E(L) = -SUM_problems [ SUM_{i<j} answer_agree(L_i, L_j)
                         + lam * SUM_{i<j} step_overlap(C_i, C_j) ]
where C_i is a chain in language i supporting label L_i, and step_overlap is
Jaccard over INTERMEDIATE VALUES (rung3 measured 0.86 both-right vs 0.27
one-wrong vs 0.03 null -- found, filed as a failed decision rule, never used
as an objective).

Annealing needs an energy whose landscape is not already flattened by the
greedy rule. So this probe answers, before any annealer is built:

 Q1 REDUNDANCY: is step_overlap merely a proxy for answer agreement?
    Measure step-overlap separation between chain pairs that AGREE on the
    answer but differ in correctness (both-right vs both-wrong-same-answer).
    If separation ~0 there, the term adds nothing the vote lacks. This is
    the make-or-break number: those pairs are exactly the correlated-error
    pools (25-30x lift) that answer-agreement cannot see BY CONSTRUCTION.
 Q2 QUESTION-NUMBER CONTAMINATION: rung3 did not exclude values copied from
    the problem statement. Recompute with and without them.
 Q3 LANDSCAPE: does E separate the converged labeling from (a) gold, (b)
    random label perturbations? If E(gold) < E(converged), a search has
    somewhere to go. If E(converged) is already minimal, annealing is moot.
 Q4 GRADIENT: among pools the loop got WRONG, would a single-flip step
    downhill in E move toward gold more often than chance?

PRE-REGISTERED: the energy is worth building an annealer on ONLY IF
Q1 separation >= 0.10 AND Q3 shows E(gold) < E(converged) AND Q4 > 55%.
Any failure -> report that derivation coherence is not an exploitable
objective at this scale and stop.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match, norm
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load

from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
LAMBDA = 1.0
SEED = 20260823


def values_in(text):
    """Normalized numeric values appearing in a chain."""
    out = set()
    for tok in NUMBER_RE.findall(text):
        v = norm(tok)
        if v is not None:
            out.add(v)
    return out


def intermediate_values(chain_text, question_values, final_answer):
    """Values a derivation PRODUCES: exclude numbers copied from the
    question (Q2) and the final answer itself (else this is answer
    agreement wearing a hat)."""
    vals = values_in(chain_text) - question_values
    return {v for v in vals if not match(v, final_answer)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- value extraction ---")
    check("parses and normalizes", values_in("48 / 2 = 24.0 and 1,000"),
          {"48", "2", "24", "1000"})
    print("--- intermediate values exclude question + answer ---")
    qv = {"48", "2"}
    check("keeps only derived, non-answer values",
          intermediate_values("48 / 2 = 24\n48 + 24 = 72\n#### 72", qv, "72"),
          {"24"})
    check("empty when everything is question or answer",
          intermediate_values("48 / 2 = 72", {"48", "2"}, "72"), set())
    print("--- jaccard ---")
    check("identical", jaccard({"1", "2"}, {"1", "2"}), 1.0)
    check("disjoint", jaccard({"1"}, {"2"}), 0.0)
    check("half", round(jaccard({"1", "2"}, {"2", "3"}), 4), 0.3333)
    check("empty guard", jaccard(set(), {"1"}), 0.0)
    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load(args.root)
    world = World(pools)
    state, contested, _ = converge(pools, world)
    rng = random.Random(SEED)
    results = {}

    qvals = {iid: values_in(p["question"]) for iid, p in pools.items()}
    # representative chain per (pool, answer)
    chain_for = {}
    for iid, pool in pools.items():
        for c in pool["chains"]:
            key = (iid, c["answer"])
            if key not in chain_for or len(c["text"]) < len(chain_for[key]["text"]):
                chain_for[key] = c
    steps = {}
    for (iid, ans), c in chain_for.items():
        steps[(iid, ans)] = intermediate_values(c["text"], qvals[iid], ans)

    by_problem = defaultdict(list)
    for iid in pools:
        by_problem[iid.rsplit("_", 1)[-1]].append(iid)

    # ---- Q1: does step-overlap separate right from wrong AMONG PAIRS THAT
    # AGREE ON THE ANSWER? (the region answer-voting is blind to)
    print("=== Q1: separation within answer-agreeing pairs ===")
    both_right, both_wrong_same, disagree = [], [], []
    sampled = 0
    for problem, iids in by_problem.items():
        gold = pools[iids[0]]["gold"]
        for a_i in range(len(iids)):
            for b_i in range(a_i + 1, len(iids)):
                i, j = iids[a_i], iids[b_i]
                ai, aj = state[i]["answer"], state[j]["answer"]
                if ai is None or aj is None:
                    continue
                si, sj = steps.get((i, ai)), steps.get((j, aj))
                if not si or not sj:
                    continue
                ov = jaccard(si, sj)
                sampled += 1
                if match(ai, aj):
                    (both_right if match(ai, gold) else both_wrong_same).append(ov)
                else:
                    disagree.append(ov)
    def mean(v):
        return sum(v) / len(v) if v else float("nan")
    sep = mean(both_right) - mean(both_wrong_same)
    print(f"  pairs sampled: {sampled}")
    print(f"  agree+both-right   n={len(both_right):>7} mean overlap {mean(both_right):.4f}")
    print(f"  agree+both-WRONG   n={len(both_wrong_same):>7} mean overlap {mean(both_wrong_same):.4f}")
    print(f"  disagree           n={len(disagree):>7} mean overlap {mean(disagree):.4f}")
    print(f"  Q1 SEPARATION (right - wrong, both agreeing) = {sep:+.4f}  "
          f"[bar 0.10] {'PASS' if sep >= 0.10 else 'FAIL'}")
    results["Q1"] = {"n_both_right": len(both_right),
                     "n_both_wrong_same": len(both_wrong_same),
                     "mean_both_right": round(mean(both_right), 4),
                     "mean_both_wrong_same": round(mean(both_wrong_same), 4),
                     "mean_disagree": round(mean(disagree), 4),
                     "separation": round(sep, 4),
                     "pass": bool(sep >= 0.10)}

    # ---- Q2: contamination from question numbers
    print("\n=== Q2: question-number contamination ===")
    steps_raw = {k: values_in(chain_for[k]["text"]) for k in chain_for}
    br_raw, bw_raw = [], []
    for problem, iids in by_problem.items():
        gold = pools[iids[0]]["gold"]
        for a_i in range(len(iids)):
            for b_i in range(a_i + 1, len(iids)):
                i, j = iids[a_i], iids[b_i]
                ai, aj = state[i]["answer"], state[j]["answer"]
                if ai is None or aj is None or not match(ai, aj):
                    continue
                si, sj = steps_raw.get((i, ai)), steps_raw.get((j, aj))
                if not si or not sj:
                    continue
                (br_raw if match(ai, gold) else bw_raw).append(jaccard(si, sj))
    sep_raw = mean(br_raw) - mean(bw_raw)
    print(f"  WITHOUT exclusions: right {mean(br_raw):.4f} vs wrong "
          f"{mean(bw_raw):.4f}  separation {sep_raw:+.4f}")
    print(f"  WITH exclusions:    right {mean(both_right):.4f} vs wrong "
          f"{mean(both_wrong_same):.4f}  separation {sep:+.4f}")
    results["Q2"] = {"separation_raw": round(sep_raw, 4),
                     "separation_clean": round(sep, 4)}

    # ---- Q3: landscape -- E(converged) vs E(gold) vs E(random perturbations)
    print("\n=== Q3: does the landscape have anywhere to go? ===")
    def energy(labels):
        e_ans = e_step = 0.0
        for problem, iids in by_problem.items():
            for a_i in range(len(iids)):
                for b_i in range(a_i + 1, len(iids)):
                    i, j = iids[a_i], iids[b_i]
                    ai, aj = labels.get(i), labels.get(j)
                    if ai is None or aj is None:
                        continue
                    if match(ai, aj):
                        e_ans += 1.0
                    si, sj = steps.get((i, ai)), steps.get((j, aj))
                    if si and sj:
                        e_step += jaccard(si, sj)
        return -(e_ans + LAMBDA * e_step), e_ans, e_step

    conv = {i: state[i]["answer"] for i in pools}
    e_conv, a_conv, s_conv = energy(conv)
    # gold labeling restricted to answers the language actually generated
    gold_lab = {}
    for iid, p in pools.items():
        has = any(match(c["answer"], p["gold"]) for c in p["chains"])
        gold_lab[iid] = p["gold"] if has else conv[iid]
    e_gold, a_gold, s_gold = energy(gold_lab)
    print(f"  E(converged) = {e_conv:.1f}  (answer {a_conv:.0f}, step {s_conv:.1f})")
    print(f"  E(gold*)     = {e_gold:.1f}  (answer {a_gold:.0f}, step {s_gold:.1f})")
    print(f"  gold is {'LOWER (search has somewhere to go)' if e_gold < e_conv else 'NOT lower (converged is already minimal)'}")
    perts = []
    for t in range(3):
        pert = dict(conv)
        flips = rng.sample(contested, min(200, len(contested)))
        for iid in flips:
            alts = [c["answer"] for c in pools[iid]["chains"]
                    if conv[iid] is None or not match(c["answer"], conv[iid])]
            if alts:
                pert[iid] = rng.choice(alts)
        perts.append(energy(pert)[0])
    print(f"  E(random 200-flip perturbation) = {mean(perts):.1f} "
          f"(higher than converged: {mean(perts) > e_conv})")
    results["Q3"] = {"E_converged": round(e_conv, 2), "E_gold": round(e_gold, 2),
                     "E_perturbed_mean": round(mean(perts), 2),
                     "gold_lower": bool(e_gold < e_conv),
                     "answer_term_converged": round(a_conv, 1),
                     "step_term_converged": round(s_conv, 1),
                     "answer_term_gold": round(a_gold, 1),
                     "step_term_gold": round(s_gold, 1)}

    # ---- Q4: would a downhill single flip move toward gold?
    print("\n=== Q4: single-flip gradient on pools the loop got WRONG ===")
    wrong_pools = [i for i in contested if conv[i] is not None
                   and not match(conv[i], pools[i]["gold"])
                   and any(match(c["answer"], pools[i]["gold"])
                           for c in pools[i]["chains"])]
    rng.shuffle(wrong_pools)
    sample = wrong_pools[:400]
    toward = away = 0
    for iid in sample:
        problem = iid.rsplit("_", 1)[-1]
        peers = [k for k in by_problem[problem] if k != iid]
        cur = conv[iid]
        cands = {c["answer"] for c in pools[iid]["chains"]}
        def local_e(ans):
            tot = 0.0
            for k in peers:
                ak = conv[k]
                if ak is None:
                    continue
                if match(ans, ak):
                    tot += 1.0
                si, sj = steps.get((iid, ans)), steps.get((k, ak))
                if si and sj:
                    tot += LAMBDA * jaccard(si, sj)
            return -tot
        best = min(cands, key=lambda a: (local_e(a), a))
        if match(best, pools[iid]["gold"]):
            toward += 1
        elif not match(best, cur):
            away += 1
    rate = toward / max(len(sample), 1)
    print(f"  wrong pools with gold available: {len(wrong_pools)} "
          f"(sampled {len(sample)})")
    print(f"  downhill flip lands on GOLD: {toward} ({100*rate:.1f}%)  "
          f"[bar 55%] {'PASS' if rate > 0.55 else 'FAIL'}")
    print(f"  flips to another wrong answer: {away}")
    results["Q4"] = {"n_wrong_with_gold": len(wrong_pools), "sampled": len(sample),
                     "flip_to_gold": toward, "rate": round(rate, 4),
                     "pass": bool(rate > 0.55)}

    verdict = (results["Q1"]["pass"] and results["Q3"]["gold_lower"]
               and results["Q4"]["pass"])
    results["VERDICT"] = ("BUILD THE ANNEALER" if verdict
                          else "NO EXPLOITABLE LANDSCAPE -- stop")
    print(f"\nVERDICT: {results['VERDICT']}")
    (OUT_DIR / f"stage29_results{args.tag}.json").write_text(
        json.dumps(results, indent=2))
    print("wrote stage29_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
