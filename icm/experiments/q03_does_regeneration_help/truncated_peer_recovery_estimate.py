"""ESTIMATE (CPU, cached): would TRUNCATED-peer regeneration give purer
recoveries than the shipped INTACT-peer regeneration?

Shipped (stage25, 37x1500): 1240 abstained -> 826 recovered, 754 correct
(91.3%). Those used INTACT peers, which stage28 shows induce copying
(S=0.728, no discrimination). Stage28 also scored TRUNCATED-peer
re-derivation S for candidate answers on the same contested pools.

Estimate: treat S(a) from stage28's trunc2 arm as the probability that a
truncated-peer regeneration produces answer a. With m=4 samples (the
shipped regen budget), P(a appears at least once) = 1 - (1-S)^4. Sweep a
presence threshold: recover a pool only if its best candidate's expected
appearance exceeds t. Report recovery volume vs precision, and the
resulting SFT-label purity, against the shipped operating point.

CAVEAT printed: S was measured by scoring candidate answers, not by free
sampling, so this is an upper-bound sketch of the frontier, not a
measurement. It answers 'is a GPU run worth it', nothing more.
"""
import json
from collections import defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from .regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
M = 4

S, per_pool = {}, defaultdict(dict)
for line in open(f"{RUN_DIR}/stage28_scores.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["arm"] != "trunc2":
        continue
    per_pool[r["iid"]][r["answer"]] = r["S"]

pools = load(ROOT)
world = World(pools)
state, contested, osc = converge(pools, world)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage25_regen.jsonl", encoding="utf-8")}
abst = [i for i in contested if state[i]["answer"] is None]

# shipped operating point, recomputed
ship_rec = ship_ok = 0
for iid in abst:
    ans = weighted_decide(pools[iid], world.context(state, iid),
                          regen.get(iid, []), 0.5, iid in osc)
    if ans is not None:
        ship_rec += 1
        ship_ok += match(ans, pools[iid]["gold"])
print(f"SHIPPED (intact peers): {ship_rec} recovered, {ship_ok} correct "
      f"({100*ship_ok/max(ship_rec,1):.1f}%)  of {len(abst)} abstained")
print("CAVEAT: S below was measured by scoring candidates, not free sampling;")
print("        treat as an upper-bound sketch of the frontier.\n")

scored_abst = [i for i in abst if per_pool.get(i)]
print(f"abstained pools with trunc2 scores: {len(scored_abst)}/{len(abst)}\n")
print(f"{'t':>6}{'recovered':>11}{'correct':>9}{'precision':>11}"
      f"{'vs shipped':>12}")
for t in (0.0, 0.3, 0.5, 0.7, 0.9):
    rec = ok = 0
    for iid in scored_abst:
        cands = per_pool[iid]
        best = max(cands, key=lambda a: (cands[a], a))
        p_appear = 1 - (1 - cands[best]) ** M
        if p_appear >= t:
            rec += 1
            ok += match(best, pools[iid]["gold"])
    prec = 100*ok/max(rec,1)
    print(f"{t:>6.1f}{rec:>11}{ok:>9}{prec:>10.1f}%{prec-100*ship_ok/max(ship_rec,1):>+11.1f}")

# SFT purity impact: recovered pools contribute ~4 regen chains each
print(f"\n--- SFT label purity impact (recovered pools only) ---")
print(f"shipped: {ship_ok}/{ship_rec} correct labels "
      f"-> {100*ship_ok/max(ship_rec,1):.1f}% pure")
for t in (0.5, 0.7, 0.9):
    rec = ok = 0
    for iid in scored_abst:
        cands = per_pool[iid]
        best = max(cands, key=lambda a: (cands[a], a))
        if 1 - (1 - cands[best]) ** M >= t:
            rec += 1
            ok += match(best, pools[iid]["gold"])
    lost = ship_ok - ok
    print(f"t={t}: {ok} correct labels ({100*ok/max(rec,1):.1f}% pure), "
          f"{lost:+d} correct labels vs shipped")
print("DONE")
