"""Abstention recovery via the 14B J3 judge (CPU, cached margins).

The consensus loop's final abstentions have forced-plurality accuracy 0%
(abstention_flow.log) -- consensus has NOTHING there. Route them to the
judge: among a pool's margin-scored chains, pick the argmax-margin chain's
answer; sweep a margin threshold for a coverage/accuracy frontier.

CAVEAT (printed): stage8 margins exist only for the <=4 chains/pool that the
gate sampled (up to 2 gold + 2 non-gold, class-balanced), so routing here is
a gold-ENRICHED preview: real pools of abstained items are mostly wrong
chains. Read the frontier shape, not the absolute level; final measurement
needs a full-pool scoring pass (stage18).
Margins used both raw and within-language centered (format_confound finding).
"""
import json, random, statistics as st
from collections import defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import SEED, build_demo_bank, gate_sample, load_pools, match
from .consensus_loop_15lang import initialize, update_pool, world_context
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(rows)
by_pool = defaultdict(list)
for s, r in zip(sample, rows):
    if r.get("J3") is not None:
        by_pool[s["instance_id"]].append({"answer": None, "text": s["text"],
                                          "is_gold": s["is_gold"], "J3": r["J3"],
                                          "lang": s["lang"]})
# recover each sampled chain's ANSWER by matching text within its pool
for iid, chains in by_pool.items():
    txt2ans = {c["text"]: c["answer"] for c in pools[iid]["chains"]}
    for c in chains:
        c["answer"] = txt2ans.get(c["text"])
# within-language centering of margins (label-blind)
lang_mu = defaultdict(list)
for chains in by_pool.values():
    for c in chains: lang_mu[c["lang"]].append(c["J3"])
lang_mu = {l: st.mean(v) for l, v in lang_mu.items()}
for chains in by_pool.values():
    for c in chains: c["J3c"] = c["J3"] - lang_mu[c["lang"]]

# replay w0 loop -> final abstentions (deterministic, judge-free)
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
for _ in range(4):
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world_context(pools, state, iid), None, 0.0)
    state = ns
abstained = [i for i in contested if state[i]["answer"] is None]
routable = [i for i in abstained if by_pool.get(i)]
print(f"final abstentions {len(abstained)}; with >=1 scored chain: {len(routable)}")
forced = sum(match(pools[i]["plurality"], pools[i]["gold"]) for i in abstained)
print(f"forced-plurality accuracy on all abstained: {100*forced/max(len(abstained),1):.1f}%")
print(f"CAVEAT: scored chains are the gate's class-balanced sample "
      f"(gold-enriched); absolute levels are optimistic upper bounds.\n")

for key, label in (("J3", "raw margins"), ("J3c", "language-centered margins")):
    picks = []
    for iid in routable:
        best = max(by_pool[iid], key=lambda c: c[key])
        picks.append((best[key], match(best["answer"], pools[iid]["gold"])))
    picks.sort(reverse=True)
    n_ok = sum(ok for _, ok in picks)
    print(f"--- routing by {label} ---")
    print(f"route ALL {len(picks)}: accuracy {100*n_ok/len(picks):.1f}%  "
          f"(recovered coverage +{len(picks)} pools at that precision)")
    print(f"{'keep top':>9}{'n':>5}{'acc':>8}")
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(frac*len(picks)))
        acc = sum(ok for _, ok in picks[:k]) / k
        print(f"{frac:>9.2f}{k:>5}{100*acc:>7.1f}%")
    print()
print("DONE")
