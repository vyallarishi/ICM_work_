"""Is 0% routed accuracy a bug or structural? Count abstained pools that
contain ANY gold-matching chain (full pool, not just sampled)."""
import json, random
from ..q02_can_the_model_judge_itself.verdict_judge_gate import SEED, build_demo_bank, gate_sample, load_pools, match
from .consensus_loop_15lang import initialize, update_pool, world_context
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
for _ in range(4):
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world_context(pools, state, iid), None, 0.0)
    state = ns
abstained = [i for i in contested if state[i]["answer"] is None]
with_gold = [i for i in abstained
             if any(match(c["answer"], pools[i]["gold"]) for c in pools[i]["chains"])]
print(f"abstained: {len(abstained)}; containing >=1 gold chain in FULL pool: {len(with_gold)}")
for i in with_gold[:10]:
    n_gold = sum(match(c["answer"], pools[i]["gold"]) for c in pools[i]["chains"])
    print(f"  {i}: {n_gold}/{len(pools[i]['chains'])} gold chains, plurality={pools[i]['plurality']}, gold={pools[i]['gold']}")
# sanity on the join: how many sampled abstained chains had answer recovered?
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
miss = tot = gold_sampled = 0
for s, r in zip(sample, rows):
    if s["instance_id"] in set(abstained) and r.get("J3") is not None:
        tot += 1
        txt2ans = {c["text"]: c["answer"] for c in pools[s["instance_id"]]["chains"]}
        if txt2ans.get(s["text"]) is None: miss += 1
        if s["is_gold"]: gold_sampled += 1
print(f"sampled chains in abstained pools: {tot}; answer-join misses: {miss}; is_gold among them: {gold_sampled}")
