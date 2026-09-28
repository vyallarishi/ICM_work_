"""Autopsy of stage23's 5 failures: the 3 correct->wrong flips among
answered pools and the 2 wrong recoveries (both high-tier). For each:
pool composition, world majority, regen answers, oscillator status,
whether the minority peer existed. Characterize the failure mode."""
import json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context
from .regeneration_15lang import converge_w0, weighted_decide, answer_clusters
from ...paths import RUN_DIR

pools = s17.apply_diverse_views(s17.load_pools(
    "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"))
state, contested, osc = converge_w0(pools)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage23_regen37.jsonl", encoding="utf-8")}

new_state = {i: dict(s) for i, s in state.items()}
for iid in contested:
    others = world_context(pools, new_state, iid)
    new_state[iid]["answer"] = weighted_decide(
        pools[iid], others, regen.get(iid, []), 0.5, iid in osc)

def describe(iid, kind):
    p = pools[iid]
    others = world_context(pools, state, iid)
    top = Counter(others).most_common(3)
    counts = Counter(c["answer"] for c in p["chains"])
    problem = iid.rsplit("_", 1)[-1]
    maj, mino, _ = answer_clusters(pools, state, problem, p["lang"])
    print(f"\n=== {kind}: {iid} (tier {gate.TIER[p['lang']]}) gold={p['gold']}")
    print(f"  before={state[iid]['answer']}  after={new_state[iid]['answer']}"
          f"  oscillator={iid in osc}")
    print(f"  local pool: {dict(counts)}")
    print(f"  world top-3: {top}  | peer clusters: maj={maj} min={mino}")
    print(f"  regen answers: {regen.get(iid, [])}")
    print(f"  world majority correct: {match(top[0][0], p['gold']) if top else None}")

harmed = [i for i in contested if state[i]["answer"] is not None
          and match(state[i]["answer"], pools[i]["gold"])
          and (new_state[i]["answer"] is None
               or not match(new_state[i]["answer"], pools[i]["gold"]))]
wrong_rec = [i for i in contested if state[i]["answer"] is None
             and new_state[i]["answer"] is not None
             and not match(new_state[i]["answer"], pools[i]["gold"])]
print(f"harmed flips: {len(harmed)}; wrong recoveries: {len(wrong_rec)}")
for iid in harmed: describe(iid, "HARMED")
for iid in wrong_rec: describe(iid, "WRONG RECOVERY")
print("\nDONE")
