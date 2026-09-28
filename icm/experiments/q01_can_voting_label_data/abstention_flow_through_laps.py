"""Track individual pools through the loop (w0, deterministic, full pools):
who starts abstained (vetoed), who recovers, who abstains later, final state."""
from collections import Counter
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .consensus_loop_15lang import initialize, update_pool, world_context

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
vetoed_init = {i for i in contested if state[i]["answer"] is None}
history = {i: [state[i]["answer"]] for i in contested}
for _ in range(4):
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world_context(pools, state, iid), None, 0.0)
    state = ns
    for iid in contested:
        history[iid].append(state[iid]["answer"])

rec = [i for i in vetoed_init if state[i]["answer"] is not None]
rec_ok = sum(match(state[i]["answer"], pools[i]["gold"]) for i in rec)
print(f"vetoed at init: {len(vetoed_init)}")
print(f"  recovered by the loop: {len(rec)}  ({rec_ok} correct = {100*rec_ok/max(len(rec),1):.0f}%)")
print(f"  still abstained at end: {len(vetoed_init)-len(rec)}")

transient = [i for i in contested if i not in vetoed_init
             and any(a is None for a in history[i][1:-1]) and state[i]["answer"] is not None]
t_ok = sum(match(state[i]["answer"], pools[i]["gold"]) for i in transient)
print(f"abstained mid-loop then resolved: {len(transient)} ({t_ok} correct)")

new_null = [i for i in contested if i not in vetoed_init and state[i]["answer"] is None]
forced = sum(match(pools[i]["plurality"], pools[i]["gold"]) for i in new_null)
print(f"newly abstained by loop's end: {len(new_null)} "
      f"(their local plurality would be right only {100*forced/max(len(new_null),1):.0f}% -> good refusals)")
final_null = [i for i in contested if state[i]["answer"] is None]
fn = sum(match(pools[i]["plurality"], pools[i]["gold"]) for i in final_null)
print(f"final abstentions: {len(final_null)}; forced-plurality accuracy on them {100*fn/max(len(final_null),1):.0f}%")
print("DONE")
