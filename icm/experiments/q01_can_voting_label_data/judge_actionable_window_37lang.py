"""Judge-actionable window at stage-17 scale (diverse-8, 37 langs, 36-jury)."""
from collections import Counter
from .consensus_loop_37lang import apply_diverse_views
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from .consensus_loop_15lang import initialize, update_pool, world_context
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
# stage17 already patched gate.TIER to 37 via import
pools = apply_diverse_views(gate.load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"))
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
for lap in range(1, 5):
    win = wrong_fate = 0
    for iid in contested:
        others = world_context(pools, state, iid)
        counts = Counter(c["answer"] for c in pools[iid]["chains"])
        if not others or len(counts) < 2:
            continue
        shares = sorted((sum(match(o, a) for o in others) / len(others)
                         for a in counts), reverse=True)
        if shares[0] - shares[1] <= 0.5 / max(len(others), 1) + 1e-9:
            win += 1
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world_context(pools, state, iid), None, 0.0)
    state = ns
    print(f"lap {lap}: judge-actionable near-tie pools = {win}  (of {len(contested)} contested)")
print("DONE")
