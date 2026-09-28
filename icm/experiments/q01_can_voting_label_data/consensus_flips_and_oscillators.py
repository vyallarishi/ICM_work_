"""Behind-the-scenes judge forensics (full pools, phase-A, exact CPU replay of w0):
 1. the lap-3 casualties: pools where pure consensus FLIPPED a correct answer
    to wrong as neighbors shifted (the churn the judge damped in the light arm)
 2. oscillators: pools whose answer changed 2+ times across laps
 3. per-lap judge-actionable windows (top-2 world gap within light's reach)
    and their fate under pure consensus"""
import json
from collections import Counter, defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .consensus_loop_15lang import initialize, update_pool, world_context

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
hist = {i: [state[i]["answer"]] for i in contested}
windows_per_lap = []
for lap in range(1, 5):
    # measure judge-actionable window BEFORE the update (light reach: 0.5/|jury|)
    win = []
    for iid in contested:
        others = world_context(pools, state, iid)
        counts = Counter(c["answer"] for c in pools[iid]["chains"])
        if not others or len(counts) < 2:
            continue
        shares = sorted((sum(match(o, a) for o in others) / len(others) for a in counts), reverse=True)
        if shares[0] - shares[1] <= 0.5 / max(len(others), 1) + 1e-9:
            win.append(iid)
    windows_per_lap.append(win)
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world_context(pools, state, iid), None, 0.0)
    state = ns
    for iid in contested:
        hist[iid].append(state[iid]["answer"])

print("=== 1. correct->wrong flips per lap (pure consensus churn) ===")
for lap in range(1, 5):
    flips = [i for i in contested
             if hist[i][lap-1] is not None and hist[i][lap] is not None
             and hist[i][lap] != hist[i][lap-1]
             and match(hist[i][lap-1], pools[i]["gold"])
             and not match(hist[i][lap], pools[i]["gold"])]
    if flips:
        langs = Counter(pools[i]["lang"] for i in flips)
        print(f"lap {lap}: {len(flips)} correct answers destroyed by consensus churn "
              f"({dict(langs)})")
        for i in flips[:4]:
            print(f"    {i}: had {hist[i][lap-1]} (gold={pools[i]['gold']}) -> flipped to {hist[i][lap]}")
    else:
        print(f"lap {lap}: none")

print()
print("=== 2. oscillators (answer changed 2+ times) ===")
osc = [i for i in contested
       if sum(1 for a, b in zip(hist[i], hist[i][1:]) if a != b) >= 2]
osc_final_wrong = [i for i in osc if hist[i][-1] is None or not match(hist[i][-1], pools[i]["gold"])]
print(f"oscillating pools: {len(osc)}; ended wrong/abstained: {len(osc_final_wrong)}")
for i in osc[:5]:
    print(f"    {i} ({pools[i]['lang']}): {' -> '.join(str(a) for a in hist[i])}  gold={pools[i]['gold']}")

print()
print("=== 3. judge-actionable window per lap, and its fate WITHOUT a judge ===")
for lap, win in enumerate(windows_per_lap, 1):
    wrong = sum(1 for i in win if hist[i][-1] is None or not match(hist[i][-1], pools[i]["gold"]))
    print(f"lap {lap}: {len(win)} near-tie pools within light-judge reach; "
          f"{wrong} of them ended wrong/abstained under pure consensus")
print("DONE")
