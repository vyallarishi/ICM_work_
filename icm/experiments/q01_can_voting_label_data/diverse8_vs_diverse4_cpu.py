"""Diverse-8 CPU evaluation: gold coverage, judge-free loop performance, and
the judge's winnable-window size, vs diverse-4 and full views. Zero GPU.
Diverse-8 = repo select_diverse logic at size 8: cover distinct answers
largest-group-first, then frequency-weighted fillers (so top answers get
redundant representation — the thing diverse-4 lacked)."""
import random, json
from collections import Counter, defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .consensus_loop_15lang import initialize, update_pool, world_context
from .why_judge_went_silent_on_diverse4 import window_pools, REPLAY_LAPS
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
CHAIN_MAX = 1800

def diverse_view(iid, chains, size):
    if len(chains) <= size:
        return list(chains)
    rng = random.Random((1729, iid, "diverse", size).__repr__())
    shuffled = list(chains); rng.shuffle(shuffled)
    groups = defaultdict(list)
    for c in shuffled: groups[c["answer"]].append(c)
    ordered = sorted(groups.values(), key=lambda g: -len(g))
    sel = [g[0] for g in ordered[:size]]
    if len(sel) < size:
        chosen = {id(c) for c in sel}
        for c in shuffled:
            if id(c) not in chosen:
                sel.append(c); chosen.add(id(c))
                if len(sel) == size: break
    return sel[:size]

pools = load_pools(ROOT)
print(f"pools {len(pools)}")
results = {}
for name, size in (("diverse4", 4), ("diverse8", 8), ("full", None)):
    shadow = {}
    for iid, p in pools.items():
        chains = [c for c in p["chains"] if len(c["text"]) <= CHAIN_MAX]
        if not chains: continue
        view = diverse_view(iid, chains, size) if size else chains
        counts = Counter(c["answer"] for c in view)
        shadow[iid] = dict(p, chains=view,
                           share=counts.most_common(1)[0][1] / len(view),
                           plurality=max(counts, key=lambda a: (counts[a], a)))
    gold_in = sum(any(match(c["answer"], p["gold"]) for c in p["chains"])
                  for p in shadow.values())
    state = initialize(shadow)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(REPLAY_LAPS):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_pool(shadow[iid],
                                            world_context(shadow, state, iid),
                                            None, 0.0)
        state = ns
    answered = [i for i in contested if state[i]["answer"] is not None]
    c_hits = sum(match(state[i]["answer"], shadow[i]["gold"]) for i in answered)
    committed = [i for i, s in state.items() if s["answer"] is not None]
    hits = sum(match(state[i]["answer"], shadow[i]["gold"]) for i in committed)
    window = window_pools(shadow, state, contested)
    winnable = sum(1 for w in window
                   if any(match(a, shadow[w["iid"]]["gold"]) for a in w["top2"]))
    judge_chains = sum(len(shadow[i]["chains"]) for i in contested)
    results[name] = {
        "gold_in_view": round(gold_in / len(shadow), 4),
        "contested": len(contested),
        "w0_contested_acc": round(c_hits / max(len(answered), 1), 4),
        "w0_coverage": round(len(committed) / len(shadow), 4),
        "w0_precision": round(hits / max(len(committed), 1), 4),
        "window": len(window), "winnable_window": winnable,
        "judge_chains_per_lap": judge_chains,
    }
    r = results[name]
    print(f"{name:<9} gold-in-view {100*r['gold_in_view']:.1f}%  contested {r['contested']}"
          f"  w0: {100*r['w0_contested_acc']:.1f}%/{100*r['w0_coverage']:.1f}%/{100*r['w0_precision']:.1f}%"
          f"  window {r['window']} (winnable {r['winnable_window']})"
          f"  judge chains/lap {r['judge_chains_per_lap']}")
json.dump(results, open(f"{RUN_DIR}/stage12_cpu_results.json","w"), indent=2)
print("DONE")
