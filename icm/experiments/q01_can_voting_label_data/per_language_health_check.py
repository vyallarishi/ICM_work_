"""Pre-scale verification: per-language health of the recommended config's
CPU-computable core (vote-init + iterated consensus, full pools) on Phase-A.
Checks: convergence, per-language contested accuracy, NULL/veto rates,
determinism across replays."""
import json
from collections import Counter, defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match, TIER
from .consensus_loop_15lang import initialize, update_pool, world_context, metrics

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
def replay():
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(4):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_pool(pools[iid],
                                            world_context(pools, state, iid),
                                            None, 0.0)
        state = ns
    return state, contested

s1, contested = replay()
s2, _ = replay()
identical = all(s1[i]["answer"] == s2[i]["answer"] for i in s1)
print(f"determinism across replays: {identical}")
print(f"{'lang':<6}{'contested':>10}{'answered':>10}{'acc':>8}{'NULL':>6}{'sys prec':>10}")
for lang in sorted(TIER):
    ids = [i for i in contested if pools[i]["lang"] == lang]
    ans = [i for i in ids if s1[i]["answer"] is not None]
    hits = sum(match(s1[i]["answer"], pools[i]["gold"]) for i in ans)
    lang_all = [i for i in s1 if pools[i]["lang"] == lang]
    com = [i for i in lang_all if s1[i]["answer"] is not None]
    chits = sum(match(s1[i]["answer"], pools[i]["gold"]) for i in com)
    print(f"{lang:<6}{len(ids):>10}{len(ans):>10}{100*hits/max(len(ans),1):>7.1f}%"
          f"{len(ids)-len(ans):>6}{100*chits/max(len(com),1):>9.1f}%")
print(f"\noverall: {metrics(pools, s1, contested)}")
print("DONE")
