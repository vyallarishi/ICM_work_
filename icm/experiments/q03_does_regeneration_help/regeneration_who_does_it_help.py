"""What fraction of all pools (a) reach generative MP, (b) are CHANGED by it,
(c) are HELPED by it? Full accounting on the 1500-problem run."""
import json
from collections import Counter, defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from .regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

pools = load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
world = World(pools)
state, contested, osc = converge(pools, world)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage25_regen.jsonl", encoding="utf-8")}
final = {i: dict(s) for i, s in state.items()}
for iid in contested:
    final[iid]["answer"] = weighted_decide(pools[iid], world.context(final, iid),
                                           regen.get(iid, []), 0.5, iid in osc)

N = len(pools)
committed_pools = [i for i in pools if i not in set(contested)]
print(f"TOTAL POOLS: {N}")
print(f"  never contested (landslide-committed, regen never runs): "
      f"{len(committed_pools)} ({100*len(committed_pools)/N:.1f}%)")
print(f"  reached generative MP (contested or abstained):          "
      f"{len(contested)} ({100*len(contested)/N:.1f}%)")

changed = [i for i in contested if final[i]["answer"] != state[i]["answer"]]
helped, hurt, neutral = [], [], []
for i in changed:
    was = state[i]["answer"] is not None and match(state[i]["answer"], pools[i]["gold"])
    now = final[i]["answer"] is not None and match(final[i]["answer"], pools[i]["gold"])
    (helped if (now and not was) else hurt if (was and not now) else neutral).append(i)

print(f"\nOF ALL {N} POOLS:")
print(f"  changed by generative MP: {len(changed)} ({100*len(changed)/N:.2f}%)")
print(f"    HELPED (wrong/abstain -> correct): {len(helped)} ({100*len(helped)/N:.2f}%)")
print(f"    HURT   (correct -> wrong/abstain): {len(hurt)} ({100*len(hurt)/N:.2f}%)")
print(f"    neutral (wrong -> wrong, etc):     {len(neutral)} ({100*len(neutral)/N:.2f}%)")
print(f"  net correct labels added: {len(helped)-len(hurt)} "
      f"({100*(len(helped)-len(hurt))/N:.2f}% of all pools)")

print(f"\nOF THE {len(contested)} POOLS THAT REACHED GENERATIVE MP:")
print(f"  changed: {100*len(changed)/len(contested):.1f}%   "
      f"helped: {100*len(helped)/len(contested):.1f}%   "
      f"hurt: {100*len(hurt)/len(contested):.1f}%")
print(f"  unchanged (regen ran, made no difference): "
      f"{len(contested)-len(changed)} ({100*(len(contested)-len(changed))/len(contested):.1f}%)")

# breakdown by entry state
abst = [i for i in contested if state[i]["answer"] is None]
ans = [i for i in contested if state[i]["answer"] is not None]
a_help = [i for i in helped if i in set(abst)]
n_help = [i for i in helped if i in set(ans)]
print(f"\nBY ENTRY STATE:")
print(f"  entered ABSTAINED: {len(abst)} pools -> helped {len(a_help)} "
      f"({100*len(a_help)/max(len(abst),1):.1f}% of them)")
print(f"  entered ANSWERED:  {len(ans)} pools -> helped {len(n_help)} "
      f"({100*len(n_help)/max(len(ans),1):.1f}% of them)")

print(f"\nHELPED POOLS BY TIER (as % of that tier's total pools):")
tier_tot = Counter(gate.TIER[pools[i]["lang"]] for i in pools)
tier_help = Counter(gate.TIER[pools[i]["lang"]] for i in helped)
for t in ("high", "mid", "low"):
    print(f"  {t:<5} {tier_help[t]:>4} helped / {tier_tot[t]:>6} pools "
          f"= {100*tier_help[t]/tier_tot[t]:.2f}%")

print(f"\nHELPED POOLS BY LANGUAGE (top 12):")
lang_help = Counter(pools[i]["lang"] for i in helped)
lang_tot = Counter(pools[i]["lang"] for i in pools)
for lang, n in lang_help.most_common(12):
    print(f"  {lang:<7} {n:>4} / {lang_tot[lang]} = {100*n/lang_tot[lang]:.2f}%")
print("DONE")
