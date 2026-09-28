"""Why was the sample insensitive to an rng perturbation? Investigate
gate_sample's actual dependence on rng."""
import random, sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import gate_sample, CHAIN_MAX_CHARS, match
from collections import Counter

# gate_sample shuffles gold_chains and rest, then takes [:2] of each.
# It is rng-SENSITIVE only when a pool has >2 eligible gold chains or
# >2 eligible non-gold chains. Otherwise [:2] takes everything and the
# shuffle is irrelevant to WHICH chains, though it still consumes rng.
def make_pools(ngold, nrest, npools=50):
    pools={}
    for p in range(npools):
        iid=f"train_en_{p:04d}"
        chains=[{"answer":"42","text":f"g{p}_{k}"} for k in range(ngold)]
        chains+=[{"answer":"17","text":f"r{p}_{k}"} for k in range(nrest)]
        counts=Counter(c["answer"] for c in chains)
        pools[iid]={"gold":"42","lang":"en","question":"Q","chains":chains,
                    "share":counts.most_common(1)[0][1]/len(chains),
                    "plurality":max(counts,key=lambda a:(counts[a],a))}
    return pools

for (ng,nr) in [(2,2),(3,3),(5,5),(8,4)]:
    pools=make_pools(ng,nr)
    a=[s["text"] for s in gate_sample(pools, random.Random(1))]
    b=[s["text"] for s in gate_sample(pools, random.Random(2))]
    print(f"gold={ng} rest={nr}: different rng seed -> different chains? {a!=b}")

# Confirm on REAL data shape: how many pools have >2 eligible chains per side?
print("\nNow the real question: in the actual phaseA pools, is gate_sample")
print("rng-sensitive? (needs the real data; check if present)")
import os, glob
root="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
print("phaseA root exists:", os.path.isdir(root))
print("pooled files:", len(glob.glob(f"{root}/final/pooled/generated/*.jsonl")))
