import sys, copy
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import Counter, defaultdict
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g
from ..q01_can_voting_label_data import consensus_loop_15lang as s9
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
pools=g.load_pools(ROOT)
state=s9.initialize(pools)
contested=[i for i,s in state.items() if s["kind"]=="contested"]
print("pools",len(pools),"contested",len(contested))

# How often does the veto fire, and how often would a match()-based membership
# test have given a DIFFERENT verdict than the `in counts` test?
fires=0; would_differ=0; examples=[]
for iid in contested:
    others=s9.world_context(pools,state,iid)
    if not others: continue
    counts=Counter(c["answer"] for c in pools[iid]["chains"])
    top_world, top_n = Counter(others).most_common(1)[0]
    if top_n/len(others) < 0.70: continue
    exact_absent = top_world not in counts
    match_absent = not any(match(top_world,a) for a in counts)
    if exact_absent: fires+=1
    if exact_absent != match_absent:
        would_differ+=1
        if len(examples)<8:
            examples.append((iid, pools[iid]["lang"], top_world, sorted(counts)))
print(f"veto fires (exact `in counts`): {fires}")
print(f"pools where match()-membership disagrees with `in counts`: {would_differ}")
for e in examples: print("   ",e)

print("\n=== also: initialize() veto uses match() (line 66) but update_pool uses `in` (line 95) ===")
print("  -> the SAME veto concept is implemented two different ways in one file.")

# Similarly for initialize's landslide veto
fires2=0
world=defaultdict(dict)
for iid,p in pools.items(): world[iid.rsplit("_",1)[-1]][p["lang"]]=p["plurality"]
for iid,p in pools.items():
    others=[a for l,a in world[iid.rsplit("_",1)[-1]].items() if l!=p["lang"]]
    if not others: continue
    top=Counter(others).most_common(1)
    if top[0][1]/len(others)>=0.70 and p["share"]>=0.70 and not match(top[0][0],p["plurality"]):
        fires2+=1
print(f"  initialize() landslide-veto fires on {fires2} pools")

print("\n=== ALSO: score() world_share double-counts via match() tolerance ===")
print("  ws(a) = sum(match(o,a) for o in others)/len(others). With tolerance 0.005,")
print("  two DIFFERENT local answers can both match the same world answer -> shares sum > 1.")
pool={"chains":[{"answer":"100"},{"answer":"100.4"}]}
others=["100.2"]*10
for a in ("100","100.4"):
    print(f"    ws({a}) =", sum(match(o,a) for o in others)/len(others))
print("    -> both 1.0; tie broken by epsilon/alpha, not by the world. Reachable when a pool")
print("       contains two answers within 0.5% of each other.")
near=0
for iid,p in pools.items():
    ks=sorted(Counter(c['answer'] for c in p['chains']))
    for i in range(len(ks)):
        for j in range(i+1,len(ks)):
            if match(ks[i],ks[j]): near+=1; break
        else: continue
        break
print(f"    real phaseA pools containing >=2 distinct answer strings that match() each other: {near}")
