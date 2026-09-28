# Graduated escalation: 4 chains -> 8 -> 16, only for unresolved pools.
# Nested (incremental) sampling: one seeded shuffle per pool; prefixes model
# "generate 4 more" honestly. Veto active at every level.
import sys, random
from collections import Counter, defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS, SEED
from .stage6_user_spec_config import WORLD_STRENGTH, LANDSLIDE

pools, gold = load()
rng = random.Random((SEED, "ladder").__repr__())
perm = {iid: rng.sample([a for _, a in ch], len(ch)) for iid, ch in pools.items()}

def decide(sub_pools):
    plur = defaultdict(dict)
    for iid, a in sub_pools.items():
        if a:
            c = Counter(a)
            plur[iid.rsplit("_",1)[-1]]["_".join(iid.split("_")[1:-1])] = max(c, key=lambda k:(c[k],k))
    out = {}
    for iid, a in sub_pools.items():
        if not a: out[iid] = None; continue
        c = Counter(a); L = "_".join(iid.split("_")[1:-1]); P = iid.rsplit("_",1)[-1]
        others = [p for l,p in plur[P].items() if l != L]
        w = Counter(others).most_common(1) if others else []
        strong = w and w[0][1]/len(others) >= WORLD_STRENGTH
        if strong and w[0][0] in c:
            out[iid] = w[0][0]
        elif c.most_common(1)[0][1]/len(a) >= LANDSLIDE:
            top = max(c, key=lambda k:(c[k],k))
            out[iid] = None if (strong and not match(w[0][0], top)) else top
        else:
            out[iid] = None
    return out

def stats(final, tier=None):
    com=cor=tot=0
    for iid,a in final.items():
        g=gold.get(iid)
        if g is None: continue
        if tier and TIER["_".join(iid.split("_")[1:-1])]!=tier: continue
        tot+=1
        if a is not None: com+=1; cor+=match(a,g)
    return tot,com,cor

N = len([i for i in pools if gold.get(i)])
chains_used = 0
final = {}
active = {iid: perm[iid][:4] for iid in pools}
chains_used += sum(len(v) for v in active.values())
for level, size in ((2, 8), (3, None)):
    dec = decide(active)
    resolved = {iid: a for iid, a in dec.items() if a is not None}
    final.update(resolved)
    unresolved = [iid for iid in active if dec[iid] is None]
    print(f"level n={len(set(len(v) for v in active.values()))and max(len(v) for v in active.values())}: resolved so far {len(final)}, escalating {len(unresolved)}")
    nxt = {}
    for iid in unresolved:
        take = perm[iid][:size] if size else perm[iid]
        nxt[iid] = take
        chains_used += len(take) - len(active[iid])
    # keep already-resolved pools out; escalated pools re-decided WITH full world
    # context: world = resolved answers as pluralities + escalated pools' own
    active = {**{iid: perm[iid][:0] for iid in ()}, **nxt}
    # world context needs all languages: rebuild sub with resolved pools at their
    # last size plus escalated at new size
    context = {iid: (perm[iid][:4] if iid in final else nxt.get(iid, perm[iid][:4])) for iid in pools}
    for iid in final:  # resolved pools keep their answer; represent as their sample
        pass
    active_context = context
    dec2 = decide(active_context)
    for iid in list(nxt):
        if dec2[iid] is not None:
            final[iid] = dec2[iid]
    active = {iid: nxt[iid] for iid in nxt if iid not in final}
for iid in pools:
    final.setdefault(iid, None)
print()
print(f"{'tier':<6}{'coverage':>10}{'precision':>11}")
for t in TIERS+(None,):
    tot,com,cor = stats(final,t)
    print(f"{t or 'ALL':<6}{100*com/tot:>9.1f}%{100*cor/max(com,1):>10.1f}%")
full_cost = sum(len(v) for v in perm.values())
print(f"\nchains generated: {chains_used} / {full_cost} full-pool = {100*chains_used/full_cost:.1f}% of full generation cost")
rem = [iid for iid,a in final.items() if a is None and gold.get(iid)]
print(f"unresolved after ladder: {len(rem)}")
print("DONE")
