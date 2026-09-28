# Deterministic half of the diverse-4 proposal: build diverse-4 views (repo
# logic: one chain per answer group largest-first, fillers from shuffled
# remainder), then run local vote -> veto -> global tie-break on THOSE views.
import random
from collections import Counter, defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS
WORLD=0.70; LAND=0.70
pools, gold = load()

def diverse4(iid, answers):
    if len(answers) <= 4: return list(answers)
    rng = random.Random((1729, iid, "diverse").__repr__())
    shuffled = list(answers); rng.shuffle(shuffled)
    groups = defaultdict(list)
    for a in shuffled: groups[a].append(a)
    ordered = sorted(groups.values(), key=lambda g: -len(g))
    sel = [g[0] for g in ordered[:4]]
    if len(sel) < 4:
        for a in shuffled:
            # fillers: frequency-weighted draw from remainder
            sel.append(a)
            if len(sel) == 4: break
    return sel[:4]

views = {iid: diverse4(iid, [a for _,a in ch]) for iid, ch in pools.items()}
plur = defaultdict(dict)
for iid, a in views.items():
    if a:
        c = Counter(a)
        plur[iid.rsplit("_",1)[-1]]["_".join(iid.split("_")[1:-1])] = max(c, key=lambda k:(c[k],k))
dec = {}
for iid, a in views.items():
    if not a: dec[iid]=(None,"empty"); continue
    c = Counter(a); L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
    others=[p for l,p in plur[P].items() if l!=L]
    w=Counter(others).most_common(1) if others else []
    strong = w and w[0][1]/len(others) >= WORLD
    if strong and w[0][0] in c:
        dec[iid]=(w[0][0],"global")
    elif c.most_common(1)[0][1]/len(a) >= LAND:
        top=max(c,key=lambda k:(c[k],k))
        dec[iid]=((None,"vetoed") if (strong and not match(w[0][0],top)) else (top,"local")) if True else None
        if isinstance(dec[iid][0],tuple): dec[iid]=dec[iid][0]
    else:
        dec[iid]=(None,"residual")
# fix nesting mistake robustly
fixed={}
for iid,v in dec.items():
    if isinstance(v[0],tuple): fixed[iid]=v[0]
    else: fixed[iid]=v
dec=fixed
print(f"{'tier':<6}{'coverage':>10}{'precision':>11}{'residual':>10}{'res gold-in-4':>15}")
for t in TIERS+(None,):
    tot=com=cor=res=resg=0
    for iid,(a,route) in dec.items():
        g=gold.get(iid)
        if g is None: continue
        if t and TIER["_".join(iid.split("_")[1:-1])]!=t: continue
        tot+=1
        if a is not None: com+=1; cor+=match(a,g)
        elif route in ("residual","vetoed"):
            res+=1; resg+=any(match(x,g) for x in views[iid])
    print(f"{t or 'ALL':<6}{100*com/tot:>9.1f}%{100*cor/max(com,1):>10.1f}%{res:>10}{100*resg/max(res,1):>14.1f}%")
routes=Counter(r for _,r in dec.values())
print(f"routes: {dict(routes)}")
print("DONE")
