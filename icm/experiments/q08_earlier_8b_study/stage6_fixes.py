# DIAGNOSIS -> FIX, measured:
#   Fix 1 (veto gap): global-first keeps its local fallback, but a local commit
#          contradicted by a >=70% world majority ABSTAINS instead.
#   Fix 2 (residual starvation): pools still unresolved at n=4 ESCALATE to
#          their full ~16 chains (adaptive sampling), pipeline reruns on the
#          escalated pool; still-unresolved -> best-guess world-leader.
import sys, json
from collections import Counter, defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS
from .stage6_user_spec_config import subsample_answers, global_first_decisions, WORLD_STRENGTH, LANDSLIDE

pools, gold = load()
sub = subsample_answers(pools)

def decide(sub_pools, veto=True):
    plur=defaultdict(dict)
    for iid,a in sub_pools.items():
        if a:
            c=Counter(a)
            plur[iid.rsplit("_",1)[-1]]["_".join(iid.split("_")[1:-1])]=max(c,key=lambda k:(c[k],k))
    out={}
    for iid,a in sub_pools.items():
        if not a: out[iid]={"answer":None,"route":"empty"}; continue
        c=Counter(a); L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
        others=[p for l,p in plur[P].items() if l!=L]
        w=Counter(others).most_common(1) if others else []
        strong = w and w[0][1]/len(others)>=WORLD_STRENGTH
        if strong and w[0][0] in c:
            out[iid]={"answer":w[0][0],"route":"global"}
        elif c.most_common(1)[0][1]/len(a)>=LANDSLIDE:
            top=max(c,key=lambda k:(c[k],k))
            if veto and strong and not match(w[0][0],top):
                out[iid]={"answer":None,"route":"vetoed"}
            else:
                out[iid]={"answer":top,"route":"local"}
        else:
            out[iid]={"answer":None,"route":"residual"}
    return out, plur

def stats(dec, tier=None):
    com=cor=tot=0
    for iid,d in dec.items():
        g=gold.get(iid)
        if g is None: continue
        if tier and TIER["_".join(iid.split("_")[1:-1])]!=tier: continue
        tot+=1
        if d["answer"] is not None: com+=1; cor+=match(d["answer"],g)
    return tot,com,cor

print("=== FIX 1: user-spec + veto on the local fallback (still n=4) ===")
dec_v,_ = decide(sub, veto=True)
for t in TIERS+(None,):
    tot,com,cor=stats(dec_v,t)
    print(f"{t or 'ALL':<6} coverage {100*com/tot:5.1f}%  precision {100*cor/max(com,1):5.1f}%")

print()
print("=== FIX 1+2: unresolved pools escalate to their full ~16 chains ===")
unresolved=[iid for iid,d in dec_v.items() if d["answer"] is None]
esc=dict(sub)
for iid in unresolved:
    esc[iid]=[a for _,a in pools[iid]]
dec_e, plur_e = decide(esc, veto=True)
for t in TIERS+(None,):
    tot,com,cor=stats(dec_e,t)
    print(f"{t or 'ALL':<6} coverage {100*com/tot:5.1f}%  precision {100*cor/max(com,1):5.1f}%")
esc_n=len(unresolved); esc_frac=100*esc_n/len([i for i in sub if gold.get(i)])
print(f"pools escalated: {esc_n} ({esc_frac:.1f}%) -> extra generation cost "
      f"~{esc_frac*12/16:.0f}% of a full-pool run")

print()
print("=== best-guess tier on what remains (world-leader rule) ===")
rem=[iid for iid,d in dec_e.items() if d["answer"] is None and gold.get(iid)]
hits=0; low=[0,0]
for iid in rem:
    L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
    c=Counter(esc[iid])
    others=[p for l,p in plur_e[P].items() if l!=L]
    ws={a: sum(match(p,a) for p in others)/max(len(others),1) for a in c}
    pick=max(c,key=lambda a:(ws[a],a))
    ok=match(pick,gold[iid]); hits+=ok
    if TIER[L]=="low": low[0]+=1; low[1]+=ok
print(f"remaining {len(rem)} pools: world-leader best-guess {100*hits/max(len(rem),1):.1f}% "
      f"(low tier: {low[1]}/{low[0]} = {100*low[1]/max(low[0],1):.1f}%)")
print("DONE")
