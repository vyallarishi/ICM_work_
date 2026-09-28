import sys, copy, random
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import Counter, defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
base=g.load_pools(ROOT)
print("pools:",len(base))

true={i:{"share":p["share"],"plur":p["plurality"],"gold":p["gold"],"lang":p["lang"],"n":len(p["chains"])}
      for i,p in base.items()}
d8=s17.apply_diverse_views(copy.deepcopy(base))

LAND=0.70
rows=defaultdict(Counter)
for i,p in d8.items():
    t=true[i]; r=rows[t["lang"]]
    r["n"]+=1
    r["share_drop"] += (t["share"]-p["share"])
    if (t["share"]>=LAND)!=(p["share"]>=LAND):
        r["land_flip"]+=1
        r["land_flip_to_contested"] += (t["share"]>=LAND and p["share"]<LAND)
        r["land_flip_to_committed"] += (t["share"]<LAND and p["share"]>=LAND)
    if t["plur"]!=p["plurality"]: r["plur_flip"]+=1
    r["acc_true"] += g.match(t["plur"],t["gold"])
    r["acc_d8"]   += g.match(p["plurality"],p["gold"])
    r["contested_true"] += (t["share"]<LAND); r["contested_d8"] += (p["share"]<LAND)

print(f"\n{'lang':<6}{'tier':<6}{'n':>5}{'landflip':>9}{'->cont':>8}{'->comm':>8}{'plurflip':>9}{'cont_true':>10}{'cont_d8':>9}{'acc_true':>10}{'acc_d8':>9}{'mean_dshare':>13}")
tot=Counter()
for l in sorted(rows):
    r=rows[l]; tot.update(r)
    print(f"{l:<6}{g.TIER.get(l,'?'):<6}{r['n']:>5}{r['land_flip']:>9}{r['land_flip_to_contested']:>8}"
          f"{r['land_flip_to_committed']:>8}{r['plur_flip']:>9}{r['contested_true']:>10}{r['contested_d8']:>9}"
          f"{r['acc_true']/r['n']:>10.3f}{r['acc_d8']/r['n']:>9.3f}{r['share_drop']/r['n']:>13.4f}")
print(f"{'ALL':<6}{'':<6}{tot['n']:>5}{tot['land_flip']:>9}{tot['land_flip_to_contested']:>8}"
      f"{tot['land_flip_to_committed']:>8}{tot['plur_flip']:>9}{tot['contested_true']:>10}{tot['contested_d8']:>9}"
      f"{tot['acc_true']/tot['n']:>10.3f}{tot['acc_d8']/tot['n']:>9.3f}{tot['share_drop']/tot['n']:>13.4f}")

print("\n=== order-sensitivity of diverse8_view on REAL pools ===")
rng=random.Random(99); diff=0; checked=0; plurdiff=0
for i,p in list(base.items()):
    ch=list(p["chains"])
    if len(ch)<=8: continue
    checked+=1
    a=s17.diverse8_view(i,ch)
    sh=list(ch); rng.shuffle(sh)
    b=s17.diverse8_view(i,sh)
    if Counter(c["answer"] for c in a)!=Counter(c["answer"] for c in b):
        diff+=1
        ca=Counter(c["answer"] for c in a); cb=Counter(c["answer"] for c in b)
        pa=max(ca,key=lambda x:(ca[x],x)); pb=max(cb,key=lambda x:(cb[x],x))
        if pa!=pb: plurdiff+=1
print(f"  pools with >8 chains: {checked}")
print(f"  answer-multiset differs when input order permuted: {diff} ({100*diff/max(checked,1):.1f}%)")
print(f"  ... of which the resulting PLURALITY differs: {plurdiff}")
