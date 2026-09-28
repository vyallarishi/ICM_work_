import sys
from collections import defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS
from .stage6_user_spec_config import subsample_answers
pools, gold = load()
sub = subsample_answers(pools)
per = defaultdict(lambda: [0,0,0])   # tier -> [n, gold_in_4, gold_in_16]
for iid, answers4 in sub.items():
    g = gold.get(iid)
    if g is None or not answers4: continue
    t = TIER["_".join(iid.split("_")[1:-1])]
    full = [a for _, a in pools[iid]]
    p = per[t]; p[0]+=1
    p[1] += any(match(a,g) for a in answers4)
    p[2] += any(match(a,g) for a in full)
print(f"{'tier':<6}{'n':>7}{'gold in the 4':>15}{'gold in all ~16':>17}")
tot=[0,0,0]
for t in TIERS:
    n,g4,g16 = per[t]; tot[0]+=n; tot[1]+=g4; tot[2]+=g16
    print(f"{t:<6}{n:>7}{100*g4/n:>14.1f}%{100*g16/n:>16.1f}%")
print(f"{'ALL':<6}{tot[0]:>7}{100*tot[1]/tot[0]:>14.1f}%{100*tot[2]/tot[0]:>16.1f}%")
print("DONE")
