# Replicates scripts/candidate_processing.select_diverse answer-coverage logic
# (label-blind): groups by predicted answer, largest groups first, one chain
# per group, seeded shuffle for ties. Measures gold coverage of the view.
import random
from collections import Counter, defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS
pools, gold = load()
SIZE=4
per=defaultdict(lambda:[0,0,0,0])  # tier -> [n, gold_in_div4, gold_in_rand4, gold_in_full]
rng_master=random.Random(1729)
rand=random.Random((20260810, 4, 0).__repr__())  # same stream family as stage6
for iid, chains in pools.items():
    g=gold.get(iid)
    answers=[a for _,a in chains]
    if g is None or not answers: continue
    t=TIER["_".join(iid.split("_")[1:-1])]
    # diverse-4: top-SIZE answer groups by count (seeded tie shuffle)
    order=list(Counter(answers).items())
    rng=random.Random((1729, iid, "diverse").__repr__())
    rng.shuffle(order)
    order.sort(key=lambda kv:-kv[1])
    div_answers={a for a,_ in order[:SIZE]}
    # random-4 for comparison (same as stage6 trial 0 family)
    r4 = rand.sample(answers,4) if len(answers)>4 else answers
    p=per[t]; p[0]+=1
    p[1]+= any(match(a,g) for a in div_answers)
    p[2]+= any(match(a,g) for a in r4)
    p[3]+= any(match(a,g) for a in answers)
print(f"{'tier':<6}{'n':>7}{'diverse-4':>11}{'random-4':>10}{'full ~16':>10}")
tot=[0]*4
for t in TIERS:
    n,d,r,f=per[t]
    for i,v in enumerate((n,d,r,f)): tot[i]+=v
    print(f"{t:<6}{n:>7}{100*d/n:>10.1f}%{100*r/n:>9.1f}%{100*f/n:>9.1f}%")
n,d,r,f=tot
print(f"{'ALL':<6}{n:>7}{100*d/n:>10.1f}%{100*r/n:>9.1f}%{100*f/n:>9.1f}%")
print("DONE")
