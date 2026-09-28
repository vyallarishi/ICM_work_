"""Verify stage8.auc() tie handling against a brute-force pairwise reference."""
import sys, random
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc
def brute(pos,neg):
    n=s=0
    for p in pos:
        for q in neg:
            n+=1
            s+= 1.0 if p>q else 0.5 if p==q else 0.0
    return s/n
rng=random.Random(12345); bad=0
for _ in range(4000):
    dom=rng.choice([[0,1],[0,1,2],[5],[0,0,0,1]])
    pos=[rng.choice(dom) for _ in range(rng.randint(1,8))]
    neg=[rng.choice(dom) for _ in range(rng.randint(1,8))]
    a,b=auc(pos,neg),brute(pos,neg)
    if abs(a-b)>1e-9: bad+=1
print("heavy-tie mismatches:",bad)
bad=0
for _ in range(4000):
    vals=[round(rng.uniform(-2,2),1) for _ in range(4)]
    pos=[rng.choice(vals) for _ in range(rng.randint(1,10))]
    neg=[rng.choice(vals) for _ in range(rng.randint(1,10))]
    a,b=auc(pos,neg),brute(pos,neg)
    if abs(a-b)>1e-9: bad+=1
print("float mismatches:",bad)
print("all tied:",auc([1,1,1],[1,1]),"expect 0.5")
print("empty pos:",auc([],[1]))
