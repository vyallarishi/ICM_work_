import random, itertools, sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc

def brute(pos, neg):
    # pairwise concordance with midrank tie credit
    n=0; s=0.0
    for p in pos:
        for q in neg:
            n+=1
            if p>q: s+=1.0
            elif p==q: s+=0.5
    return s/n

rng=random.Random(12345)
bad=0
for trial in range(20000):
    npos=rng.randint(1,8); nneg=rng.randint(1,8)
    # heavy ties: small integer domain
    dom=rng.choice([[0,1],[0,1,2],[0,1,2,3],[5],[0,0,0,1]])
    pos=[rng.choice(dom) for _ in range(npos)]
    neg=[rng.choice(dom) for _ in range(nneg)]
    a=auc(pos,neg); b=brute(pos,neg)
    if a is None or abs(a-b)>1e-9:
        bad+=1
        if bad<=5: print("MISMATCH", pos, neg, a, b)
print("auc int-domain trials mismatches:", bad)

# float domain incl exact ties
bad2=0
for trial in range(20000):
    npos=rng.randint(1,10); nneg=rng.randint(1,10)
    vals=[round(rng.uniform(-2,2),1) for _ in range(4)]
    pos=[rng.choice(vals) for _ in range(npos)]
    neg=[rng.choice(vals) for _ in range(nneg)]
    a=auc(pos,neg); b=brute(pos,neg)
    if a is None or abs(a-b)>1e-9:
        bad2+=1
        if bad2<=5: print("MISMATCH2", pos, neg, a, b)
print("auc float-domain mismatches:", bad2)

# all-identical, empty edge cases
print("all tied:", auc([1,1,1],[1,1]), "expect 0.5")
print("empty pos:", auc([], [1]))
print("nan present:", auc([float('nan'),1.0],[0.0]))
