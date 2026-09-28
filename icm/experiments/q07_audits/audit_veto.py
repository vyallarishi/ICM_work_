import sys
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q01_can_voting_label_data.consensus_loop_15lang import update_pool
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match

print("=== BUG A: veto membership test uses `in counts` (exact string) but scoring uses match() ===")
# Local pool produced 10.0 (normed '10') ... make the world answer within tolerance but not identical
pool={"chains":[{"answer":"100"},{"answer":"100"},{"answer":"50"}]}
others=["100.4"]*10   # match('100.4','100') is True (0.4% <= 0.5%) but '100.4' not in counts
print("match('100.4','100') =", match("100.4","100"))
print("world = 10x '100.4', local pool has '100','100','50'")
print("  update_pool ->", update_pool(pool, others, None, 0.0),
      "   <-- VETOED to None even though the world answer IS the local answer within tolerance")
print()
print("=== Same situation with exact string equality: no veto ===")
print("  update_pool ->", update_pool(pool, ["100"]*10, None, 0.0))
print()
print("=== BUG A2: reverse — veto suppressed when it should fire? ===")
# world answer string-present in counts but numerically the local key is a different value? not possible
print()
print("=== BUG B: epsilon 0.001*count vs world_share gap ===")
def probe(local_counts, others, label):
    chains=[]
    for a,n in local_counts.items(): chains += [{"answer":a}]*n
    pool={"chains":chains}
    res=update_pool(pool, others, None, 0.0)
    # compute world shares
    ws={a: sum(match(o,a) for o in others)/len(others) for a in local_counts}
    print(f"  {label}: local={local_counts} world_share={ {k:round(v,4) for k,v in ws.items()} } -> pick {res}")
    return res

print("Q: can 0.001*count override a genuine world_share gap? Yes when gap < 0.001*delta_count.")
# world of size N: share gap of 1/N is the smallest possible nonzero gap.
for N in (500,1000,1200,2000):
    others=["A"]*( (N//2)+1 ) + ["B"]*(N - (N//2) -1)
    # A has strictly more world votes than B (gap = 1/N ~ 0.002 for N=500)
    lc={"A":1,"B":20}
    r=probe(lc, others, f"N_world={N} gapshare={(1/N):.5f}")
print()
print("Concrete flip: world_share gap 1/N beaten by epsilon when 0.001*(cB-cA) > 1/N, i.e. N > 1/(0.001*dc).")
for dc in (1,5,10,19):
    print(f"  delta_count={dc:>3}: epsilon={0.001*dc:.4f} -> flips any world gap < {0.001*dc:.4f}; needs |world| > {int(1/(0.001*dc))+1}")
print()
print("With |others| <= 14 (15 langs, phaseA), min nonzero world gap = 1/14 = 0.0714 > max epsilon 0.001*20=0.02 -> SAFE there.")
print("But stage25 full37: |others| <= 36, min gap = 1/36 = 0.0278 > 0.02 -> still safe, margin thin.")
for n in (36,40,50,60,80,100):
    print(f"  |others|={n:>4}: min nonzero world gap = {1/n:.4f}  vs max epsilon 0.001*20=0.0200  -> {'UNSAFE' if 1/n<0.02 else 'safe'}")
