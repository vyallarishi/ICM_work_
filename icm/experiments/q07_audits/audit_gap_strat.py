"""How much of the P3 gap survives controlling for peer count?
stage28_scores.jsonl lacks n_peers, so use the stage27-scale reconstruction
to estimate the S-vs-peercount slope, then check stage27's own gap."""
import json
from collections import defaultdict
rows=[json.loads(l) for l in open("stage27_scores.jsonl",encoding="utf-8")]
by=defaultdict(list)
for r in rows: by[r["arm"]].append(r)
print("stage27 arms:")
for a,v in by.items():
    print(f"  {a:<14} n={len(v):>5} mean_S={sum(r['S'] for r in v)/len(v):.4f} "
          f"leak_rate={sum(r['leaked'] for r in v)/len(v):.3f}")
res=json.load(open("stage27_results.json"))
print("\nstage27 P3:", res.get("P3_gap"))

print("\n=== Does mean_S even measure what P3 claims? ===")
print("P3 compares S(settled|trunc) vs S(wrong|trunc). But `settled` candidates")
print("are frequently CORRECT and `wrong` candidates are by construction NOT the")
print("consensus. Check: how often is the settled candidate actually gold?")
import sys  # [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
pools=load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
for arm in ("truncated","trunc_wrong"):
    n=g=0
    for r in by[arm]:
        p=pools.get(r["iid"])
        if not p: continue
        n+=1; g+=bool(match(r["answer"], p["gold"]))
    print(f"  {arm:<14} candidates that are GOLD: {g}/{n} = {100*g/max(n,1):.1f}%")
print("\n=> If the wrong arm is ~0% gold and settled ~50%+ gold, the 'gap' is")
print("   partly just 'consensus answers are easier to re-derive', which is")
print("   what P3 intends -- BUT peer selection differs too (see audit_peercount).")

print("\n=== Same-answer control: restrict to pools where BOTH arms scored the")
print("    SAME answer value (isolates peer-selection effect) ===")
sa=defaultdict(dict)
for arm in ("truncated","trunc_wrong"):
    for r in by[arm]: sa[(r["iid"],r["answer"])][arm]=r["S"]
both=[(v["truncated"],v["trunc_wrong"]) for v in sa.values() if len(v)==2]
print(f"  pairs with identical (pool,answer) in both arms: {len(both)}")
if both:
    print(f"  mean S truncated={sum(a for a,_ in both)/len(both):.4f} "
          f"trunc_wrong={sum(b for _,b in both)/len(both):.4f} "
          f"gap={(sum(a for a,_ in both)-sum(b for _,b in both))/len(both):+.4f}")
