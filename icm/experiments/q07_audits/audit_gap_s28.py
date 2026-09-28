"""Same-answer control on the POWERED stage28 sample (11,907 contested pools).
If the P3 'selectivity' gap is real (correct derivations complete better),
it must survive holding the ANSWER VALUE fixed. If it vanishes/reverses,
the gap is driven by which answers each arm selects, not by soundness."""
import json
from collections import defaultdict
rows=[json.loads(l) for l in open("stage28_scores.jsonl",encoding="utf-8")]
by=defaultdict(list)
for r in rows: by[r["arm"]].append(r)

for settled,wrong,tag in (("truncated","trunc_wrong","rule1"),
                          ("trunc2","trunc2_wrong","rule2")):
    sa=defaultdict(dict)
    for arm in (settled,wrong):
        for r in by[arm]: sa[(r["iid"],r["answer"])][arm]=r
    both=[v for v in sa.values() if len(v)==2]
    print(f"\n=== {tag}: {settled} vs {wrong} ===")
    print(f"  reported gap (all): "
          f"{sum(r['S'] for r in by[settled])/len(by[settled]) - sum(r['S'] for r in by[wrong])/len(by[wrong]):+.4f}")
    if both:
        ms=sum(v[settled]['S'] for v in both)/len(both)
        mw=sum(v[wrong]['S'] for v in both)/len(both)
        print(f"  SAME (pool,answer) pairs: n={len(both)}")
        print(f"    S_{settled}={ms:.4f}  S_{wrong}={mw:.4f}  gap={ms-mw:+.4f}")
        lf=[v for v in both if not v[settled]['leaked'] and not v[wrong]['leaked']]
        if lf:
            ms2=sum(v[settled]['S'] for v in lf)/len(lf)
            mw2=sum(v[wrong]['S'] for v in lf)/len(lf)
            print(f"    leak-free subset n={len(lf)}: {ms2:.4f} vs {mw2:.4f} gap={ms2-mw2:+.4f}")

print("\n=== Gold-status decomposition of each arm's candidates ===")
import sys  # [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
print("(stage28 uses the 37-lang full37-1500 root; check availability)")
import os
root="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
print("  root exists:", os.path.isdir(root))

print("\n=== Within-arm: does S depend on whether the candidate is CORRECT? ===")
print("(This is the question P3 *should* answer, and P2 already reports it.)")
res=json.load(open("stage28_results.json"))
for arm in ("truncated","trunc2","trunc_wrong","trunc2_wrong"):
    b=res.get(arm,{})
    for sub in ("all","leak_free"):
        d=b.get(sub,{})
        print(f"  {arm:<14}[{sub:<9}] P1_win_rate={d.get('P1_win_rate')} "
              f"P2_auc={d.get('P2_auc')} n_pools={d.get('P1_pools')}")
