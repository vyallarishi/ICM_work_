"""Item: the +0.217 selectivity gap (stage28 P3).
P3 compares mean_S of arm `trunc2` vs arm `trunc2_wrong`.
CRITICAL: are these computed over the SAME pools / comparable candidate sets?
mean_S for a subset is built from `allvals` = every candidate in every pool
that has >=1 kept candidate. The two arms have DIFFERENT pool sets and
DIFFERENT numbers of candidates per pool."""
import json
from collections import defaultdict, Counter
rows=[json.loads(l) for l in open("stage28_scores.jsonl",encoding="utf-8")]
print("total rows", len(rows))
by=defaultdict(list)
for r in rows: by[r["arm"]].append(r)
for a,v in by.items():
    print(f"  {a:<14} n={len(v):>6}  pools={len({r['iid'] for r in v}):>6} "
          f"leak_rate={sum(r['leaked'] for r in v)/len(v):.3f} mean_S={sum(r['S'] for r in v)/len(v):.4f}")

print("\n=== reproduce reported P3 gaps ===")
res=json.load(open("stage28_results.json"))
for k in res:
    if k.startswith("P3"): print(" ", k, res[k])

print("\n=== POOL OVERLAP between settled and wrong arms ===")
for settled,wrong in (("truncated","trunc_wrong"),("trunc2","trunc2_wrong")):
    ps={r["iid"] for r in by[settled]}; pw={r["iid"] for r in by[wrong]}
    print(f"  {settled} pools={len(ps)}  {wrong} pools={len(pw)}  "
          f"intersection={len(ps&pw)}  settled-only={len(ps-pw)}  wrong-only={len(pw-ps)}")

print("\n=== candidates per pool differ? (mean_S averages over candidates) ===")
for arm in ("trunc2","trunc2_wrong"):
    c=Counter()
    per=defaultdict(int)
    for r in by[arm]: per[r["iid"]]+=1
    print(f"  {arm:<14} candidates/pool: {Counter(per.values()).most_common()}")

print("\n=== RECOMPUTE gap restricted to SHARED pools (apples-to-apples) ===")
for settled,wrong,tag in (("truncated","trunc_wrong","rule1"),("trunc2","trunc2_wrong","rule2")):
    ps={r["iid"] for r in by[settled]}&{r["iid"] for r in by[wrong]}
    for subset in ("all","leak_free"):
        f=(lambda r: True) if subset=="all" else (lambda r: not r["leaked"])
        s=[r["S"] for r in by[settled] if r["iid"] in ps and f(r)]
        w=[r["S"] for r in by[wrong]   if r["iid"] in ps and f(r)]
        if not s or not w: continue
        ms,mw=sum(s)/len(s),sum(w)/len(w)
        print(f"  {tag:<6}[{subset:<9}] shared pools={len(ps):>5}  "
              f"S_settled={ms:.4f}(n={len(s)})  S_wrong={mw:.4f}(n={len(w)})  gap={ms-mw:+.4f}")

print("\n=== KEY CONFOUND CHECK ===")
print("The `wrong` arms use peers_supporting() (any chain matching the wrong")
print("answer, from ANY other language) while the settled arms use")
print("peers_for_answer() (that language's COMMITTED answer must match).")
print("So the two arms differ in PEER SELECTION, not only in answer correctness.")
print("\nHow many peers does each arm actually get? (n=3 cap both)")
