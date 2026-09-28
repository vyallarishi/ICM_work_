import sys, glob, json, statistics
from pathlib import Path
from collections import Counter, defaultdict
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g
ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
TIER=g.TIER

def build(root, drop_length):
    pools={}
    for path in sorted(glob.glob(f"{root}/final/pooled/generated/*.jsonl")):
        lang=Path(path).stem
        if lang not in TIER: continue
        for line in open(path,encoding="utf-8"):
            row=json.loads(line)
            gold=g.norm(row.get("gold_number")); chains=[]
            for c in row.get("candidates",[]):
                a=g.norm(c.get("pred_number")); t=c.get("text") or ""
                if a is None or not t.strip(): continue
                if drop_length and c.get("finish_reason")=="length": continue
                chains.append({"answer":a})
            if not chains or gold is None: continue
            counts=Counter(c["answer"] for c in chains)
            pools[row["id"]]={"gold":gold,"lang":lang,"share":counts.most_common(1)[0][1]/len(chains),
                              "plurality":max(counts,key=lambda a:(counts[a],a))}
    return pools

A=build(ROOT,True)   # as shipped
B=build(ROOT,False)  # keeping truncated chains
print("pools with-filter:",len(A)," without:",len(B))
LAND=0.70
rows=defaultdict(Counter)
for iid in A:
    if iid not in B: continue
    la=A[iid]["lang"]; r=rows[la]
    r["n"]+=1
    ca = A[iid]["share"]>=LAND; cb = B[iid]["share"]>=LAND
    if ca!=cb: r["landslide_flip"]+=1
    if A[iid]["plurality"]!=B[iid]["plurality"]: r["plurality_flip"]+=1
    r["acc_with"] += g.match(A[iid]["plurality"],A[iid]["gold"])
    r["acc_without"] += g.match(B[iid]["plurality"],B[iid]["gold"])
    r["contested_with"] += (not ca); r["contested_without"] += (not cb)
print(f"\n{'lang':<6}{'tier':<6}{'n':>5}{'landflip':>10}{'plurflip':>10}{'contested_w':>13}{'contested_wo':>14}{'plur_acc_w':>12}{'plur_acc_wo':>13}")
tot=Counter()
for l in sorted(rows):
    r=rows[l]; tot.update(r)
    print(f"{l:<6}{TIER[l]:<6}{r['n']:>5}{r['landslide_flip']:>10}{r['plurality_flip']:>10}"
          f"{r['contested_with']:>13}{r['contested_without']:>14}"
          f"{r['acc_with']/r['n']:>12.3f}{r['acc_without']/r['n']:>13.3f}")
print(f"{'ALL':<6}{'':<6}{tot['n']:>5}{tot['landslide_flip']:>10}{tot['plurality_flip']:>10}"
      f"{tot['contested_with']:>13}{tot['contested_without']:>14}"
      f"{tot['acc_with']/tot['n']:>12.3f}{tot['acc_without']/tot['n']:>13.3f}")
print("\nBy TIER:")
tt=defaultdict(Counter)
for l in rows: tt[TIER[l]].update(rows[l])
for t in ("high","mid","low"):
    r=tt[t]
    print(f"  {t:<5} n={r['n']:<5} landslide_flips={r['landslide_flip']:<5} plurality_flips={r['plurality_flip']:<5} "
          f"contested_with={r['contested_with']:<5} contested_without={r['contested_without']:<5} "
          f"plur_acc {r['acc_with']/r['n']:.3f} -> {r['acc_without']/r['n']:.3f}")
