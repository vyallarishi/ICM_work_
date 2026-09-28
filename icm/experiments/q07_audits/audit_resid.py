"""AUDIT item 4: residualization correctness + monotonicity + per-language."""
import json, statistics as st
from collections import defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import auc

# --- 4a: is the slope cov/var, and does resid change ANY within-pool ranking? ---
print("="*72); print("4a. slope formula + rank-invariance"); print("="*72)
xs=[1,2,3,4,5]; ys=[2.0,3.9,6.2,7.8,10.1]
mx,my=st.mean(xs),st.mean(ys)
var=sum((x-mx)**2 for x in xs)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var
cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/len(xs); varn=var/len(xs)
print(f"  code slope {slope:.6f}   cov/var {cov/varn:.6f}   equal: {abs(slope-cov/varn)<1e-12}")
print("  -> OLS slope is correct (the /n cancels).")
print("\n  KEY: resid = raw - my - slope*(n-mx) is a STRICTLY MONOTONE transform")
print("  of raw ONLY within a fixed n_steps. Across different n it re-ranks.")
print("  For a pool where all chains share n_steps, resid ordering == raw ordering.")

# --- 4b: real data ---
scores=[]
for sh in range(4):
    for line in open(f"stage35_scores_{sh}of4.jsonl", encoding="utf-8"):
        r=json.loads(line); scores.append(r)
print(f"\nloaded {len(scores)} scored chains")

print("="*72); print("4b. residual vs n_steps: is the linear fit adequate?"); print("="*72)
xs=[r["n_steps"] for r in scores]; ys=[r["p_clean"] for r in scores]
mx,my=st.mean(xs),st.mean(ys)
var=sum((x-mx)**2 for x in xs)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var
print(f"  global slope {slope:+.4f}/step   mean n_steps {mx:.3f}")
by=defaultdict(list)
for r in scores:
    resid=r["p_clean"]-(my+slope*(r["n_steps"]-mx))
    by[r["n_steps"]].append(resid)
print(f"  {'n_steps':>8}{'count':>9}{'mean raw':>12}{'mean resid':>12}")
raw_by=defaultdict(list)
for r in scores: raw_by[r["n_steps"]].append(r["p_clean"])
for n in sorted(by):
    if len(by[n])<50: continue
    print(f"  {n:>8}{len(by[n]):>9}{st.mean(raw_by[n]):>12.3f}{st.mean(by[n]):>12.3f}")
print("\n  If 'mean resid' is not ~0 across n, the LINEAR fit leaves residual")
print("  length bias -> the correction over/under-shoots at the extremes.")

print()
print("="*72); print("4c. PER-LANGUAGE: does one global slope fit all languages?"); print("="*72)
bylang=defaultdict(list)
for r in scores: bylang[r["lang"]].append(r)
print(f"  {'lang':<7}{'tier':<6}{'n':>8}{'mean steps':>12}{'slope':>10}{'mean resid(global fit)':>24}")
rowsout=[]
for lang in sorted(bylang, key=lambda l:(gate.TIER.get(l,'z'),l)):
    v=bylang[lang]
    lx=[r["n_steps"] for r in v]; ly=[r["p_clean"] for r in v]
    lmx,lmy=st.mean(lx),st.mean(ly)
    lvar=sum((x-lmx)**2 for x in lx)
    lslope=sum((x-lmx)*(y-lmy) for x,y in zip(lx,ly))/lvar if lvar else 0.0
    mres=st.mean([r["p_clean"]-(my+slope*(r["n_steps"]-mx)) for r in v])
    rowsout.append((lang,gate.TIER.get(lang,'?'),len(v),lmx,lslope,mres))
    print(f"  {lang:<7}{gate.TIER.get(lang,'?'):<6}{len(v):>8}{lmx:>12.2f}{lslope:>10.4f}{mres:>24.3f}")
sl=[r[4] for r in rowsout]
print(f"\n  per-language slope range: {min(sl):+.4f} .. {max(sl):+.4f}  (global {slope:+.4f})")
mr=[r[5] for r in rowsout]
print(f"  per-language mean residual range under the GLOBAL fit: {min(mr):+.3f} .. {max(mr):+.3f}")
print("  A nonzero per-language mean residual = the global correction imports a")
print("  LANGUAGE offset into 'resid'. Selection is within-pool (single language),")
print("  so a constant per-language offset cancels there -- but any per-language")
print("  SLOPE difference does NOT cancel within a pool that mixes step counts.")
