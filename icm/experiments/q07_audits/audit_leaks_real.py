"""Item 3 (real data): leaks() calls match(norm(n), answer). But `answer`
comes from candidates_for()/wrong_candidate() -- are those already norm()ed?
And match() requires DECIMAL-parseable strings. Test the actual failure mode:
match() raises or misbehaves when norm() returns None."""
import sys, re, json
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import norm, match
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def leaks(text, answer):
    if answer is None: return False
    return any(match(norm(n), answer) for n in NUMBER_RE.findall(text))

print("=== does norm() ever return None for regex-extracted numbers? ===")
# norm returns None on absurd magnitude (long digit runs)
big = "1"*40
print(f"  norm('{big[:12]}...' 40 digits) = {norm(big)!r}")
huge = "9"*400
print(f"  norm(400-digit run) = {norm(huge)!r}")
print(f"  leaks(400-digit-run text, '40') = {leaks(huge, '40')}  (norm->None, match returns False: safe)")

print("\n=== REAL DATA: measure leak-flag prevalence and the space-separator risk ===")
rows=[json.loads(l) for l in open("stage27_scores.jsonl",encoding="utf-8")]
from collections import Counter
c=Counter((r["arm"], r["leaked"]) for r in rows)
print("  stage27 arm/leaked counts:", dict(c))
for arm in ("full","truncated","trunc_wrong"):
    tot=sum(v for (a,l),v in c.items() if a==arm)
    lk=sum(v for (a,l),v in c.items() if a==arm and l)
    if tot: print(f"    {arm:<12} leak rate {100*lk/tot:5.1f}%  ({lk}/{tot})")

print("\n=== do space-separated / EU-format numbers actually occur in the corpus? ===")
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools
pools=load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
SPACE_NUM=re.compile(r"\d\s\d{3}\b")
EU=re.compile(r"\d\.\d{3},\d")
SCI=re.compile(r"\d[eE][+-]?\d")
ns=ne=nsci=tot=0
for iid,p in pools.items():
    for ch in p["chains"]:
        t=ch["text"]; tot+=1
        if SPACE_NUM.search(t): ns+=1
        if EU.search(t): ne+=1
        if SCI.search(t): nsci+=1
print(f"  chains total {tot}")
print(f"  space-separated thousands (e.g. '1 234'): {ns} ({100*ns/tot:.2f}%)")
print(f"  EU decimal-comma (e.g. '1.234,56'):       {ne} ({100*ne/tot:.2f}%)")
print(f"  scientific notation (e.g. '1e3'):         {nsci} ({100*nsci/tot:.2f}%)")
