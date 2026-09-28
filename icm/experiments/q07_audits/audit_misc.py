import sys, glob, json
from pathlib import Path
from collections import Counter
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g

print("=== vacuous self-test assertion (stage9_loop.py:225-226 / stage17:271-272) ===")
print("  expression: all(label == match(...) or True for ...)")
print("  `X or True` is always True ->", all(False or True for _ in range(3)), "-> assertion can never fail")

print("\n=== real-data magnitude check: any answers >= 1e28 in phaseA? ===")
ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
big=Counter(); tot=0; biggest=[]
from decimal import Decimal, InvalidOperation
for path in sorted(glob.glob(f"{ROOT}/final/pooled/generated/*.jsonl")):
    lang=Path(path).stem
    for line in open(path,encoding="utf-8"):
        row=json.loads(line)
        for c in row.get("candidates",[]):
            raw=c.get("pred_number")
            if raw is None: continue
            tot+=1
            try: d=Decimal(str(raw).replace(",","").replace(" ",""))
            except Exception: continue
            if not d.is_finite(): continue
            if g.norm(raw) is None:
                big[lang]+=1
                if len(biggest)<10: biggest.append((lang,raw))
print(f"  total non-null pred_numbers: {tot}")
print(f"  values norm() silently drops as None (>=29 sig digits): {sum(big.values())} {dict(big)}")
print(f"  examples: {biggest}")

print("\n=== gate_sample: does it exclude demo instances? main() does; self_test does NOT ===")
print("  main(): sample filtered by `s['instance_id'] not in demo_instances`  -> OK")

print("\n=== build_demo_bank early-continue skips remaining chains once bank full? ===")
print("  line 136: `if len(pos)>=2 and len(neg)>=1: continue` -> correct short-circuit")
print("  BUT: a language with NO landslide pool, or whose landslide pools are all unanimous,")
print("       gets NO negative demo -> bank of size 2, silently. Count on real data:")
import random
pools=g.load_pools(ROOT)
demos=g.build_demo_bank(pools, random.Random(20260815))
for lang in sorted(demos):
    n=len(demos[lang]); labels=sorted(l for _,l in demos[lang])
    flag="" if n==3 else "  <== INCOMPLETE BANK"
    print(f"    {lang:<6} tier={g.TIER[lang]:<5} demos={n} labels={labels}{flag}")
missing=[l for l in g.TIER if l not in demos]
print("  languages with NO bank at all:", missing)
