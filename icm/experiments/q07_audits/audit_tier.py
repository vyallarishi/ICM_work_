import sys, glob
from pathlib import Path
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself import verdict_judge_gate as g
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q08_earlier_8b_study import rung3_reasoning_structure as r3

print("=== TIER dict gating: languages silently dropped by load_pools ===")
for root,label in [
  ("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA","phaseA"),
  ("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37","full37"),
  ("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500","full37-1500"),
]:
    files=sorted(Path(p).stem for p in glob.glob(f"{root}/final/pooled/generated/*.jsonl"))
    if not files:
        print(f"  {label}: (root not present)"); continue
    s8=g.TIER if not hasattr(g,'_orig') else g.TIER
    dropped8=[f for f in files if f not in {"en","de","es","fr","ru","hi","ar","ko","fa","am","sw","kn","eu","ml","is"}]
    dropped17=[f for f in files if f not in s17.TIER]
    print(f"  {label}: {len(files)} lang files on disk")
    print(f"     dropped by stage8's own 15-lang TIER: {len(dropped8)} {dropped8}")
    print(f"     dropped by stage17's 37-lang TIER:    {len(dropped17)} {dropped17}")

print("\n=== IMPORT-ORDER GLOBAL MUTATION: stage17 rebinds gate.TIER at import ===")
print("  stage17_diverse8_37.py:47  `_gate.TIER = TIER`  (module-level side effect)")
from ..q02_can_the_model_judge_itself import verdict_judge_gate as stage8_judge_gate; import importlib
print("  after importing stage17, stage8_judge_gate.TIER has", len(stage8_judge_gate.TIER), "languages")
print("  -> any module importing stage8 AFTER stage17 sees the 37-lang TIER;")
print("     importing BEFORE (and not importing stage17) sees 15. Results depend on import order.")
print("  stage25_scale.py imports s17 (line 29) THEN gate (line 30) -> gets 37-lang TIER. Intended, but implicit.")

print("\n=== gate_sample uses TIER[lang] for the tier slice ===")
print("  stage8:174  s['tier']=TIER[pool['lang']] -> if TIER was mutated by an stage17 import,")
print("  the SAME script yields different tier slices. Demonstrate:")
print("   'ky' in stage8.TIER now?", "ky" in stage8_judge_gate.TIER, "(would KeyError before stage17 import)")

print("\n=== rung3 TIER (37) vs stage8 TIER (15): different tier ASSIGNMENTS for shared langs? ===")
S8={"en":"high","de":"high","es":"high","fr":"high","ru":"mid","hi":"mid","ar":"mid","ko":"mid","fa":"mid",
    "am":"low","sw":"low","kn":"low","eu":"low","ml":"low","is":"low"}
mismatch=[(l,S8[l],r3.TIER[l]) for l in S8 if l in r3.TIER and S8[l]!=r3.TIER[l]]
print("  tier-assignment mismatches stage8-15 vs rung3/stage17-37:", mismatch)
