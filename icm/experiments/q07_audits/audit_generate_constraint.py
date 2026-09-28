"""AUDIT D (empirical): on the real 37x1500 run, is every committed answer one
the language ACTUALLY produced (in its own chains, or in its own regen)?"""
import json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load, ALPHA
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
pools=load(ROOT); world=World(pools)
state,contested,osc=converge(pools,world)
regen={}
for l in open(f"{RUN_DIR}/stage25_regen.jsonl",encoding="utf-8"):
    r=json.loads(l); regen[r["iid"]]=r["answers"]
final={i:dict(s) for i,s in state.items()}
for iid in contested:
    final[iid]["answer"]=weighted_decide(pools[iid],world.context(final,iid),
                                         regen.get(iid,[]),ALPHA,iid in osc)

print("=== D. generate-constraint over ALL 55,485 pools, final state ===")
viol_exact=viol_match=0; ex=[]
for iid,s in final.items():
    a=s["answer"]
    if a is None: continue
    own=set(c["answer"] for c in pools[iid]["chains"]) | set(regen.get(iid,[]))
    if a not in own:
        viol_exact+=1
        if not any(match(a,o) for o in own):
            viol_match+=1; ex.append((iid,a,sorted(own)[:6]))
print(f"  committed pools: {sum(1 for s in final.values() if s['answer'] is not None)}")
print(f"  answers not EXACTLY in own chains+own regen: {viol_exact}")
print(f"  answers not even MATCH-equal to anything the lang produced: {viol_match}")
for e in ex[:5]: print("   ",e)

print("\n=== D2. is the constraint tight on the CONVERGED (pre-regen) state too? ===")
v=0
for iid,s in state.items():
    a=s["answer"]
    if a is None: continue
    own=set(c["answer"] for c in pools[iid]["chains"])
    if a not in own: v+=1
print(f"  converged answers absent from own chains: {v}")

print("\n=== D3. NOTE: diverse-8 VIEW shrinks the pool. Is 'generated itself'")
print("    judged against the 8-chain view or the full ~27-chain pool? ===")
raw=s17.load_pools(ROOT)
shrunk=sum(1 for i in raw if len(raw[i]['chains'])>8)
print(f"  pools with >8 raw chains (view discards some): {shrunk}/{len(raw)}")
# does any final answer come from a DISCARDED chain? that's still self-generated,
# so it would be fine -- but check the reverse: view-only answers.
print("  (discarding chains only TIGHTENS the constraint; no violation possible)")

print("\n=== D4. regen provenance: is regen for iid generated in iid's LANGUAGE? ===")
print("  regen prompt = static_pos_demos[lang] (target-lang format demos)")
print("  + peers from OTHER languages' chains (foreign text in prompt)")
print("  + the target-language question. Output language is only ANCHORED,")
print("  never verified. Check: is there any language check on regen output?")
import subprocess
print(subprocess.run(["grep","-n","lang",f"{RUN_DIR}/stage22_regen.py"],
                     capture_output=True,text=True).stdout)
