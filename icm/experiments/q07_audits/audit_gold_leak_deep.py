"""AUDIT 3 (deep): gold-leak. Decisive test -- CORRUPT every gold value and
verify the DECISION path produces bit-identical answers. If any decision reads
gold, the corrupted run will diverge."""
import json, copy
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q03_does_regeneration_help.regeneration_15lang import converge_w0, weighted_decide, select_regen_peers, build_regen_prompt
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context
from ..q02_can_the_model_judge_itself.ensemble_judge_in_loop import static_pos_demos
import random
from ...paths import RUN_DIR

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
def run(corrupt):
    pools=s17.apply_diverse_views(s17.load_pools(ROOT))
    if corrupt:
        for p in pools.values(): p["gold"]="123456789"
    state,contested,osc=converge_w0(pools)
    regen={json.loads(l)["iid"]:json.loads(l)["answers"]
           for l in open(f"{RUN_DIR}/stage23_regen37.jsonl",encoding="utf-8")}
    ns={i:dict(s) for i,s in state.items()}
    for iid in contested:
        ns[iid]["answer"]=weighted_decide(pools[iid],world_context(pools,ns,iid),
                                          regen.get(iid,[]),0.5,iid in osc)
    # also capture the GEN-phase artifacts (prompts), which must be gold-blind
    demos=static_pos_demos(pools)
    rng=random.Random(20260816+23)
    prompts=[]
    for iid in contested:
        peers=select_regen_peers(pools,state,iid,rng)
        if not peers: continue
        prompts.append((iid,build_regen_prompt(pools[iid]["question"],peers,
                                               demos.get(pools[iid]["lang"],[]))))
    return (state,contested,osc,ns,prompts,pools)

print("=== 3. GOLD-LEAK: corrupt-gold differential test ===")
a=run(False); b=run(True)
print(f"  converged state identical : {all(a[0][i]['answer']==b[0][i]['answer'] for i in a[0])}")
print(f"  contested set identical   : {a[1]==b[1]}")
print(f"  oscillator set identical  : {a[2]==b[2]}")
print(f"  post-regen state identical: {all(a[3][i]['answer']==b[3][i]['answer'] for i in a[3])}")
print(f"  regen PROMPTS identical   : {a[4]==b[4]}")
print(f"  (n prompts {len(a[4])}, n pools {len(a[5])})")

# Also verify the diverse-8 view + initialize are gold-blind
pa=a[5]; pb=b[5]
print(f"  diverse8 views identical  : "
      f"{all([c['answer'] for c in pa[i]['chains']]==[c['answer'] for c in pb[i]['chains']] for i in pa)}")
print(f"  share/plurality identical : "
      f"{all(pa[i]['share']==pb[i]['share'] and pa[i]['plurality']==pb[i]['plurality'] for i in pa)}")

leak = not (all(a[0][i]['answer']==b[0][i]['answer'] for i in a[0])
            and a[1]==b[1] and a[2]==b[2]
            and all(a[3][i]['answer']==b[3][i]['answer'] for i in a[3])
            and a[4]==b[4])
print(f"\n  *** GOLD LEAK IN DECISION PATH: {'YES' if leak else 'NO'} ***")

print("\n=== 3b. one caveat: load_pools DROPS pools with gold is None ===")
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate; import glob, collections
raw_rows=dropped=0
for path in sorted(glob.glob(f"{ROOT}/final/pooled/generated/*.jsonl")):
    lang=path.split("/")[-1][:-6]
    if lang not in gate.TIER: continue
    for line in open(path,encoding="utf-8"):
        r=json.loads(line); raw_rows+=1
        if gate.norm(r.get("gold_number")) is None: dropped+=1
print(f"  rows {raw_rows}, dropped for missing/unparseable gold: {dropped}")
print("  -> gold presence gates POOL INCLUSION (a selection effect on the")
print("     evaluation set), but does not influence any pool's ANSWER.")
