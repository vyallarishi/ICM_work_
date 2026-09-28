"""AUDIT A/B/C/F: independently recompute stage25 headline metrics from raw
pools + the cached regen answers, WITHOUT calling stage25.phase_decide."""
import json, sys
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load, ALPHA
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
pools=load(ROOT)
world=World(pools)
state,contested,osc=converge(pools,world)
print(f"pools={len(pools)} contested={len(contested)} oscillators={len(osc)}")

regen={}
for l in open(f"{RUN_DIR}/stage25_regen.jsonl",encoding="utf-8"):
    r=json.loads(l); regen[r["iid"]]=r["answers"]
print(f"regen cache: {len(regen)} pools, {sum(len(a) for a in regen.values())} answers")

# ---- replicate stage25.phase_decide EXACTLY (sequential, reads new_state) ----
seq={i:dict(s) for i,s in state.items()}
for iid in contested:
    seq[iid]["answer"]=weighted_decide(pools[iid],world.context(seq,iid),
                                       regen.get(iid,[]),ALPHA,iid in osc)
# ---- alternative: SYNCHRONOUS update (read from frozen converged state) ----
syn={i:dict(s) for i,s in state.items()}
for iid in contested:
    syn[iid]["answer"]=weighted_decide(pools[iid],world.context(state,iid),
                                       regen.get(iid,[]),ALPHA,iid in osc)

def report(tag,ns):
    N=len(pools)
    committed=[i for i,s in ns.items() if s["answer"] is not None]
    hits=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in committed)
    abst=[i for i in contested if state[i]["answer"] is None]
    rec=[i for i in abst if ns[i]["answer"] is not None]
    rec_ok=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in rec)
    withdrawn=wrongflip=0
    for iid in contested:
        if state[iid]["answer"] is None: continue
        was=match(state[iid]["answer"],pools[iid]["gold"])
        now=ns[iid]["answer"]
        if was and now is None: withdrawn+=1
        elif was and not match(now,pools[iid]["gold"]): wrongflip+=1
    print(f"\n--- {tag} ---")
    print(f"  coverage  {len(committed)}/{N} = {len(committed)/N:.4f}")
    print(f"  precision {hits}/{len(committed)} = {hits/len(committed):.4f}")
    print(f"  abstained_before {len(abst)}  recovered {len(rec)}  correct {rec_ok}"
          f"  rec_prec {rec_ok/max(len(rec),1):.4f}")
    print(f"  correct_withdrawn {withdrawn}  correct->wrong flips {wrongflip}")
    print(f"  wrong_commits_added (as published) = {wrongflip}+{len(rec)}-{rec_ok} = {wrongflip+len(rec)-rec_ok}")
    tiers={}
    for t in ("high","mid","low"):
        ta=[i for i in abst if gate.TIER[pools[i]["lang"]]==t]
        tr=[i for i in ta if ns[i]["answer"] is not None]
        tok=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in tr)
        tiers[t]={"abstained":len(ta),"recovered":len(tr),"correct":tok,
                  "prec":round(tok/max(len(tr),1),4)}
    print(f"  tiers {json.dumps(tiers)}")
    return dict(coverage=round(len(committed)/N,4),precision=round(hits/len(committed),4),
                recovered=len(rec),recovered_correct=rec_ok,
                rec_prec=round(rec_ok/max(len(rec),1),4),withdrawn=withdrawn,
                abstained_before=len(abst),tiers=tiers)

r_seq=report("SEQUENTIAL (what stage25 actually does)",seq)
r_syn=report("SYNCHRONOUS (what stage22/23 comment implies / stage9 loop does)",syn)

pub=json.load(open(f"{RUN_DIR}/stage25_results.json"))["with_regen"]
print("\n=== published stage25_results.json vs recompute ===")
for k,pk in (("coverage","coverage"),("precision","precision"),
             ("recovered","recovered"),("recovered_correct","recovered_correct"),
             ("rec_prec","recovery_precision"),("withdrawn","correct_withdrawn"),
             ("abstained_before","abstained_before")):
    print(f"  {pk:<22} published={pub[pk]!r:<10} seq={r_seq[k]!r:<10} syn={r_syn[k]!r}")
print("  published tiers:", json.dumps(pub["recovery_by_tier"]))

diff=[i for i in contested if seq[i]["answer"]!=syn[i]["answer"]]
print(f"\n*** pools where sequential != synchronous: {len(diff)} ***")
