"""AUDIT C/4: harm accounting + iteration-order dependence."""
import json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
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

seq={i:dict(s) for i,s in state.items()}
for iid in contested:
    seq[iid]["answer"]=weighted_decide(pools[iid],world.context(seq,iid),
                                       regen.get(iid,[]),ALPHA,iid in osc)

print("=== C. HARM ACCOUNTING: full transition matrix over contested pools ===")
cats=Counter()
harmed=[]
for iid in contested:
    b=state[iid]["answer"]; a=seq[iid]["answer"]
    bs = "NULL" if b is None else ("OK" if match(b,pools[iid]["gold"]) else "WRONG")
    as_ = "NULL" if a is None else ("OK" if match(a,pools[iid]["gold"]) else "WRONG")
    cats[(bs,as_)]+=1
    if bs=="OK" and as_!="OK": harmed.append(iid)
for k in sorted(cats): print(f"  {k[0]:<5} -> {k[1]:<5} : {cats[k]}")
print(f"\n  correct->WRONG  : {cats[('OK','WRONG')]}")
print(f"  correct->NULL   : {cats[('OK','NULL')]}   <- 'zero correct withdrawn' refers ONLY to this")
print(f"  TOTAL correct answers destroyed = {len(harmed)}")
print(f"  published 'correct_withdrawn' = 0, published 'wrong_commits_added' = 82")
print(f"  82 = {cats[('OK','WRONG')]} (correct->wrong) + 72 (wrong recoveries)")
print(f"  -> the {cats[('OK','WRONG')]} correct->wrong flips are NOT reported as harm anywhere;")
print(f"     stage23 DOES report them as 'harm_correct_to_wrong'. stage25 dropped that key.")

print("\n=== 4. ORDER DEPENDENCE: stage25 reads MUTATING new_state ===")
# stage25:174  others = world.context(new_state, iid)   <- new_state being mutated
# stage22:312  others = world_context(pools, new_state, iid)  <- same pattern
# stage9 loop:346 others = world_context(pools, state, iid)   <- frozen state
# converge():84  world.context(state, iid)  <- frozen. So the REGEN step alone
# is sequential while every converge lap is synchronous.
syn={i:dict(s) for i,s in state.items()}
for iid in contested:
    syn[iid]["answer"]=weighted_decide(pools[iid],world.context(state,iid),
                                       regen.get(iid,[]),ALPHA,iid in osc)
diff=[i for i in contested if seq[i]["answer"]!=syn[i]["answer"]]
print(f"  pools differing seq vs sync: {len(diff)}")
for iid in diff[:8]:
    print(f"    {iid}: seq={seq[iid]['answer']!r} sync={syn[iid]['answer']!r} "
          f"gold={pools[iid]['gold']} converged={state[iid]['answer']!r}")

# order sensitivity: shuffle contested order, redo sequential
import random
print("\n  -- sequential result under 5 different pool orderings --")
for s in range(5):
    order=list(contested); random.Random(s).shuffle(order)
    ns={i:dict(st) for i,st in state.items()}
    for iid in order:
        ns[iid]["answer"]=weighted_decide(pools[iid],world.context(ns,iid),
                                          regen.get(iid,[]),ALPHA,iid in osc)
    N=len(pools)
    comm=[i for i,st in ns.items() if st["answer"] is not None]
    hits=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in comm)
    abst=[i for i in contested if state[i]["answer"] is None]
    rec=[i for i in abst if ns[i]["answer"] is not None]
    rok=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in rec)
    wd=sum(1 for i in contested if state[i]["answer"] is not None
           and match(state[i]["answer"],pools[i]["gold"]) and ns[i]["answer"] is None)
    print(f"   order{s}: cov={len(comm)/N:.4f} prec={hits/len(comm):.4f} "
          f"rec={len(rec)} rec_ok={rok} rec_prec={rok/max(len(rec),1):.4f} withdrawn={wd}")
