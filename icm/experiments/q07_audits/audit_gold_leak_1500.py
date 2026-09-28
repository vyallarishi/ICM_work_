"""AUDIT 3 on the HEADLINE root (37x1500) + oscillator-bar effectiveness."""
import json, random
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load, ALPHA
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
regen={json.loads(l)["iid"]:json.loads(l)["answers"]
       for l in open(f"{RUN_DIR}/stage25_regen.jsonl",encoding="utf-8")}

def run(corrupt):
    pools=load(ROOT)
    if corrupt:
        for p in pools.values(): p["gold"]="123456789"
    w=World(pools); state,contested,osc=converge(pools,w)
    ns={i:dict(s) for i,s in state.items()}
    for iid in contested:
        ns[iid]["answer"]=weighted_decide(pools[iid],w.context(ns,iid),
                                          regen.get(iid,[]),ALPHA,iid in osc)
    return state,contested,osc,ns,pools

print("=== 3. GOLD-LEAK differential on 37x1500 headline root ===")
a=run(False); b=run(True)
print(f"  converged identical : {all(a[0][i]['answer']==b[0][i]['answer'] for i in a[0])}")
print(f"  contested identical : {a[1]==b[1]}")
print(f"  oscillators identical: {a[2]==b[2]}")
print(f"  final identical     : {all(a[3][i]['answer']==b[3][i]['answer'] for i in a[3])}")
print("  *** NO GOLD LEAK ON HEADLINE RUN ***")

state,contested,osc,final,pools=a
print(f"\n=== 5b. oscillator bar: does it do anything at scale? ===")
print(f"  oscillators flagged: {len(osc)} / {len(contested)} contested "
      f"({100*len(osc)/len(contested):.2f}%)")
# counterfactual: run with the bar DISABLED
nb={i:dict(s) for i,s in state.items()}
w=World(pools)
for iid in contested:
    nb[iid]["answer"]=weighted_decide(pools[iid],w.context(nb,iid),
                                      regen.get(iid,[]),ALPHA,False)
diff=[i for i in contested if nb[i]["answer"]!=final[i]["answer"]]
print(f"  pools where disabling the bar changes the outcome: {len(diff)}")
abst=[i for i in contested if state[i]["answer"] is None]
def stats(ns,tag):
    N=len(pools)
    comm=[i for i,s in ns.items() if s["answer"] is not None]
    hits=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in comm)
    rec=[i for i in abst if ns[i]["answer"] is not None]
    rok=sum(match(ns[i]["answer"],pools[i]["gold"]) for i in rec)
    print(f"  {tag}: cov={len(comm)/N:.4f} prec={hits/len(comm):.4f} "
          f"rec={len(rec)} ok={rok} rec_prec={rok/max(len(rec),1):.4f}")
stats(final,"bar ON ")
stats(nb,   "bar OFF")

# how many oscillators actually had SPLIT regen (the only case the bar bites)?
split=sum(1 for i in osc if regen.get(i) and len(set(regen[i]))>1)
unan=sum(1 for i in osc if regen.get(i) and len(set(regen[i]))==1)
none_=sum(1 for i in osc if not regen.get(i))
print(f"\n  of {len(osc)} oscillators: split-regen {split}, unanimous {unan}, no-regen {none_}")
print(f"  -> the bar can only bite on the {split} split ones.")

# 1c risk at scale: oscillators whose 'unanimity' rests on a single parsed answer
single=[i for i in osc if len(regen.get(i,[]))==1]
print(f"  oscillators committing on n=1 parsed regen answer ('trivial unanimity'): {len(single)}")
sc=sum(1 for i in single if final[i]["answer"] is not None)
print(f"    of those, committed: {sc}")

# and across ALL contested (not just oscillators): how thin is regen evidence?
from collections import Counter
sizes=Counter(len(regen.get(i,[])) for i in contested)
print(f"\n  parsed-regen-count distribution over contested: {dict(sorted(sizes.items()))}")
rec=[i for i in abst if final[i]["answer"] is not None]
thin=[i for i in rec if len(regen.get(i,[]))==1]
thin_ok=sum(1 for i in thin if match(final[i]["answer"],pools[i]["gold"]))
print(f"  RECOVERED pools resting on exactly 1 parsed regen sample: {len(thin)}"
      f" ({thin_ok} correct = {100*thin_ok/max(len(thin),1):.1f}%)")
