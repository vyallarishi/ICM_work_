"""harvest_world.py AUC 0.926 -- is the 'world-agreement' feature circular
with the genuine/contaminated LABEL?

Label: genuine   = pool plurality matches gold AND chain is_gold
       contaminated = pool plurality WRONG AND chain not gold
Feature: fraction of OTHER languages' pluralities matching the chain answer.

For a genuine chain: answer == gold. For a contaminated chain: answer ==
the pool's wrong plurality. Other languages mostly get gold right, so
world-agreement ~ 'is this answer gold' almost by construction."""
import sys, random, json
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools, match

pools=load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
r8=[json.loads(l) for l in open("stage8_chains.jsonl",encoding="utf-8")]
rng=random.Random(SEED); demos=build_demo_bank(pools,rng)
di={d["iid"] for ds in demos.values() for d,_ in ds}
sample=[s for s in gate_sample(pools,rng) if s["instance_id"] not in di]
for s,r in zip(sample,r8): s["J3"]=r.get("J3")

lang_plur=defaultdict(dict)
for iid,p in pools.items(): lang_plur[iid.rsplit("_",1)[-1]][p["lang"]]=p["plurality"]

gen,con=[],[]
for s in sample:
    if s["contested"] or s.get("J3") is None: continue
    pool=pools[s["instance_id"]]
    t2a={c["text"]:c["answer"] for c in pool["chains"]}
    ans=t2a.get(s["text"])
    if ans is None: continue
    prob=s["instance_id"].rsplit("_",1)[-1]
    others=[a for l,a in lang_plur[prob].items() if l!=s["lang"]]
    world=sum(match(ans,a) for a in others)/max(len(others),1)
    cr=match(pool["plurality"],pool["gold"])
    rec={"world":world,"ans":ans,"gold":pool["gold"],"is_gold":s["is_gold"]}
    if cr and s["is_gold"]: gen.append(rec)
    elif not cr and not s["is_gold"]: con.append(rec)

print(f"{len(gen)} genuine / {len(con)} contaminated")
print(f"world AUC = {auc([g['world'] for g in gen],[c['world'] for c in con]):.3f} (log 0.926)")

print("\n=== CIRCULARITY EVIDENCE ===")
print(f"genuine chains whose answer == gold:      "
      f"{sum(match(g['ans'],g['gold']) for g in gen)}/{len(gen)}")
print(f"contaminated chains whose answer == gold: "
      f"{sum(match(c['ans'],c['gold']) for c in con)}/{len(con)}")
print("\n-> The label is (essentially) 'answer==gold'. The feature is")
print("   'other languages' majority answer == this answer'. Since other")
print("   languages are usually right, the feature is a NOISY COPY of the")
print("   label. High AUC is expected and does NOT demonstrate a usable")
print("   filter on unlabeled data beyond 'trust the multilingual vote'.")

print("\n=== Is world-agreement available at deployment? YES (no gold needed) ===")
print("   so it IS a legitimate model-free filter -- the caveat is only that")
print("   0.926 must not be read as 'detects bad reasoning'.")

print("\n=== base rates ===")
import statistics as st
print(f"  mean world(genuine)      = {st.mean(g['world'] for g in gen):.3f}")
print(f"  mean world(contaminated) = {st.mean(c['world'] for c in con):.3f}")
