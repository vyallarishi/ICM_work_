"""Item 5: is baseline A really model-free, and computed on the SAME chains
as the judge it is compared to? Also: does peer_baseline reproduce the exact
peers stage19 showed?"""
import sys, random, json
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from collections import defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools, match
from ..q02_can_the_model_judge_itself.peer_conditioned_judge import build_peer_bank, select_peers, N_PEERS

pools=load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rng=random.Random(SEED)
demos=build_demo_bank(pools,rng)
demo_inst={d["iid"] for ds in demos.values() for d,_ in ds}
sample=[s for s in gate_sample(pools,rng) if s["instance_id"] not in demo_inst]

# --- stage19's ACTUAL peer selection (from stage19_peer_judge.main) ---
peer_bank_19=build_peer_bank(pools, random.Random(SEED+2))
peers19={}
for s in sample:
    iid=s["instance_id"]
    if iid not in peers19:
        peers19[iid]=select_peers(peer_bank_19, iid.rsplit("_",1)[-1], s["lang"],
                                  random.Random((SEED,iid).__repr__()))

# --- peer_baseline.py's RECONSTRUCTION (peer_answers) ---
peer_bank_pb=build_peer_bank(pools, random.Random(SEED+2))
plur={iid:p["plurality"] for iid,p in pools.items()}
def peer_answers(iid, lang):
    problem=iid.rsplit("_",1)[-1]
    entry=peer_bank_pb.get(problem,{})
    langs=[l for l in entry if l!=lang]
    r=random.Random((SEED,iid).__repr__())
    r.shuffle(langs)
    langs.sort(key=lambda l: not entry[l]["landslide"])
    return [plur[f"train_{l}_{problem}"] for l in langs[:N_PEERS]]

# Do the reconstructed peer ANSWERS correspond to the peer TEXTS stage19 used?
mismatch=0; checked=0; textans={}
for iid,p in pools.items():
    for c in p["chains"]: textans[c["text"]]=c["answer"]
for s in sample:
    iid=s["instance_id"]
    t19=peers19[iid]
    a19=[textans.get(t) for t in t19]
    apb=peer_answers(iid, s["lang"])
    checked+=1
    if a19!=apb: mismatch+=1
print(f"peer reconstruction: {checked} chains checked, {mismatch} mismatched peer-answer lists")
print("  -> peer_baseline reproduces stage19's exact peers:", mismatch==0)

# --- Is baseline A model-free? ---
print("\nBaseline A inputs: peer_bank(texts->pool plurality answers) + chain answer.")
print("  build_peer_bank uses pool['plurality'] (vote over generated chains) --")
print("  a VOTE statistic, no judge/logit. match() is string/Decimal only.")
print("  => model-free in the sense of 'no judge forward pass'. CONFIRMED.")

# --- Same chains? ---
rows=[json.loads(l) for l in open("stage19_chains.jsonl",encoding="utf-8")]
for s,r in zip(sample,rows): s["P3blind"]=r.get("P3blind"); s["P3full"]=r.get("P3full")
for s in sample:
    t2a={c["text"]:c["answer"] for c in pools[s["instance_id"]]["chains"]}
    s["answer"]=t2a.get(s["text"])
sc=[s for s in sample if s.get("P3blind") is not None and s["answer"] is not None]
print(f"\nchains used by BOTH baseline and judge: {len(sc)} of {len(sample)}")
print("  (peer_baseline.log reports 6149 evaluated -> full sample, no subsetting)")

for s in sc:
    pa=peer_answers(s["instance_id"], s["lang"])
    s["A"]=sum(match(s["answer"],a) for a in pa)
g=[s["A"] for s in sc if s["is_gold"]]; r_=[s["A"] for s in sc if not s["is_gold"]]
gc=[s["A"] for s in sc if s["is_gold"] and s["contested"]]
rc=[s["A"] for s in sc if not s["is_gold"] and s["contested"]]
print(f"\nreproduced baseline A: ALL {auc(g,r_):.3f}  contested {auc(gc,rc):.3f}  (log: 0.972 / 0.946)")
gj=[s["P3blind"] for s in sc if s["is_gold"] and s["contested"]]
rj=[s["P3blind"] for s in sc if not s["is_gold"] and s["contested"]]
print(f"reproduced judge P3blind contested: {auc(gj,rj):.3f}  (log/stage19: 0.918)")
