import sys, json, random
from collections import Counter, defaultdict
from .rung7_small_pools_and_routing import load, match, TIER, TIERS, SEED
from .stage6_user_spec_config import subsample_answers, global_first_decisions, N_CHAINS, WORLD_STRENGTH
from ...paths import RUN_DIR

pools, gold = load()
sub = subsample_answers(pools)
decisions, plur = global_first_decisions(sub)

print("=== DIAGNOSIS 1: where do the committed errors come from? ===")
buckets=defaultdict(lambda: [0,0,0])
veto_gap=[0,0]
for iid,d in decisions.items():
    g=gold.get(iid)
    if g is None or d["answer"] is None: continue
    t=TIER["_".join(iid.split("_")[1:-1])]
    wrong=not match(d["answer"],g)
    b=buckets[(d["route"],t)]; b[0]+=1; b[1]+=wrong
    if wrong: b[2]+= any(match(a,g) for a in sub[iid])
    if d["route"]=="local":
        L="_".join(iid.split("_")[1:-1]); P=iid.rsplit("_",1)[-1]
        others=[p for l,p in plur[P].items() if l!=L]
        if others:
            w=Counter(others).most_common(1)[0]
            if w[1]/len(others)>=WORLD_STRENGTH and not match(w[0],d["answer"]):
                veto_gap[0]+=1; veto_gap[1]+=wrong
print(f"{'route':<8}{'tier':<6}{'n':>7}{'wrong':>7}{'wrong%':>8}{'gold-in-4':>11}")
for route in ("global","local"):
    for t in TIERS:
        n,w,gis=buckets[(route,t)]
        if n: print(f"{route:<8}{t:<6}{n:>7}{w:>7}{100*w/n:>7.1f}%{gis:>7}/{w}")
print(f"veto gap (local commit vs >=70% world contradiction): {veto_gap[0]} commits, {veto_gap[1]} wrong ({100*veto_gap[1]/max(veto_gap[0],1):.0f}%)")

print()
print("=== DIAGNOSIS 2: the residual's ceiling at n=4 ===")
items=[json.loads(l) for l in open(f"{RUN_DIR}/stage6_items.jsonl",encoding="utf-8")]
per=defaultdict(lambda:[0,0,0,0])
for it in items:
    iid=it["instance_id"]; t=it["tier"]; g=it["gold"]
    gis=any(match(a,g) for a in sub[iid])
    p=per[t]; p[0]+=1; p[1]+=gis
    if gis:
        p[2]+=match(it["world_leader"],g)
        p[3]+=match(it["fusion_J3"],g)
print(f"{'tier':<6}{'n':>6}{'gold-in-4':>11}{'WL|gold':>10}{'fusionJ3|gold':>15}")
for t in TIERS:
    n,gis,wl,fu=per[t]
    print(f"{t:<6}{n:>6}{gis:>6} ({100*gis/n:.0f}%){wl:>6}/{gis} ({100*wl/max(gis,1):.0f}%){fu:>7}/{gis}")

print()
print("=== DIAGNOSIS 3: demo-bank label quality (J3's input) ===")
import glob, pyarrow.parquet as pq
from .rung7_small_pools_and_routing import norm
from .stage6_user_spec_config import build_demo_bank
wanted={i for i,d in decisions.items() if d["answer"] is not None}
chain_lookup=defaultdict(list)
for f in glob.glob("/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200/export/**/broad.parquet",recursive=True):
    for r in pq.read_table(f,columns=["instance_id","canonical_answer","parse_status","question","chain_text"]).to_pylist():
        if r["instance_id"] not in wanted or r["parse_status"]!="parsed": continue
        a=norm(r["canonical_answer"])
        if a is not None:
            chain_lookup[r["instance_id"]].append({"answer":a,"question":r["question"],"chain_text":r["chain_text"],"iid":r["instance_id"]})
rng=random.Random(SEED+1)
bank=build_demo_bank(sub, decisions, chain_lookup, rng)
tier_demo=defaultdict(lambda:[0,0])
for lang,demoset in bank.items():
    t=TIER[lang]
    for chain,label in demoset:
        g=gold.get(chain.get("iid"))
        if g is None: continue
        truth=match(chain["answer"],g)
        tier_demo[t][0]+=1; tier_demo[t][1]+=(label==truth)
for t in TIERS:
    n,ok=tier_demo[t]
    print(f"{t:<6} demo labels actually correct: {ok}/{n} ({100*ok/max(n,1):.0f}%)")
print("DONE")
