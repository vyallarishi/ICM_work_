"""Item 2 DEFINITIVE: on the REAL phaseA pools, replay each stage's rng
consumption and compare selected chain texts + verify against cached files."""
import random, sys, json, hashlib
# [ICM_work_] removed sys.path.insert(0,'/home/rvyalla/phase0') -- imports are package-relative
from ..q02_can_the_model_judge_itself.verdict_judge_gate import build_demo_bank, gate_sample, load_pools, SEED

ROOT="/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
pools=load_pools(ROOT)
print("pools:", len(pools))

def replay():
    rng=random.Random(SEED)
    demos=build_demo_bank(pools,rng)
    excl={d["iid"] for ds in demos.values() for d,_ in ds}
    return [s for s in gate_sample(pools,rng) if s["instance_id"] not in excl]

runs={n:replay() for n in ("stage8","stage18","stage19","stage20","stage41")}
base=runs["stage8"]
h=lambda ss: hashlib.sha256("\x00".join(s["text"] for s in ss).encode()).hexdigest()[:16]
print("sample size:", len(base), "text-hash:", h(base))
for n,r in runs.items():
    print(f"  {n:8s} n={len(r)} hash={h(r)} identical={[s['text'] for s in r]==[s['text'] for s in base]}")

# rng sensitivity on REAL data
rng=random.Random(SEED); demos=build_demo_bank(pools,rng); rng.random()
excl={d["iid"] for ds in demos.values() for d,_ in ds}
pert=[s for s in gate_sample(pools,rng) if s["instance_id"] not in excl]
pt=[s["text"] for s in pert]; bt=[s["text"] for s in base]
print("\nreal-data sensitivity to ONE extra rng draw:")
print("  identical texts:", pt==bt, " differing slots:", sum(1 for a,b in zip(pt,bt) if a!=b), "/", len(bt))

# match cached stage8_chains.jsonl exactly
rows=[json.loads(l) for l in open("stage8_chains.jsonl",encoding="utf-8")]
KEYS=("instance_id","lang","tier","is_gold","contested")
print("\nreplay matches cached stage8_chains.jsonl metadata, in order:",
      [tuple(s[k] for k in KEYS) for s in base]==[tuple(r[k] for k in KEYS) for r in rows],
      f"({len(base)} vs {len(rows)})")

# how many pools actually give the shuffle a choice?
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match, CHAIN_MAX_CHARS
ng_multi=nr_multi=0
for iid,p in pools.items():
    g=[c for c in p["chains"] if match(c["answer"],p["gold"]) and len(c["text"])<=CHAIN_MAX_CHARS]
    r=[c for c in p["chains"] if not match(c["answer"],p["gold"]) and len(c["text"])<=CHAIN_MAX_CHARS]
    if len(g)>2: ng_multi+=1
    if len(r)>2: nr_multi+=1
print(f"pools where shuffle actually selects: gold>2 in {ng_multi}, rest>2 in {nr_multi} (of {len(pools)})")
