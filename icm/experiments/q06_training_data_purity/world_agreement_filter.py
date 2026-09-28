"""Exact world-agreement as a harvest cleaner: contaminated chains agree
with their LOCAL commit, but do they agree with the WORLD? Score each
harvested chain by the fraction of other languages' pluralities matching
its answer (exact, CPU). Compare vs J3 0.677 and continuation 0.634,
and check the combination for the double-consensus blind spot."""
import json, random
from collections import defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import SEED, auc, build_demo_bank, gate_sample, load_pools, match
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
r8 = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
for s, r in zip(sample, r8): s["J3"] = r.get("J3")

lang_plur = defaultdict(dict)
for iid, p in pools.items():
    lang_plur[iid.rsplit("_", 1)[-1]][p["lang"]] = p["plurality"]

genuine, contaminated = [], []
for s in sample:
    if s["contested"] or s.get("J3") is None:
        continue
    pool = pools[s["instance_id"]]
    txt2ans = {c["text"]: c["answer"] for c in pool["chains"]}
    ans = txt2ans.get(s["text"])
    if ans is None: continue
    problem = s["instance_id"].rsplit("_", 1)[-1]
    others = [a for l, a in lang_plur[problem].items() if l != s["lang"]]
    world = sum(match(ans, a) for a in others) / max(len(others), 1)
    rec = {"world": world, "J3": s["J3"]}
    commit_right = match(pool["plurality"], pool["gold"])
    if commit_right and s["is_gold"]: genuine.append(rec)
    elif not commit_right and not s["is_gold"]: contaminated.append(rec)

print(f"{len(genuine)} genuine vs {len(contaminated)} contaminated")
print(f"world-agreement (exact, no model): AUC "
      f"{auc([g['world'] for g in genuine], [c['world'] for c in contaminated]):.3f}")
print(f"J3 (reference):                    AUC "
      f"{auc([g['J3'] for g in genuine], [c['J3'] for c in contaminated]):.3f}")

# the blind spot: contaminated chains whose answer ALSO matches the world
# plurality (double consensus). Only there does a content judge matter.
for thresh in (0.5, 0.3):
    hi_c = [c for c in contaminated if c["world"] >= thresh]
    hi_g = [g for g in genuine if g["world"] >= thresh]
    a = auc([g["J3"] for g in hi_g], [c["J3"] for c in hi_c])
    print(f"world>= {thresh}: {len(hi_g)} genuine / {len(hi_c)} contaminated "
          f"remain; J3 AUC there {'nan' if a is None else f'{a:.3f}'}")
# practical two-stage filter: drop world<0.3, then J3 on the rest
kept_g = [g for g in genuine if g["world"] >= 0.3]
kept_c = [c for c in contaminated if c["world"] >= 0.3]
print(f"\ntwo-stage: world>=0.3 keeps {len(kept_g)}/{len(genuine)} genuine "
      f"({100*len(kept_g)/len(genuine):.1f}%), "
      f"{len(kept_c)}/{len(contaminated)} contaminated "
      f"({100*len(kept_c)/len(contaminated):.1f}%)")
print(f"contamination rate: before {100*len(contaminated)/(len(genuine)+len(contaminated)):.2f}% "
      f"-> after {100*len(kept_c)/(len(kept_g)+len(kept_c)):.2f}%")
print("DONE")
