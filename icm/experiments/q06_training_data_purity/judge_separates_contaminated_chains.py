"""Direct harvest-value measurement: in the would-be training set (chains from
landslide pools), can the judge separate TRUE positives (chains agreeing with
a CORRECT commit) from FALSE positives (chains agreeing with a WRONG commit)?
Uses stage8's saved 14B margins joined against the real pools."""
import json
from collections import Counter
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match, auc
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]

true_pos, false_pos = [], []
for r in rows:
    if r.get("contested") or r.get("J3") is None:
        continue                       # landslide pools only (the harvest source)
    pool = pools.get(r["instance_id"])
    if pool is None:
        continue
    commit_right = match(pool["plurality"], pool["gold"])
    # harvested chains = those agreeing with the commit:
    #   right commit -> its gold chains are harvested (true positives)
    #   wrong commit -> its non-gold plurality chains are harvested (false pos)
    if commit_right and r["is_gold"]:
        true_pos.append(r["J3"])
    elif not commit_right and not r["is_gold"]:
        false_pos.append(r["J3"])

print(f"harvest sample: {len(true_pos)} genuine chains, {len(false_pos)} contaminated")
score = auc(true_pos, false_pos)
print(f"judge (J3) separates genuine from contaminated: AUC {score:.3f}")

# practical filter: drop the lowest-margin fraction of the harvest
both = sorted([(m, 1) for m in true_pos] + [(m, 0) for m in false_pos])
n_fp = len(false_pos)
for drop in (0.05, 0.10, 0.20):
    k = int(drop * len(both))
    dropped = both[:k]
    fp_removed = sum(1 for _, is_true in dropped if is_true == 0)
    tp_lost = k - fp_removed
    print(f"drop lowest {int(drop*100)}% by margin: removes {fp_removed}/{n_fp} contaminated "
          f"({100*fp_removed/max(n_fp,1):.0f}%), costs {tp_lost} genuine "
          f"({100*tp_lost/max(len(true_pos),1):.1f}%)")
print("DONE")
