"""Harvest-cleaning frontier (CPU, cached J3 margins) — the judge's arena
that voting cannot enter: contaminated chains AGREE with the committed
majority, so no vote distinguishes them. Extends harvest_evidence.py with
(a) within-language centered margins (format_confound finding),
(b) per-tier AUC, (c) a full retention/contamination-rate curve."""
import json, random, statistics as st
from collections import defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import (SEED, TIER, auc, build_demo_bank, gate_sample,
                               load_pools, match)
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage8_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(rows)

lang_all = defaultdict(list)
for s, r in zip(sample, rows):
    if r.get("J3") is not None:
        lang_all[s["lang"]].append(r["J3"])
mu = {l: st.mean(v) for l, v in lang_all.items()}

harvest = []   # (margin_centered, margin_raw, genuine?, tier)
for s, r in zip(sample, rows):
    if r.get("contested") or r.get("J3") is None:
        continue
    pool = pools.get(s["instance_id"])
    if pool is None:
        continue
    commit_right = match(pool["plurality"], pool["gold"])
    if commit_right and s["is_gold"]:
        harvest.append((r["J3"] - mu[s["lang"]], r["J3"], True, s["tier"]))
    elif not commit_right and not s["is_gold"]:
        harvest.append((r["J3"] - mu[s["lang"]], r["J3"], False, s["tier"]))

gen = [h for h in harvest if h[2]]; con = [h for h in harvest if not h[2]]
print(f"harvest sample: {len(gen)} genuine, {len(con)} contaminated "
      f"(base contamination {100*len(con)/len(harvest):.2f}%)")
for label, idx in (("raw", 1), ("lang-centered", 0)):
    print(f"\n=== {label} margins ===")
    print(f"AUC genuine-vs-contaminated: "
          f"{auc([h[idx] for h in gen], [h[idx] for h in con]):.3f}")
    for tier in ("high", "mid", "low"):
        g = [h[idx] for h in gen if h[3] == tier]
        c = [h[idx] for h in con if h[3] == tier]
        a = auc(g, c)
        print(f"  {tier:<5} AUC {a if a else float('nan'):.3f}   ({len(g)} gen / {len(c)} con)")
    both = sorted(harvest, key=lambda h: h[idx])
    n = len(both)
    print(f"{'drop%':>6}{'kept':>7}{'contam kept':>12}{'contam rate':>12}{'genuine cost':>13}")
    for drop in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        k = int(drop * n)
        kept = both[k:]
        ck = sum(1 for h in kept if not h[2])
        gl = sum(1 for h in both[:k] if h[2])
        print(f"{100*drop:>5.0f}%{len(kept):>7}{ck:>12}{100*ck/len(kept):>11.2f}%"
              f"{100*gl/len(gen):>12.1f}%")
print("\nDONE")
