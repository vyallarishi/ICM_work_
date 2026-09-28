"""Is the peer judge (0.928/0.918 contested) beating TRIVIAL answer-matching,
or re-deriving it? Baselines on the same gate sample, no model:
  A: n peers (of the exact 3 shown) whose pool-plurality answer matches
     the chain's answer
  B: chain answer matches the world plurality over ALL 14 other languages
  C: fraction of 14 other languages whose plurality matches chain answer
Then: AUC of the judge WITHIN baseline-A strata (does the model add signal
beyond peer-answer identity?)."""
import json, random
from collections import Counter, defaultdict
from .verdict_judge_gate import (SEED, TIER, auc, build_demo_bank, gate_sample,
                               load_pools, match)
from .peer_conditioned_judge import build_peer_bank, select_peers, N_PEERS
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
rows = [json.loads(l) for l in open(f"{RUN_DIR}/stage19_chains.jsonl", encoding="utf-8")]
rng = random.Random(SEED)
demos = build_demo_bank(pools, rng)
demo_inst = {d["iid"] for ds in demos.values() for d, _ in ds}
sample = [s for s in gate_sample(pools, rng) if s["instance_id"] not in demo_inst]
assert len(sample) == len(rows)
for s, r in zip(sample, rows):
    s["P3full"], s["P3blind"] = r.get("P3full"), r.get("P3blind")

# reconstruct the exact peers each instance saw (same deterministic rng)
peer_bank = build_peer_bank(pools, random.Random(SEED + 2))
plur = {iid: p["plurality"] for iid, p in pools.items()}
lang_plur = defaultdict(dict)
for iid, p in pools.items():
    lang_plur[iid.rsplit("_", 1)[-1]][p["lang"]] = plur[iid]

def peer_answers(iid, lang):
    problem = iid.rsplit("_", 1)[-1]
    entry = peer_bank.get(problem, {})
    langs = [l for l in entry if l != lang]
    r = random.Random((SEED, iid).__repr__())
    r.shuffle(langs)
    langs.sort(key=lambda l: not entry[l]["landslide"])
    return [plur[f"train_{l}_{problem}"] for l in langs[:N_PEERS]]

# recover each sampled chain's answer via text join
for s in sample:
    txt2ans = {c["text"]: c["answer"] for c in pools[s["instance_id"]]["chains"]}
    s["answer"] = txt2ans.get(s["text"])
sc = [s for s in sample if s.get("P3blind") is not None and s["answer"] is not None]
print(f"{len(sc)} chains evaluated")

for s in sc:
    pa = peer_answers(s["instance_id"], s["lang"])
    s["A"] = sum(match(s["answer"], a) for a in pa)
    problem = s["instance_id"].rsplit("_", 1)[-1]
    others = [a for l, a in lang_plur[problem].items() if l != s["lang"]]
    top = Counter(others).most_common(1)[0][0] if others else None
    s["B"] = 1.0 if (top is not None and match(s["answer"], top)) else 0.0
    s["C"] = (sum(match(s["answer"], a) for a in others) / len(others)) if others else 0.0

def row(label, key):
    g = [s[key] for s in sc if s["is_gold"]]
    r = [s[key] for s in sc if not s["is_gold"]]
    gc = [s[key] for s in sc if s["is_gold"] and s["contested"]]
    rc = [s[key] for s in sc if not s["is_gold"] and s["contested"]]
    print(f"{label:<38} ALL {auc(g,r):.3f}   contested {auc(gc,rc):.3f}")

row("baseline A: match among 3 shown peers", "A")
row("baseline B: match world plurality (14)", "B")
row("baseline C: world match fraction (14)", "C")
row("judge P3full", "P3full")
row("judge P3blind", "P3blind")

print("\n--- judge WITHIN peer-match strata (is there signal beyond A?) ---")
for stratum in (0, 1, 2, 3):
    sub = [s for s in sc if s["A"] == stratum]
    g = [s["P3blind"] for s in sub if s["is_gold"]]
    r = [s["P3blind"] for s in sub if not s["is_gold"]]
    a = auc(g, r)
    print(f"A={stratum}: n={len(sub):>5} ({len(g)} gold/{len(r)} rest)  "
          f"P3blind AUC {'nan' if a is None else f'{a:.3f}'}")
print("\n--- and among CONTESTED only, the deployment-relevant strata ---")
for stratum in (0, 1, 2, 3):
    sub = [s for s in sc if s["A"] == stratum and s["contested"]]
    g = [s["P3blind"] for s in sub if s["is_gold"]]
    r = [s["P3blind"] for s in sub if not s["is_gold"]]
    a = auc(g, r)
    print(f"A={stratum}: n={len(sub):>5} ({len(g)} gold/{len(r)} rest)  "
          f"P3blind AUC {'nan' if a is None else f'{a:.3f}'}")
print("DONE")
