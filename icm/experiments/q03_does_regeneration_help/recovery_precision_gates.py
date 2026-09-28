"""Recovery-precision refinement sweep (CPU, stage25 cache): the 91.3%
overall recovery precision hides a tier gradient (low 94.3%, mid 89.1%,
high 68.6%). High-resource abstention usually signals a globally-hard
problem (world likely wrong), not language weakness. Test label-blind
gates: (a) world-majority strength threshold for recovery commits,
(b) tier-gated commits, (c) regen-majority within recovery. Report the
recovery/precision frontier of each."""
import json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from .regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

pools = load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
world = World(pools)
state, contested, osc = converge(pools, world)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage25_regen.jsonl", encoding="utf-8")}
abst = [i for i in contested if state[i]["answer"] is None]

# recompute recoveries once (T=1, alpha=0.5, as shipped)
rec = {}
for iid in abst:
    others = world.context(state, iid)
    ans = weighted_decide(pools[iid], others, regen.get(iid, []), 0.5,
                          iid in osc)
    if ans is not None:
        top, n = Counter(others).most_common(1)[0] if others else (None, 0)
        ws = n / len(others) if others else 0.0
        rc = Counter(regen.get(iid, []))
        rec[iid] = {"ans": ans, "ok": match(ans, pools[iid]["gold"]),
                    "tier": gate.TIER[pools[iid]["lang"]],
                    "world_share": ws,
                    "regen_votes": rc.get(ans, 0)}
print(f"recoveries {len(rec)}; correct {sum(r['ok'] for r in rec.values())}")

def frontier(name, keep_fn):
    kept = [r for r in rec.values() if keep_fn(r)]
    ok = sum(r["ok"] for r in kept)
    print(f"{name:<42} kept {len(kept):>4} ({100*len(kept)/len(rec):.0f}%)  "
          f"precision {100*ok/max(len(kept),1):.1f}%")

frontier("no gate (shipped)", lambda r: True)
for t in (0.8, 0.9, 1.0):
    frontier(f"world share >= {t}", lambda r, t=t: r["world_share"] >= t)
frontier("low tier only", lambda r: r["tier"] == "low")
frontier("low+mid tiers", lambda r: r["tier"] != "high")
for v in (2, 3):
    frontier(f"regen votes >= {v}", lambda r, v=v: r["regen_votes"] >= v)
frontier("low+mid AND world>=0.8",
         lambda r: r["tier"] != "high" and r["world_share"] >= 0.8)
frontier("world>=0.9 OR low tier",
         lambda r: r["world_share"] >= 0.9 or r["tier"] == "low")
print("DONE")
