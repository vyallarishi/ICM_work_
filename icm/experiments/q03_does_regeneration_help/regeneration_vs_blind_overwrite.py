"""Control for stage22: is regen recovery just 'trust the world majority'?
For the 79 converged abstentions, compute accuracy of simply OVERWRITING
with the strong world answer (the move the constraint forbids). If regen's
98% ~= overwrite accuracy, regen = constraint-respecting delivery of world
knowledge + an in-language chain; if regen > overwrite, the language's
re-derivation FILTERS bad world majorities (extra signal)."""
import json
from collections import Counter
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context
from .regeneration_15lang import converge_w0
from ...paths import RUN_DIR

pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA")
state, contested, osc = converge_w0(pools)
abst = [i for i in contested if state[i]["answer"] is None]
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage22_regen.jsonl", encoding="utf-8")}

ow_n = ow_ok = 0
for iid in abst:
    others = world_context(pools, state, iid)
    if not others: continue
    top, n = Counter(others).most_common(1)[0]
    if n / len(others) >= 0.70:
        ow_n += 1
        ow_ok += match(top, pools[iid]["gold"])
print(f"abstained pools with a strong world majority: {ow_n}/{len(abst)}")
print(f"forced world-overwrite accuracy there: {ow_ok}/{ow_n} = {100*ow_ok/max(ow_n,1):.1f}%")

# where regen FAILED to inject gold: was the world majority wrong there?
no_inject = [i for i in abst if not any(match(a, pools[i]["gold"]) for a in regen.get(i, []))]
wrong_world = 0
for iid in no_inject:
    others = world_context(pools, state, iid)
    if others:
        top, n = Counter(others).most_common(1)[0]
        if n/len(others) >= 0.70 and not match(top, pools[iid]["gold"]):
            wrong_world += 1
print(f"pools where regen produced no gold: {len(no_inject)}; "
      f"of these, world majority strong AND wrong: {wrong_world}")
print("DONE")
