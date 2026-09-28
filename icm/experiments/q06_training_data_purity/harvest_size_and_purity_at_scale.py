"""SFT harvest accounting at 1500-problem scale: how many chains, and how
PURE, under (a) no filter, (b) world-agreement filter, (c) with recovered
pools' regen chains included. Purity = fraction of harvested chains whose
answer matches gold (mislabeled chains poison SFT)."""
import json
from collections import Counter, defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide
from ...paths import RUN_DIR

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
pools = load(ROOT)
world = World(pools)
state, contested, osc = converge(pools, world)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage25_regen.jsonl", encoding="utf-8")}
final = {i: dict(s) for i, s in state.items()}
for iid in contested:
    final[iid]["answer"] = weighted_decide(pools[iid], world.context(final, iid),
                                           regen.get(iid, []), 0.5, iid in osc)

# world-agreement score per pool (exact, label-blind)
lang_ans = defaultdict(dict)
for iid, s in final.items():
    if s["answer"] is not None:
        lang_ans[iid.rsplit("_",1)[-1]][pools[iid]["lang"]] = s["answer"]
def world_share(iid, ans):
    prob = iid.rsplit("_",1)[-1]; lang = pools[iid]["lang"]
    others = [a for l,a in lang_ans[prob].items() if l != lang]
    return (sum(match(a, ans) for a in others)/len(others)) if others else 0.0

# harvest = chains agreeing with the committed answer
rows = []   # (world_share, correct?, tier, lang, is_recovered)
recovered = {i for i in contested if state[i]["answer"] is None
             and final[i]["answer"] is not None}
for iid, s in final.items():
    ans = s["answer"]
    if ans is None: continue
    ws = world_share(iid, ans)
    ok = match(ans, pools[iid]["gold"])
    n = sum(1 for c in pools[iid]["chains"] if match(c["answer"], ans))
    for _ in range(n):
        rows.append((ws, ok, gate.TIER[pools[iid]["lang"]], pools[iid]["lang"], False))
    if iid in recovered:   # regen chains supporting the committed answer
        n_regen = sum(1 for a in regen.get(iid, []) if match(a, ans))
        for _ in range(n_regen):
            rows.append((ws, ok, gate.TIER[pools[iid]["lang"]], pools[iid]["lang"], True))

def report(name, sel):
    sub = [r for r in rows if sel(r)]
    if not sub: print(f"{name:<44} (empty)"); return
    good = sum(r[1] for r in sub)
    print(f"{name:<44} {len(sub):>8} chains  purity {100*good/len(sub):.2f}%  "
          f"contamination {100*(len(sub)-good)/len(sub):.2f}%")

print(f"=== SFT HARVEST @ 1500 problems x 37 langs ===")
report("A. all committed pools, no filter", lambda r: not r[4])
report("B. + world-agreement >= 0.3", lambda r: not r[4] and r[0] >= 0.3)
report("C. + world-agreement >= 0.5", lambda r: not r[4] and r[0] >= 0.5)
report("D. B + recovered pools' regen chains", lambda r: r[0] >= 0.3)
print()
print("--- low-resource tier only (the SFT-scarce languages) ---")
report("A_low. no filter", lambda r: r[2]=="low" and not r[4])
report("B_low. world >= 0.3", lambda r: r[2]=="low" and not r[4] and r[0]>=0.3)
report("D_low. + regen chains", lambda r: r[2]=="low" and r[0]>=0.3)
print()
n_regen_chains = sum(1 for r in rows if r[4])
regen_good = sum(r[1] for r in rows if r[4])
print(f"regen chains added: {n_regen_chains}, purity {100*regen_good/max(n_regen_chains,1):.2f}%")
for lang in ("am","sw","eu","ky","kn"):
    tot = sum(1 for r in rows if r[3]==lang and r[0]>=0.3)
    base = sum(1 for r in rows if r[3]==lang and r[0]>=0.3 and not r[4])
    print(f"  {lang}: {base} -> {tot} chains (+{100*(tot-base)/max(base,1):.1f}%)")
print("DONE")
