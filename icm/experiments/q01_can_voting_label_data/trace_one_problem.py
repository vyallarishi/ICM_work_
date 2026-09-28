"""Full-pipeline trace for 5 languages on one real problem (full37 data).
Auto-picks a problem maximizing path diversity among {en,es,ru,am,sw}:
want a landslide-commit, a contested pool, and a veto->regen recovery."""
import json
from collections import Counter
from . import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from .consensus_loop_15lang import initialize, update_pool
from ..q04_full_scale_headline.full_scale_37x1500 import World
from ..q03_does_regeneration_help.regeneration_15lang import weighted_decide, select_regen_peers
import random
from ...paths import RUN_DIR

LANGS = ["en", "es", "ru", "am", "sw"]
pools = s17.apply_diverse_views(s17.load_pools(
    "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"))
world = World(pools)
regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
         for l in open(f"{RUN_DIR}/stage23_regen37.jsonl", encoding="utf-8")}

# converge with full history
state = initialize(pools)
contested = [i for i, s in state.items() if s["kind"] == "contested"]
hist = {i: [state[i]["answer"]] for i in contested}
for lap in range(4):
    ns = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        ns[iid]["answer"] = update_pool(pools[iid], world.context(state, iid), None, 0.0)
    state = ns
    for iid in contested: hist[iid].append(state[iid]["answer"])
osc = {i for i in contested if sum(a!=b for a,b in zip(hist[i],hist[i][1:]))>=2}
final = {i: dict(s) for i, s in state.items()}
for iid in contested:
    final[iid]["answer"] = weighted_decide(pools[iid], world.context(final, iid),
                                           regen.get(iid, []), 0.5, iid in osc)

# score problems by path diversity among our 5 langs
def paths(problem):
    kinds = set()
    for lang in LANGS:
        iid = f"train_{lang}_{problem}"
        if iid not in pools: return None
        init_kind = ("committed" if f"train_{lang}_{problem}" not in contested
                     else ("vetoed" if hist[iid][0] is None else "contested"))
        was_null = iid in contested and state[iid]["answer"] is None
        rec = was_null and final[iid]["answer"] is not None
        kinds.add((init_kind, was_null, rec))
    return kinds
problems = sorted({i.rsplit("_",1)[-1] for i in pools}, key=int)
best = max(problems, key=lambda p: (len(paths(p) or []),
           any(k[2] for k in (paths(p) or []))))
print(f"CHOSEN PROBLEM: {best}  (gold = {pools[f'train_en_{best}']['gold']})")
print(f"question (en): {pools[f'train_en_{best}']['question'][:200]}")
for lang in LANGS:
    iid = f"train_{lang}_{best}"
    p = pools[iid]
    counts = Counter(c["answer"] for c in p["chains"])
    print(f"\n--- {lang} ({iid}) ---")
    print(f"  pool (diverse-8 view): {dict(counts)}  share={p['share']:.2f} plurality={p['plurality']}")
    if iid in contested:
        others0 = [a for a in world.context(initialize(pools), iid)]
        top = Counter(others0).most_common(2)
        print(f"  init: {'VETOED' if hist[iid][0] is None else 'CONTESTED'};"
              f" world@init top: {top}")
        print(f"  lap trajectory: {hist[iid]}")
        print(f"  regen answers: {regen.get(iid, [])}")
        print(f"  final: {final[iid]['answer']}  "
              f"(correct={match(final[iid]['answer'], p['gold']) if final[iid]['answer'] else 'ABSTAIN'})")
    else:
        print(f"  init: COMMITTED (landslide, unvetoed) -> frozen at {p['plurality']} "
              f"(correct={match(p['plurality'], p['gold'])})")
# one real chain + one regen peer set for grounding
iid_am = f"train_am_{best}"
print(f"\n--- one real am chain (first 250 chars) ---")
print(pools[iid_am]["chains"][0]["text"][:250])
if iid_am in contested:
    peers = select_regen_peers(pools, state, iid_am, random.Random(0))
    print(f"\n--- first regen peer shown to am (first 200 chars) ---")
    print(peers[0][:200] if peers else "(none)")
print("\nDONE")
