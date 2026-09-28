"""Threshold schedules for the consensus loop (CPU, 37x1500 data).

ICM's annealing escaped local optima in a SEARCH over labelings. Our
decision rule is a per-pool argmax with no search, so there is nothing to
anneal in that sense -- but the two 0.70 thresholds (LANDSLIDE for local
commit, WORLD_STRENGTH for the veto) act as hard constraints, and those CAN
be scheduled across laps.

ARMS: flat thresholds at 0.60/0.65/0.70/0.75/0.80, plus annealed schedules
  loosen  0.80 -> 0.60 (start strict, relax: commit late)
  tighten 0.60 -> 0.80 (start permissive, harden: commit early then prune)
  split   landslide tightens while world loosens, and vice versa
PRE-REGISTERED: a schedule 'wins' only if it beats flat-0.70 on BOTH
coverage and precision. Prediction: none does; the loop converges
monotonically (changes 215->27->5->0), so there is no basin-hopping for a
schedule to exploit; loosening should trade precision for coverage.
"""
import json
from collections import Counter, defaultdict
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q04_full_scale_headline.full_scale_37x1500 import World, load
from ..q01_can_voting_label_data.consensus_loop_15lang import metrics

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
ITER = 4
pools = load(ROOT)
world = World(pools)

def initialize_t(pools, world, landslide, world_strength):
    state = {}
    plur = {i: p["plurality"] for i, p in pools.items()}
    wmap = defaultdict(dict)
    for i, p in pools.items():
        wmap[i.rsplit("_",1)[-1]][p["lang"]] = plur[i]
    for iid, p in pools.items():
        others = [a for l,a in wmap[iid.rsplit("_",1)[-1]].items() if l != p["lang"]]
        top = Counter(others).most_common(1) if others else []
        strong = bool(top) and top[0][1]/len(others) >= world_strength
        if p["share"] >= landslide:
            if strong and not match(top[0][0], plur[iid]):
                state[iid] = {"answer": None, "kind": "contested"}
            else:
                state[iid] = {"answer": plur[iid], "kind": "committed"}
        else:
            state[iid] = {"answer": plur[iid], "kind": "contested"}
    return state

def update_t(pool, others, world_strength):
    counts = Counter(c["answer"] for c in pool["chains"])
    if not counts: return None
    if others:
        tw, tn = Counter(others).most_common(1)[0]
        if tn/len(others) >= world_strength and tw not in counts:
            return None
    def score(a):
        ws = (sum(match(o,a) for o in others)/len(others)) if others else 0.0
        return ws + 0.001*counts[a]
    return max(counts, key=lambda a: (score(a), a))

def run(landslide_sched, world_sched, label):
    state = initialize_t(pools, world, landslide_sched[0], world_sched[0])
    contested = [i for i,s in state.items() if s["kind"]=="contested"]
    for lap in range(ITER):
        ws = world_sched[min(lap+1, len(world_sched)-1)]
        ns = {i: dict(s) for i,s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_t(pools[iid], world.context(state, iid), ws)
        state = ns
    m = metrics(pools, state, contested)
    print(f"{label:<34} cov {m['coverage']:.4f}  prec {m['precision']:.4f}  "
          f"contested_acc {m['contested_accuracy']:.4f}  n_contested {len(contested)}")
    return m

print("=== FLAT ===")
base = None
for t in (0.60, 0.65, 0.70, 0.75, 0.80):
    m = run([t]*5, [t]*5, f"flat {t}")
    if t == 0.70: base = m
print("\n=== ANNEALED (landslide, world) ===")
loosen = [0.80,0.75,0.70,0.65,0.60]
tighten = [0.60,0.65,0.70,0.75,0.80]
run(loosen, loosen, "loosen 0.80->0.60 (both)")
run(tighten, tighten, "tighten 0.60->0.80 (both)")
run([0.70]*5, loosen, "world loosens, landslide 0.70")
run([0.70]*5, tighten, "world tightens, landslide 0.70")
run(loosen, [0.70]*5, "landslide loosens, world 0.70")
run(tighten, [0.70]*5, "landslide tightens, world 0.70")
print(f"\nbaseline flat-0.70: cov {base['coverage']:.4f} prec {base['precision']:.4f}")
print("A schedule WINS only if it beats BOTH.")
print("DONE")
