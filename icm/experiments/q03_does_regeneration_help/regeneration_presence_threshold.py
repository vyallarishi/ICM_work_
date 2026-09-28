"""Veto-dissolution threshold sweep (CPU, cached regen answers, both scales).
Current rule: ANY regenerated occurrence of an answer makes it 'present'
(T=1). Sweep T=1..4: an answer NOT in the original pool only counts as
present (for veto dissolution and voting) if regenerated >= T times.
PRE-REGISTERED PICK: smallest T keeping >=90% of T=1's recoveries at
recovery precision >= 97%, without new harm."""
import json
from collections import Counter
from ..q01_can_voting_label_data import consensus_loop_37lang as s17
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context, metrics
from .regeneration_15lang import converge_w0
from ...paths import RUN_DIR

WORLD_STRENGTH = 0.70

def decide_T(pool, others, regen_answers, alpha, unanimity_required, T):
    orig = Counter(c["answer"] for c in pool["chains"])
    rc = Counter(regen_answers)
    counts = Counter()
    for a, n in orig.items():
        counts[a] += float(n)
    for a, n in rc.items():
        if a in orig or n >= T:
            counts[a] += alpha * n
    if not counts:
        return None
    if unanimity_required and regen_answers and len(rc) > 1:
        counts = Counter({a: float(n) for a, n in orig.items()})
        if not counts:
            return None
    if others:
        top_world, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) >= WORLD_STRENGTH and top_world not in counts:
            return None
    def score(a):
        ws = (sum(match(o, a) for o in others) / len(others)) if others else 0.0
        return ws + 0.001 * counts[a]
    return max(counts, key=lambda a: (score(a), a))

CONFIGS = {
    "15lang": ("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA",
               "stage22_regen.jsonl", False),
    "37lang": ("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37",
               "stage23_regen37.jsonl", True),
}
for key, (root, cache, diverse) in CONFIGS.items():
    pools = s17.load_pools(root)
    if diverse:
        pools = s17.apply_diverse_views(pools)
    state, contested, osc = converge_w0(pools)
    regen = {json.loads(l)["iid"]: json.loads(l)["answers"]
             for l in open(f"{RUN_DIR}/{cache}", encoding="utf-8")}
    abst = [i for i in contested if state[i]["answer"] is None]
    print(f"\n===== {key}: baseline {metrics(pools, state, contested)}")
    print(f"{'T':>2}{'recovered':>10}{'correct':>9}{'rec prec':>10}"
          f"{'wrong commits':>15}{'corr withdrawn':>16}{'coverage':>10}{'precision':>11}")
    for T in (1, 2, 3, 4):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            others = world_context(pools, ns, iid)
            ns[iid]["answer"] = decide_T(pools[iid], others,
                                         regen.get(iid, []), 0.5,
                                         iid in osc, T)
        m = metrics(pools, ns, contested)
        rec = [i for i in abst if ns[i]["answer"] is not None]
        rec_ok = sum(match(ns[i]["answer"], pools[i]["gold"]) for i in rec)
        wrong = len(rec) - rec_ok
        withdrawn = 0
        for iid in contested:
            if state[iid]["answer"] is None: continue
            was_ok = match(state[iid]["answer"], pools[iid]["gold"])
            now = ns[iid]["answer"]
            if was_ok and now is None: withdrawn += 1
            elif was_ok and not match(now, pools[iid]["gold"]): wrong += 1
        print(f"{T:>2}{len(rec):>10}{rec_ok:>9}"
              f"{rec_ok/max(len(rec),1):>10.4f}{wrong:>15}{withdrawn:>16}"
              f"{m['coverage']:>10.4f}{m['precision']:>11.4f}")
print("\nDONE")
