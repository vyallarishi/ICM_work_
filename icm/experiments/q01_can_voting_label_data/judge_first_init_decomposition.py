"""Decomposition: where does the synthesis config's gain come from?
  arm A: MP-first init, judge OFF in laps      (pure ordering effect)
  arm B: vote init, judge per-lap (annealed)   (pure annealing effect)
  arm C: both                                   (the synthesis run)
Synthetic judge at p_gold=0.55, 8 trials each. Reference: vote init judge-off."""
import random, statistics as st
from collections import Counter
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .consensus_loop_15lang import initialize, world_context
from .judge_first_init_synthesis import apply_views, scheduled_update, W_VOTE, W_JUDGE, metrics

pools = apply_views(load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"))
vote_state = initialize(pools)
contested = [i for i, s in vote_state.items() if s["kind"] == "contested"]

def run(init_answers, judge_fn):
    state = {i: dict(s) for i, s in vote_state.items()}
    for iid in contested: state[iid]["answer"] = init_answers[iid]
    for lap in range(1, 7):
        wv, wj = W_VOTE[lap-1], W_JUDGE[lap-1]
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = scheduled_update(
                pools[iid], world_context(pools, state, iid),
                judge_fn(iid) if judge_fn else None, wv, wj)
        state = ns
    return metrics(pools, state, contested)

def make_judge(seed, p=0.55):
    r = random.Random(seed)
    def judge(iid):
        pool = pools[iid]
        answers = list({c["answer"] for c in pool["chains"]})
        gold = [a for a in answers if match(a, pool["gold"])]
        if gold and r.random() < p: return gold[0]
        non = [a for a in answers if not match(a, pool["gold"])] or answers
        return r.choice(non)
    return judge

vote_init = {i: vote_state[i]["answer"] for i in contested}
base = run(vote_init, None)
print(f"reference (vote init, judge off): {100*base['contested_accuracy']:.1f}% / prec {100*base['precision']:.1f}%")
for name, use_mp_init, lap_judge in (("A: MP-init only", True, False),
                                     ("B: lap-judge only", False, True),
                                     ("C: both (synthesis)", True, True)):
    accs, precs = [], []
    for t in range(8):
        j_init = make_judge(("init", t).__repr__())
        j_lap = make_judge(("lap", t).__repr__())
        init = ({iid: j_init(iid) for iid in contested} if use_mp_init else dict(vote_init))
        m = run(init, j_lap if lap_judge else None)
        accs.append(m["contested_accuracy"]); precs.append(m["precision"])
    print(f"{name:<22} final {100*st.mean(accs):.1f}% (±{100*st.pstdev(accs):.1f}) / prec {100*st.mean(precs):.1f}%")
print("DONE")
