"""CPU pre-run of stage15 (MP-first + annealed weights, diverse-8):
A) EXACT ordering-sensitivity: same dynamics, three initializations
   (vote, adversarial-2nd-place, random) -> when/where do they merge?
B) Monte-Carlo synthetic judge at calibrated accuracy bands (p_gold from
   stage8's contested measurements) -> predicted stage15 trajectory bands.
"""
import random, json, statistics as st
from collections import Counter, defaultdict
from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .consensus_loop_15lang import initialize, world_context
from .judge_first_init_synthesis import apply_views, scheduled_update, W_VOTE, W_JUDGE, metrics

pools = apply_views(load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"))
vote_state = initialize(pools)
contested = [i for i, s in vote_state.items() if s["kind"] == "contested"]
print(f"pools {len(pools)}, contested {len(contested)}")

def run_loop(init_answers, judge_fn, laps=6):
    state = {i: dict(s) for i, s in vote_state.items()}
    for iid in contested:
        state[iid]["answer"] = init_answers[iid]
    traj = [metrics(pools, state, contested)]
    for lap in range(1, laps + 1):
        wv, wj = W_VOTE[lap-1], W_JUDGE[lap-1]
        picks = {iid: judge_fn(iid, lap) for iid in contested}
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = scheduled_update(
                pools[iid], world_context(pools, state, iid), picks[iid], wv, wj)
        state = ns
        traj.append(metrics(pools, state, contested))
    return state, traj

def answers_of(state):
    return {i: state[i]["answer"] for i in contested}

# ---------- A) exact ordering sensitivity (judge off) ----------
inits = {}
inits["vote"] = {i: vote_state[i]["answer"] for i in contested}
adv = {}
rnd = {}
rng = random.Random(7)
for iid in contested:
    counts = Counter(c["answer"] for c in pools[iid]["chains"])
    ranked = [a for a, _ in counts.most_common()]
    adv[iid] = ranked[1] if len(ranked) > 1 else ranked[0]
    rnd[iid] = rng.choice(ranked)
inits["adversarial"], inits["random"] = adv, rnd

print("\n=== A) ordering sensitivity, judge OFF (exact, deterministic) ===")
finals = {}
for name, init in inits.items():
    state, traj = run_loop(init, lambda iid, lap: None)
    finals[name] = answers_of(state)
    print(f"init={name:<12} lap0 acc {100*traj[0]['contested_accuracy']:.1f}% "
          f"-> final {100*traj[-1]['contested_accuracy']:.1f}% "
          f"(prec {100*traj[-1]['precision']:.1f}%)")
for name in ("adversarial", "random"):
    diff = sum(1 for i in contested if finals[name][i] != finals["vote"][i])
    print(f"final answers differing from vote-init ({name}): {diff}/{len(contested)}")

# ---------- B) Monte-Carlo synthetic judge ----------
print("\n=== B) synthetic-judge stage15 simulation (p_gold scenarios) ===")
def make_judge(p_gold, seed):
    r = random.Random(seed)
    def judge(iid, lap):
        pool = pools[iid]
        answers = list({c["answer"] for c in pool["chains"]})
        gold_in = [a for a in answers if match(a, pool["gold"])]
        if gold_in and r.random() < p_gold:
            return gold_in[0]
        non = [a for a in answers if not match(a, pool["gold"])] or answers
        return r.choice(non)
    return judge

for p_gold in (0.40, 0.55, 0.70):
    init_accs, final_accs, final_precs = [], [], []
    for trial in range(8):
        judge = make_judge(p_gold, (p_gold, trial).__repr__())
        # MP-first init: the judge proposes (one draw per pool)
        mp_init = {iid: judge(iid, 0) for iid in contested}
        state, traj = run_loop(mp_init, judge)
        init_accs.append(traj[0]["contested_accuracy"])
        final_accs.append(traj[-1]["contested_accuracy"])
        final_precs.append(traj[-1]["precision"])
    print(f"p_gold={p_gold:.2f}: MP-init acc {100*st.mean(init_accs):.1f}% "
          f"-> final contested {100*st.mean(final_accs):.1f}% "
          f"(± {100*st.pstdev(final_accs):.1f}), "
          f"precision {100*st.mean(final_precs):.1f}%")
print("\nreference: vote-init judge-off final above; stage13 GPU run = vote-init real judge")
print("DONE")
