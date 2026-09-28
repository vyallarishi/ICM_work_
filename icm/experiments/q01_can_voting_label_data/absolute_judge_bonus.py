"""Does un-normalizing the judge bonus (absolute share-points instead of
w/|jury|) restore judge value at 37 languages? Two CPU analyses on the real
diverse-8 full37 data:
 A) EXACT frontier: at the converged judge-free state, for each absolute bonus
    size b, count in-band pools (top-2 world gap <= b): how many are WRONG
    (fixable by a correct judge) vs RIGHT (breakable by a wrong judge), and
    the break-even judge accuracy that makes b net-positive.
 B) DYNAMICS sim: full loop with absolute-bonus judge at measured-quality
    brackets p_gold in {0.45, 0.60}, vs w0 and vs the old normalized formula.
"""
import random, statistics as st
from collections import Counter
from .consensus_loop_37lang import apply_diverse_views
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from .consensus_loop_15lang import initialize, world_context

pools = apply_diverse_views(gate.load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"))
base_state = initialize(pools)
contested = [i for i, s in base_state.items() if s["kind"] == "contested"]

def update(pool, others, judge_pick, bonus_abs):
    counts = Counter(c["answer"] for c in pool["chains"])
    if not counts: return None
    if others:
        tw, tn = Counter(others).most_common(1)[0]
        if tn/len(others) >= 0.70 and tw not in counts: return None
    def score(a):
        ws = (sum(match(o,a) for o in others)/len(others)) if others else 0.0
        b = bonus_abs if (judge_pick is not None and match(a, judge_pick)) else 0.0
        return ws + b + 0.001*counts[a]
    return max(counts, key=lambda a: (score(a), a))

def run(judge_fn, bonus):
    state = {i: dict(s) for i, s in base_state.items()}
    for _ in range(4):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update(pools[iid], world_context(pools, state, iid),
                                       judge_fn(iid) if judge_fn else None, bonus)
        state = ns
    ans = [i for i in contested if state[i]["answer"] is not None]
    hits = sum(match(state[i]["answer"], pools[i]["gold"]) for i in ans)
    com = [i for i,s in state.items() if s["answer"] is not None]
    p = sum(match(state[i]["answer"], pools[i]["gold"]) for i in com)/max(len(com),1)
    return state, hits/max(len(ans),1), p

w0_state, w0_acc, w0_prec = run(None, 0.0)
print(f"w0 reference: contested {100*w0_acc:.1f}%  precision {100*w0_prec:.1f}%")

print("\n=== A) exact frontier at the converged state ===")
print(f"{'bonus b':>8}{'in-band':>9}{'wrong(fixable)':>16}{'right(at risk)':>16}{'break-even judge acc':>22}")
for b in (0.014, 0.036, 0.07, 0.10, 0.14):
    n=w=r=0
    for iid in contested:
        if w0_state[iid]["answer"] is None: continue
        others = world_context(pools, w0_state, iid)
        counts = Counter(c["answer"] for c in pools[iid]["chains"])
        if not others or len(counts)<2: continue
        shares = sorted((sum(match(o,a) for o in others)/len(others) for a in counts), reverse=True)
        if shares[0]-shares[1] <= b + 1e-9:
            n+=1
            if match(w0_state[iid]["answer"], pools[iid]["gold"]): r+=1
            else: w+=1
    be = r/max(r+w,1)
    print(f"{b:>8.3f}{n:>9}{w:>16}{r:>16}{100*be:>21.1f}%")

print("\n=== B) loop dynamics with absolute-bonus synthetic judge (8 trials) ===")
def make_judge(p, seed):
    rr = random.Random(seed)
    def j(iid):
        answers = list({c["answer"] for c in pools[iid]["chains"]})
        g = [a for a in answers if match(a, pools[iid]["gold"])]
        if g and rr.random() < p: return g[0]
        non = [a for a in answers if not match(a, pools[iid]["gold"])] or answers
        return rr.choice(non)
    return j
for p in (0.45, 0.60):
    for b, label in ((0.5/36, "normalized (old)"), (0.036, "abs 0.036"), (0.07, "abs 0.07")):
        accs=[]; precs=[]
        for t in range(8):
            _, a, pr = run(make_judge(p, (p,b,t).__repr__()), b)
            accs.append(a); precs.append(pr)
        print(f"p_gold={p:.2f} {label:<18} contested {100*st.mean(accs):.1f}% (±{100*st.pstdev(accs):.1f})  prec {100*st.mean(precs):.2f}%")
print("DONE")
