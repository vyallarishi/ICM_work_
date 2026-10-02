"""H5: how label quality scales with the size of the jury and the number of chains.

(a) JURY SIZE. For each pool the jury is k other languages, drawn at random
    per problem (5 seeds). Rule: the pipeline's first pass + one vote round
    (icm.method.vote.initialize logic, then update_pool against the jury),
    v2 rules, exact scoring. k in {1, 2, 4, 8, 12, 16, 24, 36}.
    PRE-REGISTERED: precision increases from k=1 to k=16 by >= 2 points, and
    the gain from k=24 to k=36 is < 0.5 points (diminishing returns). That
    predicts what the 13 extra languages of the 50-language run can add.
(b) CHAINS PER POOL. Each pool keeps n random chains (3 seeds), n in
    {2, 4, 8, 12, 16, all}; full pipeline voting (v2, all 37 languages).
    PRE-REGISTERED: coverage rises with n; precision at n=8 is within 1 point
    of precision with all chains (so ~8 samples per language suffice).

Gold is read only to score. Voting only (no regeneration) in both, so the
curves isolate the effect of the vote's inputs.

    python -m icm.experiments.q10_pipeline_hypotheses.h5_scaling
"""
import random
from collections import Counter

import numpy as np

from ...method import data, vote
from ...method.rules import V2
from .common import ROOT, SEED, by_problem, exact, write

KS = (1, 2, 4, 8, 12, 16, 24, 36)
NS = (2, 4, 8, 12, 16, None)


def first_pass_plurality(pools):
    return {iid: p["plurality"] for iid, p in pools.items()}


def fast_update(counts, said):
    """vote.update_pool under V2 (exact match). Answers are canonical strings from
    data.norm, so numeric equality is string equality and counting replaces the
    pairwise Decimal comparisons. Equivalence is checked in self_test()."""
    if not counts:
        return None
    if said:
        wc = Counter(said); top, n = wc.most_common(1)[0]
        if n / len(said) >= data.WORLD_STRENGTH and top not in counts:
            return None
        return max(counts, key=lambda a: (wc[a] / len(said) + 0.001 * counts[a], a))
    return max(counts, key=lambda a: (0.001 * counts[a], a))


def decide_with_jury(pool, jury_answers):
    """First pass (landslide / veto) + one cross-lingual vote round, against a given jury."""
    said = [a for a in jury_answers if a is not None]
    counts = Counter(c["answer"] for c in pool["chains"])
    if pool["share"] >= data.LANDSLIDE:
        if said:
            top, n = Counter(said).most_common(1)[0]
            if n / len(said) >= data.WORLD_STRENGTH and top != pool["plurality"]:
                return fast_update(counts, said)        # vetoed landslide -> contested treatment
        return pool["plurality"]
    return fast_update(counts, said)


def self_test():
    rng = random.Random(0); ok = True
    for _ in range(3000):
        answers = [str(rng.choice([1, 2, 3, 200, 201])) for _ in range(rng.randint(1, 8))]
        said = [str(rng.choice([1, 2, 3, 4, 200, 201])) for _ in range(rng.randint(0, 12))]
        pool = {"chains": [{"answer": a} for a in answers]}
        if fast_update(Counter(answers), said) != vote.update_pool(pool, said, V2):
            ok = False; print("FAIL", answers, said); break
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def jury_curve(pools):
    probs = by_problem(pools); plur = first_pass_plurality(pools)
    rows = {}
    for k in KS:
        prec, cov = [], []
        for seed in range(5):
            rng = random.Random(f"{SEED}-{k}-{seed}")
            answers = {}
            for prob, ids in probs.items():
                for iid in ids:
                    others = [o for o in ids if pools[o]["lang"] != pools[iid]["lang"]]
                    jury = rng.sample(others, min(k, len(others)))
                    answers[iid] = decide_with_jury(pools[iid], [plur[o] for o in jury])
            answered = [i for i, a in answers.items() if a is not None]
            cov.append(len(answered) / len(answers))
            prec.append(sum(exact(answers[i], pools[i]["gold"]) for i in answered) / len(answered))
        rows[k] = {"precision": round(float(np.mean(prec)), 4), "precision_sd": round(float(np.std(prec)), 4),
                   "coverage": round(float(np.mean(cov)), 4)}
        print(f"jury k={k:>2}: precision {100 * rows[k]['precision']:.2f} (sd {100 * rows[k]['precision_sd']:.2f})  "
              f"coverage {100 * rows[k]['coverage']:.2f}", flush=True)
    return rows


def chains_curve(full):
    rows = {}
    for n in NS:
        prec, cov = [], []
        for seed in range(3 if n else 1):
            pools = {}
            for iid, p in full.items():
                ch = list(p["chains"])
                if n and len(ch) > n:
                    ch = random.Random(f"{SEED}-{iid}-{n}-{seed}").sample(ch, n)
                pools[iid] = dict(p, chains=ch)
            pools = data.apply_diverse_views_fixed(pools)
            state, _, _ = vote.converge(pools, rules=V2)
            answered = [i for i, s in state.items() if s["answer"] is not None]
            cov.append(len(answered) / len(state))
            prec.append(sum(exact(state[i]["answer"], pools[i]["gold"]) for i in answered) / len(answered))
        key = str(n or "all")
        rows[key] = {"precision": round(float(np.mean(prec)), 4), "coverage": round(float(np.mean(cov)), 4)}
        print(f"chains n={key:>3}: precision {100 * rows[key]['precision']:.2f}  coverage {100 * rows[key]['coverage']:.2f}",
              flush=True)
    return rows


def main():
    import sys
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    if not self_test():
        sys.exit("self-test failed")
    full = data.load_pools(ROOT)
    pools = data.apply_diverse_views_fixed({k: dict(v, chains=list(v["chains"])) for k, v in full.items()})
    print("== (a) jury size")
    jury = jury_curve(pools)
    print("== (b) chains per pool")
    chains = chains_curve(full)
    a = 100 * (jury[16]["precision"] - jury[1]["precision"]); b = 100 * (jury[36]["precision"] - jury[24]["precision"])
    c = 100 * abs(chains["8"]["precision"] - chains["all"]["precision"])
    covs = [chains[str(n or "all")]["coverage"] for n in NS]
    verdict = {"jury_gain_1_to_16": round(a, 2), "jury_gain_24_to_36": round(b, 2),
               "H5a_holds": a >= 2.0 and b < 0.5,
               "chains_precision_gap_8_vs_all": round(c, 2),
               "coverage_monotone": all(x <= y + 1e-4 for x, y in zip(covs, covs[1:])),
               "H5b_holds": c <= 1.0 and all(x <= y + 1e-4 for x, y in zip(covs, covs[1:]))}
    print(f"\nH5a jury: +{a:.2f} (1->16), {b:+.2f} (24->36) -> {'HOLDS' if verdict['H5a_holds'] else 'FAILS'}")
    print(f"H5b chains: |prec(8) - prec(all)| = {c:.2f}, coverage monotone {verdict['coverage_monotone']} -> "
          f"{'HOLDS' if verdict['H5b_holds'] else 'FAILS'}")
    write("h5_scaling.json", {"jury": jury, "chains": chains, "verdict": verdict})


if __name__ == "__main__":
    main()
