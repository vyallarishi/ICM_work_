"""The whole method on a toy example: 5 languages, 3 problems, runs in a second.

    python examples/walkthrough.py

Nothing is hard-coded about the outcome: every line printed is computed by the
same functions the real run uses (icm/method). Read the output next to the
code in icm/method/vote.py and icm/method/regenerate.py.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icm.method import regenerate, vote                     # noqa: E402
from icm.method.data import LANDSLIDE, WORLD_STRENGTH, norm  # noqa: E402
from icm.method.metrics import metrics                       # noqa: E402
from icm.method.run import decide                            # noqa: E402

LANGS = ["en", "de", "hi", "sw", "am"]
QUESTIONS = {"p1": "A box holds 6 eggs. How many eggs in 4 boxes?",       # gold 24
             "p2": "Tom has 50 marbles and gives away 8. How many left?",  # gold 42
             "p3": "5 pens cost 2 each. Total cost?"}                      # gold 10
GOLD = {"p1": "24", "p2": "42", "p3": "10"}

# What each language "answered", chain by chain. Real pools have ~20 chains;
# 4-5 is enough to show every kind of decision.
ANSWERS = {
    "p1": {"en": "24 24 24 24", "de": "24 24 24 30", "hi": "24 24 24 24",
           "sw": "24 24 24 24", "am": "24 24 24 20"},
    "p2": {"en": "42 42 42 42", "de": "42 42 42 42", "hi": "42 42 42 40",
           "sw": "42 42 40 40 41",            # split: nobody has 70%
           "am": "40 40 40 40"},              # confident, and wrong
    "p3": {"en": "10 10 10 10", "de": "10 10 10 10", "sw": "10 10 10 10",
           "hi": "12 12 10",                  # split, and its own majority is wrong
           "am": "12 10 10 12 15"},           # split, tie broken the wrong way
}


def make_pools():
    """Build the same dict icm.method.data.load() would produce."""
    pools = {}
    for problem, per_lang in ANSWERS.items():
        for lang, answers in per_lang.items():
            chains = [{"answer": norm(a), "text": f"[{lang}] work for {problem}: ... #### {a}"}
                      for a in answers.split()]
            counts = Counter(c["answer"] for c in chains)
            pools[f"train_{lang}_{problem}"] = {
                "lang": lang, "question": QUESTIONS[problem], "gold": GOLD[problem],
                "chains": chains, "share": counts.most_common(1)[0][1] / len(chains),
                "plurality": max(counts, key=lambda a: (counts[a], a)),
            }
    return pools


def world_of(pools, state, iid):
    """The other languages' current answers for this pool's problem."""
    return vote.World(pools).context(state, iid)


def explain_init(pools, state, iid):
    p, s = pools[iid], state[iid]
    # initialize() looks at the other languages' own majorities (pluralities)
    problem = iid.rsplit("_", 1)[-1]
    others = [q["plurality"] for j, q in pools.items()
              if j.rsplit("_", 1)[-1] == problem and q["lang"] != p["lang"]]
    top, n = Counter(others).most_common(1)[0]
    world = f"others say {dict(Counter(others))}"
    if s["kind"] == "committed":
        return f"COMMITTED {s['answer']:>3}   share {p['share']:.2f} >= {LANDSLIDE}; {world}"
    if s["answer"] is None:
        return (f"VETOED        share {p['share']:.2f} >= {LANDSLIDE} but {n}/{len(others)} "
                f"others say {top}, which {p['lang']} never produced; {world}")
    return f"CONTESTED {s['answer']:>3}  share {p['share']:.2f} < {LANDSLIDE}; {world}"


def main():
    pools = make_pools()
    print("=" * 72)
    print("STEP 1  initialize: every pool is committed, contested or vetoed")
    print("=" * 72)
    init = vote.initialize(pools)
    for problem in QUESTIONS:
        print(f"\n{problem}: {QUESTIONS[problem]}  (gold {GOLD[problem]})")
        for lang in LANGS:
            iid = f"train_{lang}_{problem}"
            print(f"  {lang:<3} {ANSWERS[problem][lang]:<16} {explain_init(pools, init, iid)}")

    print("\n" + "=" * 72)
    print(f"STEP 2  converge: contested pools re-vote for {vote.ITERATIONS} rounds")
    print("=" * 72)
    state, contested, oscillators = vote.converge(pools)
    for iid in contested:
        others = world_of(pools, state, iid)
        before, after = init[iid]["answer"], state[iid]["answer"]
        why = ("abstains: the world's answer is not in its own pool" if after is None
               else f"picks the answer most of the world holds: {dict(Counter(others))}")
        print(f"  {iid:<14} {before!s:>4} -> {after!s:<4}  {why}")
    print(f"\n  after voting: {metrics(pools, state, contested)}")

    print("\n" + "=" * 72)
    print("STEP 3  regenerate: stuck languages see 3 peer solutions and try again")
    print("=" * 72)
    jobs = regenerate.build_jobs(pools, state, contested)
    abstained = [i for i in contested if state[i]["answer"] is None]
    print(f"  abstained pools: {abstained}")
    print(f"  prompts built for: {[iid for iid, _ in jobs]}")
    iid, prompt = next((j for j in jobs if j[0] in abstained), jobs[0])
    print(f"\n  the prompt {iid} would be given (demos in its own language, then peers):")
    for line in prompt.splitlines():
        print("    | " + line)

    # On the real run the model answers here (icm.method.regenerate.generate, GPU).
    # We pretend: each stuck language re-derives the world's answer 3 times out
    # of 4 and its old answer once. (Label-free: the world's answer, not gold.)
    regen = {}
    for iid in abstained:
        world_answer = regenerate.answer_clusters(
            pools, state, iid.rsplit("_", 1)[-1], pools[iid]["lang"])[0]
        regen[iid] = [world_answer] * 3 + [pools[iid]["plurality"]]
    print(f"\n  pretend the model answered: {regen}")

    print("\n" + "=" * 72)
    print(f"STEP 4  decide again with the new chains at weight {regenerate.ALPHA}")
    print("=" * 72)
    result = decide(pools, regen)
    final_state = vote.converge(pools)[0]
    for iid in contested:
        others = world_of(pools, final_state, iid)
        new = regenerate.weighted_decide(pools[iid], others, regen.get(iid, []),
                                         regenerate.ALPHA, iid in oscillators)
        print(f"  {iid:<14} {state[iid]['answer']!s:>4} -> {new!s:<4}"
              + ("  recovered" if state[iid]["answer"] is None and new is not None else ""))
    print(f"\n  final: {result['with_regen']}")
    print("\nGold was used only in the metrics lines. Every decision above was label-free.")


if __name__ == "__main__":
    main()
