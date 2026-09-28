"""Deterministic diverse-8 view with an UNDISTORTED vote share.

TWO BUGS THIS REPLACES (audit findings, both confirmed on real data):

(a) SHARE DISTORTION -- the serious one.
    stage17.apply_diverse_views recomputes `share` and `plurality` from the
    8-chain view. But the view deliberately keeps ONE chain per distinct
    answer for the top groups, so it destroys the vote proportions that
    `share` exists to measure. A pool of 10x"10" + 10x"20" has true share
    0.500 (contested) but view share 0.625; larger imbalances cross the 0.70
    landslide threshold outright.
    Measured on real phaseA: 178 pools flip landslide status (157
    committed->contested, 21 the other way), 127 change plurality, contested
    count 419 -> 555, plurality accuracy 0.872 -> 0.855.
    Since `share >= 0.70` is the landslide test and `plurality` is what a
    pool commits to, this perturbs the exact axis every result is read along.

(b) ORDER DEPENDENCE.
    The seeded shuffle is reproducible but not canonical: groups are ordered
    by size alone, so equal-sized groups break ties by post-shuffle insertion
    order. Permuting the caller's chain list changes the selected chains for
    16.4% of real pools with >8 chains, and the plurality for 105 of them.

THE FIX
  * `share` and `plurality` are computed from the FULL pool, before any
    view is taken. The view governs which chains are SCORED; it must not
    govern what the pool VOTED.
  * group ordering is (-size, answer) so ties break on content, and the
    within-group pick is the shortest chain then its text, so the result is
    invariant to input order.
Everything else is unchanged: at most 8 chains, answer-diverse first, then
filled from the remainder.
"""

from __future__ import annotations

from collections import Counter, defaultdict


def diverse8_view(chains, k=8):
    """Up to k chains, answer-diverse, deterministic and order-invariant."""
    if len(chains) <= k:
        return list(chains)
    groups = defaultdict(list)
    for chain in chains:
        groups[chain["answer"]].append(chain)
    # canonical order: biggest answer-group first, ties broken by answer text
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    def canonical(group):
        return sorted(group, key=lambda c: (len(c["text"]), c["text"]))
    selected = [canonical(group)[0] for _, group in ordered[:k]]
    if len(selected) < k:
        chosen = {id(c) for c in selected}
        remainder = [c for _, group in ordered for c in canonical(group)
                     if id(c) not in chosen]
        selected.extend(remainder[:k - len(selected)])
    return selected[:k]


def apply_diverse_views(pools, k=8):
    """Attach the view WITHOUT touching the pool's true vote statistics.

    pool['chains']    -> the diverse-k view (what gets scored)
    pool['share']     -> plurality share of the FULL pool (what it voted)
    pool['plurality'] -> plurality answer of the FULL pool
    pool['full_size'] -> how many chains the pool really had
    """
    for pool in pools.values():
        full = pool["chains"]
        counts = Counter(c["answer"] for c in full)
        pool["plurality"] = max(counts, key=lambda a: (counts[a], a))
        pool["share"] = counts[pool["plurality"]] / len(full)
        pool["full_size"] = len(full)
        pool["chains"] = diverse8_view(full, k)
    return pools


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    def mk(answers):
        return [{"answer": a, "text": f"{a}-chain-{i}"}
                for i, a in enumerate(answers)]

    print("--- (a) share now reflects the FULL pool ---")
    pools = {"p": {"chains": mk(["10"] * 10 + ["20"] * 10)}}
    apply_diverse_views(pools)
    check("true 50/50 share preserved", pools["p"]["share"], 0.5)
    check("still contested at the 0.70 bar", pools["p"]["share"] < 0.70, True)
    check("view is 8 chains", len(pools["p"]["chains"]), 8)
    check("full size recorded", pools["p"]["full_size"], 20)

    landslide = {"p": {"chains": mk(["10"] * 18 + ["20"] * 2)}}
    apply_diverse_views(landslide)
    check("a real landslide is still a landslide",
          landslide["p"]["share"] >= 0.70, True)
    check("landslide share is exact", landslide["p"]["share"], 0.9)

    print("--- (b) order invariance ---")
    import random
    base = mk(["10"] * 5 + ["20"] * 5 + ["30"] * 3 + ["40"] * 3 + ["50"] * 2)
    reference = [c["text"] for c in diverse8_view(base)]
    stable = True
    for seed in range(25):
        shuffled = list(base)
        random.Random(seed).shuffle(shuffled)
        if [c["text"] for c in diverse8_view(shuffled)] != reference:
            stable = False
            break
    check("selection identical under 25 permutations", stable, True)
    # 10 equal-sized groups, k=8: two must be dropped, and WHICH two must
    # depend on the answer text, not on where they sat in the input list.
    tie = mk([a for a in ("9", "8", "7", "6", "5", "4", "3", "2", "1", "0")
              for _ in range(2)])
    picked = sorted(c["answer"] for c in diverse8_view(tie))
    check("equal-sized groups break ties by answer text",
          picked, ["0", "1", "2", "3", "4", "5", "6", "7"])
    shuffled_tie = list(tie)
    random.Random(99).shuffle(shuffled_tie)
    check("...and that choice survives shuffling the input",
          sorted(c["answer"] for c in diverse8_view(shuffled_tie)), picked)

    print("--- view still answer-diverse ---")
    many = mk([str(i % 12) for i in range(40)])
    view = diverse8_view(many)
    check("8 chains", len(view), 8)
    check("8 distinct answers", len({c["answer"] for c in view}), 8)
    check("small pools pass through", len(diverse8_view(mk(["1", "2"]))), 2)

    print("--- plurality comes from the full pool, not the view ---")
    # 6x"5" and 5 singleton answers: the view sees 5 once, the pool sees 6
    pools2 = {"p": {"chains": mk(["5"] * 6 + ["1", "2", "3", "4", "6"])}}
    apply_diverse_views(pools2)
    check("plurality is the true majority", pools2["p"]["plurality"], "5")
    check("share is 6/11", round(pools2["p"]["share"], 4), round(6 / 11, 4))

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    self_test()
