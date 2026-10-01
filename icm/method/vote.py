"""Step 1 of the method: commit, veto, and let the other languages vote.

Every pool ends in one of three states:

    committed   >= 70% of its own chains agree, and the other languages do not
                strongly disagree. Frozen from here on.
    contested   its own chains are split. The other languages decide (below).
    vetoed      its own chains agree, but >= 70% of the other languages agree on
                a DIFFERENT answer. Treated as contested with no answer yet.

A contested pool then re-picks its answer for 4 rounds (`converge`). The rule
(`update_pool`): among the answers this language itself produced, take the one
most other languages currently hold; local votes only break ties. If >= 70% of
the other languages agree on an answer this language never produced, the pool
abstains (answer None). Gold is never read here.
"""
from collections import Counter, defaultdict

from .data import LANDSLIDE, WORLD_STRENGTH, match
from .rules import RECORD

ITERATIONS = 4


def problem_of(iid):
    return iid.rsplit("_", 1)[-1]


def initialize(pools, rules=RECORD):
    """One pass over every pool: committed / vetoed / contested."""
    state = {}
    plur = {iid: p["plurality"] for iid, p in pools.items()}
    world = defaultdict(dict)
    for iid, p in pools.items():
        world[iid.rsplit("_", 1)[-1]][p["lang"]] = plur[iid]
    for iid, p in pools.items():
        others = [a for l, a in world[iid.rsplit("_", 1)[-1]].items()
                  if l != p["lang"]]
        top = Counter(others).most_common(1) if others else []
        strong = top and top[0][1] / len(others) >= WORLD_STRENGTH
        if p["share"] >= LANDSLIDE:
            if strong and not match(top[0][0], plur[iid], rules):
                state[iid] = {"answer": None, "kind": "contested"}  # vetoed
            else:
                state[iid] = {"answer": plur[iid], "kind": "committed"}
        else:
            state[iid] = {"answer": plur[iid], "kind": "contested"}
    return state


class World:
    """What the other languages currently answer for each problem.

    `context(state, iid)` returns the other languages' current answers for
    iid's problem, skipping iid's own language and pools with no answer.
    """

    def __init__(self, pools):
        self.by_problem = defaultdict(list)
        for iid, pool in pools.items():
            self.by_problem[iid.rsplit("_", 1)[-1]].append(iid)
        self.pools = pools

    def context(self, state, iid):
        lang = self.pools[iid]["lang"]
        out = []
        for other in self.by_problem[iid.rsplit("_", 1)[-1]]:
            if other == iid or self.pools[other]["lang"] == lang:
                continue
            answer = state[other]["answer"]
            if answer is not None:
                out.append(answer)
        return out


def in_pool(answer, counts, rules=RECORD):
    """Did this language produce `answer`? String equality in the record;
    match() under rules.veto_by_match."""
    if rules.veto_by_match:
        return any(match(answer, a, rules) for a in counts)
    return answer in counts


def update_pool(pool, others, rules=RECORD):
    """One contested pool's re-decision. Returns the new answer, or None to abstain.

    The experiments also tried adding a model-judge bonus to `score` (see
    experiments/q01_can_voting_label_data and experiments/q02_can_the_model_judge_itself). No judge ever helped, so the
    method has none.
    """
    counts = Counter(c["answer"] for c in pool["chains"])
    if not counts:
        return None
    if others:
        top_world, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) >= WORLD_STRENGTH and not in_pool(top_world, counts, rules):
            return None  # veto: strong world answer this language never produced
    def score(a):
        ws = (sum(match(o, a, rules) for o in others) / len(others)) if others else 0.0
        return ws + 0.001 * counts[a]
    return max(counts, key=lambda a: (score(a), a))


def converge(pools, world=None, rules=RECORD):
    """initialize, then 4 rounds of update_pool over the contested pools.

    Returns (state, contested_ids, oscillators). Oscillators are contested pools
    whose answer changed 2+ times; regeneration treats them more strictly.
    """
    world = world or World(pools)
    state = initialize(pools, rules)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    history = {i: [state[i]["answer"]] for i in contested}
    for _ in range(ITERATIONS):
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            new_state[iid]["answer"] = update_pool(
                pools[iid], world.context(state, iid), rules)
        state = new_state
        for iid in contested:
            history[iid].append(state[iid]["answer"])
    oscillators = {i for i in contested
                   if sum(a != b for a, b in zip(history[i], history[i][1:]))
                   >= 2}
    return state, contested, oscillators
