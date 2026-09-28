"""Rung 7 — (1) how bad is vote + cross-lingual consistency at 4-8 chains?
        (2) preview: route to the ICM judge when voting fails.

Part 1 (all 37 languages, zero GPU): subsample n chains per pool (seeded,
several trials), rerun the ENTIRE final pipeline on the shrunken world —
local vote -> world-check on landslides -> constrained cross-lingual
tie-break -> abstain. The reference languages are subsampled too: the whole
world lives on the same small budget. Compare against local-vote-only to see
how much the cross-lingual layer is worth as pools shrink.

Part 2 (routing preview, 6 languages: en de ru hi am sw): wherever the
pipeline still abstains at n=4, route the decision to the best judge from E3
(instructed, no demos; debiased margin = (D + D_swap)/2) and pick the
highest-margin chain's answer. CAVEAT, stated in output: E3 margins exist
only for the <=4 chains per pool that E3 sampled (up to 2 gold + 2 non-gold),
so routing coverage is partial and the margin pool is class-balanced; this is
a preview of the routing mechanism, not its final measurement.
"""

from __future__ import annotations

import glob
import gzip
import json
import random
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
E3_CHAINS = RUN_DIR / "e3_chains.jsonl.gz"
OUT = RUN_DIR / "rung7_results.json"

TIER = {
    "en": "high", "cs": "high", "af": "high", "de": "high", "vi": "high",
    "zh-tw": "high", "tr": "high", "uk": "high", "el": "high", "es": "high",
    "fr": "high", "pt": "high", "nl": "high", "pl": "high", "id": "high",
    "ms": "high",
    "hi": "mid", "th": "mid", "bn": "mid", "ru": "mid", "ar": "mid",
    "ko": "mid", "fa": "mid", "it": "mid", "bg": "mid", "ur": "mid",
    "mr": "mid", "kk": "mid", "az": "mid",
    "kn": "low", "sw": "low", "eu": "low", "am": "low", "ml": "low",
    "is": "low", "ky": "low", "ceb": "low",
}
TIERS = ("high", "mid", "low")
LANDSLIDE = 0.70
WORLD_STRENGTH = 0.70
SIZES = (4, 6, 8, 12, None)      # None = full pool
TRIALS = 8
SEED = 20260811
E3_LANGS = {"en", "de", "ru", "hi", "am", "sw"}


def norm(x):
    try:
        d = Decimal(str(x).replace(",", "").replace(" ", ""))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not d.is_finite():
        return None
    d = d.normalize()
    if d == d.to_integral_value():
        d = d.quantize(Decimal(1))
    return f"{d:f}"


def match(a, b):
    if a is None or b is None:
        return False
    x, y = Decimal(a), Decimal(b)
    if x == y:
        return True
    s = max(abs(x), abs(y))
    return s > 0 and abs(x - y) / s <= Decimal("0.005")


def load():
    gold = {}
    for lang in TIER:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = norm(row.get("gold_number"))
    pools = defaultdict(list)   # instance_id -> [answer, ...] (chain order stable)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(
                path, columns=["instance_id", "canonical_answer", "parse_status",
                               "candidate_id"]).to_pylist():
            answer = norm(row["canonical_answer"])
            if row["parse_status"] == "parsed" and answer is not None:
                pools[row["instance_id"]].append((row["candidate_id"], answer))
    for instance_id in pools:
        pools[instance_id].sort()   # deterministic order before seeded sampling
    return pools, gold


def pipeline_decisions(sub: dict[str, list[str]]) -> dict[str, dict]:
    """Run the final pipeline on subsampled pools. sub: instance -> answers."""
    plur = defaultdict(dict)
    for instance_id, answers in sub.items():
        if not answers:
            continue
        counts = Counter(answers)
        lang = "_".join(instance_id.split("_")[1:-1])
        plur[instance_id.rsplit("_", 1)[-1]][lang] = max(
            counts, key=lambda a: (counts[a], a))
    decisions = {}
    for instance_id, answers in sub.items():
        if not answers:
            decisions[instance_id] = {"answer": None, "route": "empty"}
            continue
        counts = Counter(answers)
        top = max(counts, key=lambda a: (counts[a], a))
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        others = [p for l, p in plur[problem].items() if l != lang]
        world = Counter(others).most_common(1)[0] if others else (None, 0)
        world_answer, world_n = world
        world_share = world_n / len(others) if others else 0.0
        if counts[top] / len(answers) >= LANDSLIDE:
            if (others and world_share >= WORLD_STRENGTH
                    and not match(world_answer, top)):
                decisions[instance_id] = {"answer": None, "route": "flagged"}
            else:
                decisions[instance_id] = {"answer": top, "route": "landslide"}
        else:
            if (others and world_share >= WORLD_STRENGTH
                    and world_answer in counts):
                decisions[instance_id] = {"answer": world_answer, "route": "tiebreak"}
            else:
                decisions[instance_id] = {"answer": None, "route": "abstain",
                                          "local_top": top}
    return decisions


def local_only_decisions(sub: dict[str, list[str]]) -> dict[str, dict]:
    """Rung 0 alone on the same subsample (no cross-lingual layer)."""
    decisions = {}
    for instance_id, answers in sub.items():
        if not answers:
            decisions[instance_id] = {"answer": None}
            continue
        counts = Counter(answers)
        top = max(counts, key=lambda a: (counts[a], a))
        decisions[instance_id] = {
            "answer": top if counts[top] / len(answers) >= LANDSLIDE else None}
    return decisions


def score(decisions, gold, tier_filter=None):
    committed = correct = total = 0
    for instance_id, decision in decisions.items():
        g = gold.get(instance_id)
        if g is None:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        if tier_filter and TIER[lang] != tier_filter:
            continue
        total += 1
        if decision["answer"] is not None:
            committed += 1
            correct += match(decision["answer"], g)
    return {"coverage": committed / max(total, 1),
            "precision": correct / max(committed, 1), "n": total}


def main() -> None:
    pools, gold = load()
    rng = random.Random(SEED)
    results = {"part1": {}, "part2": {}}

    print("=" * 84)
    print("PART 1 — the full pipeline on small pools (mean over "
          f"{TRIALS} seeded subsamples)")
    print("=" * 84)
    print(f"{'n':>6} | {'local vote only':^23} | {'vote + cross-lingual':^23} |"
          f" {'low tier (v+cl)':^21}")
    print(f"{'':>6} | {'cov':>10}{'prec':>10} | {'cov':>10}{'prec':>10} |"
          f" {'cov':>10}{'prec':>10}")
    for size in SIZES:
        rows = {"lo": [], "pl": [], "pl_low": []}
        n_trials = 1 if size is None else TRIALS
        for trial in range(n_trials):
            trial_rng = random.Random((SEED, size or 0, trial).__repr__())
            sub = {}
            for instance_id, chains in pools.items():
                answers = [a for _, a in chains]
                if size is not None and len(answers) > size:
                    answers = trial_rng.sample(answers, size)
                sub[instance_id] = answers
            local = local_only_decisions(sub)
            full = pipeline_decisions(sub)
            rows["lo"].append(score(local, gold))
            rows["pl"].append(score(full, gold))
            rows["pl_low"].append(score(full, gold, "low"))
        def mean(key, field):
            return st.mean(r[field] for r in rows[key])
        label = "full" if size is None else str(size)
        results["part1"][label] = {
            "local_only": {"coverage": round(mean("lo", "coverage"), 4),
                           "precision": round(mean("lo", "precision"), 4)},
            "pipeline": {"coverage": round(mean("pl", "coverage"), 4),
                         "precision": round(mean("pl", "precision"), 4)},
            "pipeline_low": {"coverage": round(mean("pl_low", "coverage"), 4),
                             "precision": round(mean("pl_low", "precision"), 4)},
        }
        print(f"{label:>6} | {100*mean('lo','coverage'):>9.1f}%"
              f"{100*mean('lo','precision'):>9.1f}% |"
              f" {100*mean('pl','coverage'):>9.1f}%{100*mean('pl','precision'):>9.1f}% |"
              f" {100*mean('pl_low','coverage'):>9.1f}%{100*mean('pl_low','precision'):>9.1f}%")

    # ---------------- PART 2: routing preview --------------------------
    print()
    print("=" * 84)
    print("PART 2 — routing preview at n=4 (6 languages; E3 judge margins)")
    print("CAVEAT: judge margins exist only for E3's sampled chains (<=4/pool,")
    print("class-balanced), so this previews the mechanism, not its final number.")
    print("=" * 84)
    margins = defaultdict(dict)   # instance -> answer -> best margin
    with gzip.open(E3_CHAINS, "rt", encoding="utf-8") as handle:
        for line in handle:
            chain = json.loads(line)
            m = chain.get("margins", {})
            if "D_instructed" not in m or "D_instructed_swap" not in m:
                continue
            value = (m["D_instructed"] + m["D_instructed_swap"]) / 2
            answer_key = chain.get("is_gold")
            # e3 file has is_gold but not the raw answer; reconstruct via gold
            g = gold.get(chain["instance_id"])
            if g is None:
                continue
            answer = g if chain["is_gold"] else f"NOTGOLD_{id(chain) % 7}"
            margins[chain["instance_id"]].setdefault("entries", []).append(
                {"is_gold": chain["is_gold"], "margin": value})

    routed_stats = Counter()
    trial_rng = random.Random((SEED, "route").__repr__())
    e3_pools = {i: p for i, p in pools.items()
                if "_".join(i.split("_")[1:-1]) in E3_LANGS}
    sub = {}
    for instance_id, chains in e3_pools.items():
        answers = [a for _, a in chains]
        if len(answers) > 4:
            answers = trial_rng.sample(answers, 4)
        sub[instance_id] = answers
    # world context for the tie-break comes from ALL 37 languages at n=4
    world_sub = {}
    for instance_id, chains in pools.items():
        answers = [a for _, a in chains]
        if len(answers) > 4:
            answers = trial_rng.sample(answers, 4)
        world_sub[instance_id] = answers
    decisions = pipeline_decisions(world_sub)

    abstained = [i for i in sub
                 if decisions.get(i, {}).get("answer") is None and gold.get(i)]
    with_margins = [i for i in abstained if margins.get(i, {}).get("entries")]
    routed_right = 0
    for instance_id in with_margins:
        entries = margins[instance_id]["entries"]
        best = max(entries, key=lambda e: e["margin"])
        routed_right += best["is_gold"]
    forced_right = sum(
        match(Counter(world_sub[i]).most_common(1)[0][0], gold[i])
        for i in with_margins if world_sub[i])
    results["part2"] = {
        "abstained_at_n4": len(abstained),
        "with_judge_margins": len(with_margins),
        "routed_accuracy": round(routed_right / max(len(with_margins), 1), 4),
        "forced_plurality_same_items": round(forced_right / max(len(with_margins), 1), 4),
    }
    print(f"abstained pools at n=4 (6 langs): {len(abstained)}")
    print(f"  of those with judge margins available: {len(with_margins)}")
    print(f"  route-to-judge accuracy:   {100*routed_right/max(len(with_margins),1):.1f}%")
    print(f"  forced local plurality on the same items: "
          f"{100*forced_right/max(len(with_margins),1):.1f}%")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
