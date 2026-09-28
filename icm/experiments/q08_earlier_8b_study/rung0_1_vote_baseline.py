"""Rung 0 + Rung 1 — the stripped-down elicitation baseline, and its decision rule.

RUNG 0 (the minimal system):
  1. take each chain's parsed final answer      (frozen generation output)
  2. normalize numerically                      (2190 == 2,190 == 2190.0)
  3. count votes per problem-language pair
  4. compute the top answer's share of the vote
  5. commit the top answer iff share >= 0.70, else abstain

RUNG 1 (one added part): choose the decision rule by measurement instead of
by eye. Sweeps three label-blind statistics — top share, top-vs-runner-up
margin, normalized vote entropy — over thresholds, per tier, and reports the
full coverage/precision frontier so a rule can be pre-registered against a
coverage floor.

Nothing here calls a model. Reads only frozen exports. Evaluation vs gold is
strictly post-hoc; every decision statistic is computable without labels.

Writes rung01_results.json next to this script.
"""

from __future__ import annotations

import glob
import json
import math
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
OUT = RUN_DIR / "rung01_results.json"

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
RUNG0_THRESHOLD = 0.70


def normalize_answer(value) -> str | None:
    """Step 2: canonical numeric form, or None if unparseable/non-finite."""
    try:
        number = Decimal(str(value).replace(",", ""))
        return str(number.normalize()) if number.is_finite() else None
    except (InvalidOperation, TypeError):
        return None


def load_pools() -> tuple[dict[str, Counter], dict[str, str]]:
    """Steps 1-3: parsed answers -> normalized -> per-instance vote counts."""
    gold: dict[str, str] = {}
    for lang in TIER:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = normalize_answer(row.get("gold_number"))

    votes: dict[str, Counter] = defaultdict(Counter)
    columns = ["instance_id", "canonical_answer", "parse_status"]
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=columns).to_pylist():
            answer = normalize_answer(row["canonical_answer"])
            if row["parse_status"] != "parsed" or answer is None:
                continue
            votes[row["instance_id"]][answer] += 1
    return votes, gold


def statistics_for(counts: Counter) -> dict[str, float]:
    """Label-blind decision statistics for one pool."""
    total = sum(counts.values())
    ranked = counts.most_common()
    top_n = ranked[0][1]
    runner_n = ranked[1][1] if len(ranked) > 1 else 0
    entropy = -sum((n / total) * math.log2(n / total) for _, n in ranked)
    max_entropy = math.log2(len(ranked)) if len(ranked) > 1 else 1.0
    return {
        "share": top_n / total,
        "margin": (top_n - runner_n) / total,
        # 1 - normalized entropy: 1.0 = unanimous, 0.0 = maximally split.
        "certainty": 1.0 - (entropy / max_entropy if max_entropy > 0 else 0.0),
    }


def evaluate(items: list[dict], rule: str, threshold: float) -> dict:
    committed = [i for i in items if i["stats"][rule] >= threshold]
    correct = sum(i["top"] == i["gold"] for i in committed)
    return {
        "coverage": len(committed) / len(items),
        "precision": correct / len(committed) if committed else 0.0,
        "n_committed": len(committed),
        "n_correct": correct,
    }


def main() -> None:
    votes, gold = load_pools()

    items: list[dict] = []
    for instance_id, counts in votes.items():
        g = gold.get(instance_id)
        if g is None or not counts:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        items.append({
            "instance_id": instance_id,
            "lang": lang,
            "tier": TIER[lang],
            "gold": g,
            "top": counts.most_common(1)[0][0],
            "pool_size": sum(counts.values()),
            "stats": statistics_for(counts),
        })
    by_tier = {tier: [i for i in items if i["tier"] == tier] for tier in TIERS}
    results: dict = {"n_items": len(items),
                     "mean_pool_size": round(st.mean(i["pool_size"] for i in items), 1)}

    # ---------------- RUNG 0 -------------------------------------------
    print("=" * 76)
    print(f"RUNG 0 — vote + share >= {RUNG0_THRESHOLD:.2f}   ({len(items)} problem-language pairs)")
    print("=" * 76)
    print(f"{'tier':<6}{'n':>7}{'coverage':>11}{'precision':>11}{'abstained':>12}")
    rung0: dict = {}
    for tier in TIERS + ("ALL",):
        rows = items if tier == "ALL" else by_tier[tier]
        outcome = evaluate(rows, "share", RUNG0_THRESHOLD)
        rung0[tier] = outcome
        print(f"{tier:<6}{len(rows):>7}{100*outcome['coverage']:>10.1f}%"
              f"{100*outcome['precision']:>10.1f}%"
              f"{len(rows)-outcome['n_committed']:>12}")
    results["rung0"] = rung0

    # What the abstained bucket would score if forced — the size of the prize.
    print("\nabstained bucket (Rung 0 says 'I don't know'):")
    print(f"{'tier':<6}{'n':>7}{'forced-plurality acc':>22}")
    forced: dict = {}
    for tier in TIERS + ("ALL",):
        rows = [i for i in (items if tier == "ALL" else by_tier[tier])
                if i["stats"]["share"] < RUNG0_THRESHOLD]
        accuracy = sum(i["top"] == i["gold"] for i in rows) / len(rows) if rows else 0.0
        forced[tier] = {"n": len(rows), "forced_accuracy": round(accuracy, 4)}
        print(f"{tier:<6}{len(rows):>7}{100*accuracy:>21.1f}%")
    results["rung0_abstained_bucket"] = forced

    # ---------------- RUNG 1 -------------------------------------------
    print()
    print("=" * 76)
    print("RUNG 1 — decision-rule sweep (label-blind statistics, per tier)")
    print("=" * 76)
    rules = ("share", "margin", "certainty")
    thresholds = [round(0.05 * k, 2) for k in range(0, 21)]
    sweep: dict = {}
    for tier in TIERS + ("ALL",):
        rows = items if tier == "ALL" else by_tier[tier]
        sweep[tier] = {
            rule: [{"threshold": t, **evaluate(rows, rule, t)} for t in thresholds]
            for rule in rules
        }

    # Rule comparison at matched coverage: for each rule, the precision
    # achieved at the coverage Rung 0 delivers for that tier.
    print("precision at Rung-0-matched coverage (higher is better):")
    print(f"{'tier':<6}{'target cov':>12}" + "".join(f"{rule:>12}" for rule in rules))
    comparison: dict = {}
    for tier in TIERS + ("ALL",):
        target = rung0[tier]["coverage"]
        row = {}
        for rule in rules:
            # closest operating point at or above the target coverage
            candidates = [p for p in sweep[tier][rule] if p["coverage"] >= target - 1e-9]
            best = min(candidates, key=lambda p: p["coverage"]) if candidates else None
            row[rule] = best
        comparison[tier] = {
            rule: None if best is None else
            {"threshold": best["threshold"], "coverage": round(best["coverage"], 4),
             "precision": round(best["precision"], 4)}
            for rule, best in row.items()
        }
        print(f"{tier:<6}{100*target:>11.1f}%" + "".join(
            f"{100*row[rule]['precision']:>11.1f}%" if row[rule] else f"{'—':>12}"
            for rule in rules))
    results["rung1_matched_coverage"] = comparison

    # Frontier for the winning rule, per tier: what each threshold buys.
    print("\nshare-rule frontier (the operating points worth pre-registering):")
    print(f"{'tier':<6}" + "".join(f"{f'@{t:.2f}':>14}" for t in (0.5, 0.6, 0.7, 0.8, 0.9)))
    for tier in TIERS + ("ALL",):
        cells = []
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            point = next(p for p in sweep[tier]["share"] if abs(p["threshold"] - t) < 1e-9)
            cells.append(f"{100*point['coverage']:.0f}%/{100*point['precision']:.1f}%")
        print(f"{tier:<6}" + "".join(f"{cell:>14}" for cell in cells))
    results["rung1_sweep"] = {
        tier: {rule: [{k: round(v, 4) if isinstance(v, float) else v
                       for k, v in point.items()} for point in points]
               for rule, points in rules_map.items()}
        for tier, rules_map in sweep.items()
    }

    # Answer-normalization audit: how much would votes fragment without step 2?
    fragmented = 0
    for instance_id, counts in votes.items():
        raw = Counter()
        for answer, n in counts.items():
            raw[answer] += n
        if len(raw) != len(counts):
            fragmented += 1
    results["normalization_merged_instances"] = fragmented

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
