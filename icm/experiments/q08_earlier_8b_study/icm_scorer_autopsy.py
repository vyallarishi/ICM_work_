"""E0 — scorer autopsy: does mutual predictability discriminate at the CHAIN level?

Walks the signal chain bottom-up on frozen exports (mono-heldout200-s200):
  1. scoring coverage: which chains were actually scored (cascade view)?
  2. margin health: how many margins are exactly zero / near zero?
  3. chain-level discrimination: margin AUC for gold vs non-gold chains
  4. aggregation test: does margin-weighted voting beat plain vote counting?
  5. label quality: do assigned positives track gold better than plurality?
Per tier throughout. Zero GPU; caveat: margins are conditioned on the final
converged labeling (the scorer's verdict in its end state), not a fixed
random context — a clean re-scoring test would need GPU (E3).
"""

from __future__ import annotations

import glob
import json
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
OUT = RUN_DIR / "e0_results.json"

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


def norm(x) -> str | None:
    try:
        d = Decimal(str(x).replace(",", ""))
        return str(d.normalize()) if d.is_finite() else None
    except (InvalidOperation, TypeError):
        return None


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = [(v, 0) for v in neg] + [(v, 1) for v in pos]
    ranked.sort(key=lambda t: t[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop][0] == ranked[index][0]:
            stop += 1
        mean_rank = (index + stop + 1) / 2
        rank_sum += mean_rank * sum(flag for _, flag in ranked[index:stop])
        index = stop
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    gold: dict[str, str] = {}
    for lang in TIER:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = norm(row.get("gold_number"))

    cols = ["instance_id", "canonical_answer", "parse_status", "score_status",
            "logp_positive", "logp_negative", "mutual_predictability_margin",
            "prompt_stability", "assigned_label"]
    chains: list[dict] = []
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=cols).to_pylist():
            answer = norm(row["canonical_answer"])
            g = gold.get(row["instance_id"])
            if row["parse_status"] != "parsed" or answer is None or g is None:
                continue
            row["answer"] = answer
            row["gold"] = g
            row["is_gold"] = answer == g
            row["tier"] = TIER["_".join(row["instance_id"].split("_")[1:-1])]
            chains.append(row)

    results: dict = {}

    # ---- 1. scoring coverage --------------------------------------------
    print("=" * 74)
    print("1. SCORING COVERAGE (cascade view: which chains were even scored?)")
    print("=" * 74)
    status = Counter(c["score_status"] for c in chains)
    print(f"score_status over {len(chains)} parsed chains: {dict(status)}")
    scored = [c for c in chains if c["score_status"] == "scored"
              and c["logp_positive"] is not None]
    cover = {}
    for tier in TIERS:
        tier_chains = [c for c in chains if c["tier"] == tier]
        tier_scored = [c for c in tier_chains if c["score_status"] == "scored"]
        gold_chains = [c for c in tier_chains if c["is_gold"]]
        gold_scored = [c for c in gold_chains if c["score_status"] == "scored"]
        cover[tier] = {
            "chains": len(tier_chains), "scored_pct": round(100 * len(tier_scored) / len(tier_chains), 1),
            "gold_chains_scored_pct": round(100 * len(gold_scored) / max(len(gold_chains), 1), 1),
        }
        print(f"  {tier:<5} scored {cover[tier]['scored_pct']}% of chains | "
              f"{cover[tier]['gold_chains_scored_pct']}% of GOLD chains")
    results["coverage"] = {"status": dict(status), "by_tier": cover}

    # ---- 2. margin health ------------------------------------------------
    print()
    print("=" * 74)
    print("2. MARGIN HEALTH (is the scorer whispering?)")
    print("=" * 74)
    margins = [c["mutual_predictability_margin"] for c in scored
               if c["mutual_predictability_margin"] is not None]
    zero = sum(m == 0.0 for m in margins)
    tiny = sum(abs(m) < 0.01 for m in margins)
    abs_margins = sorted(abs(m) for m in margins)
    health = {
        "scored_chains": len(margins),
        "margin_exactly_zero_pct": round(100 * zero / len(margins), 1),
        "abs_margin_below_0.01_pct": round(100 * tiny / len(margins), 1),
        "median_abs_margin": round(abs_margins[len(abs_margins) // 2], 4),
        "p90_abs_margin": round(abs_margins[int(0.9 * len(abs_margins))], 4),
        "mean_stability": round(st.mean(c["prompt_stability"] for c in scored
                                        if c["prompt_stability"] is not None), 3),
    }
    for key, value in health.items():
        print(f"  {key}: {value}")
    results["margin_health"] = health

    # ---- 3. chain-level discrimination ----------------------------------
    print()
    print("=" * 74)
    print("3. CHAIN-LEVEL DISCRIMINATION (margin AUC, gold vs non-gold chains)")
    print("=" * 74)
    disc = {}
    print(f"  {'slice':<8}{'n_gold':>8}{'n_other':>9}{'margin AUC':>12}{'logpP AUC':>11}")
    for tier in TIERS + ("ALL",):
        rows = [c for c in scored if tier in ("ALL", c["tier"])]
        gold_margin = [c["mutual_predictability_margin"] for c in rows if c["is_gold"]]
        rest_margin = [c["mutual_predictability_margin"] for c in rows if not c["is_gold"]]
        gold_logp = [c["logp_positive"] for c in rows if c["is_gold"]]
        rest_logp = [c["logp_positive"] for c in rows if not c["is_gold"]]
        margin_auc = auc(gold_margin, rest_margin)
        logp_auc = auc(gold_logp, rest_logp)
        disc[tier] = {"n_gold": len(gold_margin), "n_other": len(rest_margin),
                      "margin_auc": round(margin_auc, 3) if margin_auc else None,
                      "logp_positive_auc": round(logp_auc, 3) if logp_auc else None}
        print(f"  {tier:<8}{len(gold_margin):>8}{len(rest_margin):>9}"
              f"{margin_auc:>12.3f}{logp_auc:>11.3f}")
    # discrimination restricted to SPLIT pools (where it actually matters):
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for c in scored:
        by_instance[c["instance_id"]].append(c)
    split_gold, split_rest = [], []
    for rows in by_instance.values():
        answers = Counter(c["answer"] for c in rows)
        if not answers:
            continue
        top = answers.most_common(1)[0][1] / sum(answers.values())
        if top < 0.7 and any(c["is_gold"] for c in rows) and any(not c["is_gold"] for c in rows):
            split_gold += [c["mutual_predictability_margin"] for c in rows if c["is_gold"]]
            split_rest += [c["mutual_predictability_margin"] for c in rows if not c["is_gold"]]
    split_auc = auc(split_gold, split_rest)
    disc["split_pools_only"] = round(split_auc, 3) if split_auc else None
    print(f"  {'SPLIT':<8}{len(split_gold):>8}{len(split_rest):>9}{split_auc:>12.3f}"
          "   <- pools where the vote can't decide")
    results["discrimination"] = disc

    # ---- 4. aggregation test: margin-weighted vote vs plain vote --------
    print()
    print("=" * 74)
    print("4. AGGREGATION (does using the scores beat counting the votes?)")
    print("=" * 74)
    agg = {}
    for tier in TIERS + ("ALL",):
        plain = weighted = n = 0
        plain_split = weighted_split = n_split = 0
        for instance_id, rows in by_instance.items():
            if tier != "ALL" and rows[0]["tier"] != tier:
                continue
            answers = Counter(c["answer"] for c in rows)
            g = rows[0]["gold"]
            score: dict[str, float] = defaultdict(float)
            for c in rows:
                score[c["answer"]] += c["mutual_predictability_margin"] or 0.0
            plain_pick = answers.most_common(1)[0][0]
            weighted_pick = max(score, key=lambda a: score[a])
            n += 1
            plain += plain_pick == g
            weighted += weighted_pick == g
            top_share = answers.most_common(1)[0][1] / sum(answers.values())
            if top_share < 0.7:
                n_split += 1
                plain_split += plain_pick == g
                weighted_split += weighted_pick == g
        agg[tier] = {
            "n": n, "plain_vote_acc": round(100 * plain / n, 1),
            "margin_weighted_acc": round(100 * weighted / n, 1),
            "split_n": n_split,
            "split_plain_acc": round(100 * plain_split / max(n_split, 1), 1),
            "split_weighted_acc": round(100 * weighted_split / max(n_split, 1), 1),
        }
        print(f"  {tier:<5} all pools: vote {agg[tier]['plain_vote_acc']}% vs "
              f"margin-weighted {agg[tier]['margin_weighted_acc']}%   | "
              f"split pools only ({n_split}): vote {agg[tier]['split_plain_acc']}% vs "
              f"weighted {agg[tier]['split_weighted_acc']}%")
    results["aggregation"] = agg

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
