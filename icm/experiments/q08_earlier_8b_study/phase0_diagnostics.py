"""Phase 0 diagnostics on the frozen mono-heldout200-s200 export.

D1  Oracle ceiling      — was gold ever in the pool? (pool vs selection failure)
D2  Correlated errors   — do languages err with the SAME wrong answer? (+ permutation null)
D3  Confidence calib.   — reliability of selection_confidence; recalibration frontier
D4  Signal autopsy      — does ANY recorded feature separate wrong from right commitments?

Reads only frozen parquet/jsonl. No GPU, no model, no generation stack.
Writes phase0_results.json next to this script and prints a full report.
"""

from __future__ import annotations

import glob
import json
import math
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
OUT = RUN_DIR / "phase0_results.json"

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


def lang_of(instance_id: str) -> str:
    return "_".join(instance_id.split("_")[1:-1])


def load_gold() -> dict[str, str]:
    gold: dict[str, str] = {}
    for lang in TIER:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = norm(row.get("gold_number"))
    return gold


def load_committed() -> dict[str, dict]:
    """instance_id -> {answer, seed_agreement, runner_up, problem_energy}."""
    committed: dict[str, dict] = {}
    for path in glob.glob(f"{RUN}/export/**/answers.parquet", recursive=True):
        for row in pq.read_table(path).to_pylist():
            if row["level"] == "global" or not row["instance_id"]:
                continue
            committed[row["instance_id"]] = {
                "answer": norm(row["answer"]),
                "seed_agreement": row["seed_agreement"],
                "runner_up": norm(row["runner_up_answer"]),
                "problem_energy": row["problem_energy"],
            }
    return committed


BROAD_COLS = [
    "problem_id", "instance_id", "canonical_answer", "assigned_label",
    "parse_status", "selection_confidence", "mutual_predictability_margin",
    "prompt_stability", "arithmetic_valid", "arithmetic_invalid",
    "local_seed_agreement", "chain_text",
]


def load_pools() -> dict[str, list[dict]]:
    pools: dict[str, list[dict]] = defaultdict(list)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        table = pq.read_table(path, columns=BROAD_COLS)
        for row in table.to_pylist():
            row["canonical_answer"] = norm(row["canonical_answer"])
            row["chain_len"] = len(row.pop("chain_text") or "")
            pools[row["instance_id"]].append(row)
    return pools


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(random pos > random neg), ties = 0.5. pos=wrong, neg=right."""
    if not pos or not neg:
        return None
    ranked = sorted((v, 0) for v in neg) + [(v, 1) for v in pos]
    ranked.sort(key=lambda t: t[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop][0] == ranked[index][0]:
            stop += 1
        mean_rank = (index + stop + 1) / 2  # 1-based average rank of tie block
        rank_sum += mean_rank * sum(flag for _, flag in ranked[index:stop])
        index = stop
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    gold = load_gold()
    committed = load_committed()
    pools = load_pools()
    results: dict = {}

    # Per-instance outcome table used by every section.
    instances = []
    for instance_id, pool in pools.items():
        g = gold.get(instance_id)
        if g is None:
            continue
        parsed = [c for c in pool if c["parse_status"] == "parsed"
                  and c["canonical_answer"] is not None]
        answers = Counter(c["canonical_answer"] for c in parsed)
        com = committed.get(instance_id, {})
        answer = com.get("answer")
        outcome = ("abstain" if answer is None
                   else "correct" if answer == g else "wrong")
        instances.append({
            "id": instance_id,
            "lang": lang_of(instance_id),
            "tier": TIER[lang_of(instance_id)],
            "problem": instance_id.rsplit("_", 1)[-1],
            "gold": g,
            "pool": parsed,
            "answers": answers,
            "gold_in_pool": g in answers,
            "gold_share": answers[g] / max(sum(answers.values()), 1),
            "plurality": answers.most_common(1)[0][0] if answers else None,
            "committed": answer,
            "outcome": outcome,
            "seed_agreement": com.get("seed_agreement"),
            "runner_up": com.get("runner_up"),
            "problem_energy": com.get("problem_energy"),
        })

    # ---------------- D1: oracle ceiling ----------------
    print("=" * 72)
    print("D1 — ORACLE CEILING (was gold ever in the pool?)")
    print("=" * 72)
    d1 = {}
    header = (f"{'tier':<6}{'n':>6}{'gold-in-pool':>13}{'wrong:pool':>11}"
              f"{'wrong:select':>13}{'abst:pool':>10}{'abst:select':>12}")
    print(header)
    for tier in TIERS + ("ALL",):
        rows = [i for i in instances if tier in ("ALL", i["tier"])]
        n = len(rows)
        ceiling = sum(i["gold_in_pool"] for i in rows)
        wrong = [i for i in rows if i["outcome"] == "wrong"]
        abst = [i for i in rows if i["outcome"] == "abstain"]
        wrong_pool = sum(not i["gold_in_pool"] for i in wrong)
        abst_pool = sum(not i["gold_in_pool"] for i in abst)
        d1[tier] = {
            "n": n, "gold_in_pool": ceiling,
            "wrong": len(wrong), "wrong_pool_failure": wrong_pool,
            "wrong_selection_failure": len(wrong) - wrong_pool,
            "abstain": len(abst), "abstain_pool_failure": abst_pool,
            "abstain_selection_missed": len(abst) - abst_pool,
            "mean_gold_share_when_present": round(st.mean(
                [i["gold_share"] for i in rows if i["gold_in_pool"]] or [0]), 3),
        }
        print(f"{tier:<6}{n:>6}{100*ceiling/max(n,1):>12.1f}%"
              f"{wrong_pool:>8}/{len(wrong):<3}"
              f"{len(wrong)-wrong_pool:>9}/{len(wrong):<3}"
              f"{abst_pool:>7}/{len(abst):<4}"
              f"{len(abst)-abst_pool:>7}/{len(abst):<4}")
    results["D1"] = d1

    # ---------------- D2: correlated errors ----------------
    print()
    print("=" * 72)
    print("D2 — CORRELATED-ERROR CENSUS (same wrong answer across languages?)")
    print("=" * 72)
    d2 = {}
    for level, key in (("pool-plurality", "plurality"), ("committed", "committed")):
        wrong_by_problem: dict[str, dict[str, str]] = defaultdict(dict)
        for i in instances:
            value = i[key]
            if value is not None and value != i["gold"]:
                wrong_by_problem[i["problem"]][i["lang"]] = value
        pair_total = pair_same = 0
        share_max = []
        for langs in wrong_by_problem.values():
            values = list(langs.values())
            if len(values) < 2:
                continue
            counts = Counter(values)
            share_max.append(counts.most_common(1)[0][1] / len(values))
            for a in range(len(values)):
                for b in range(a + 1, len(values)):
                    pair_total += 1
                    pair_same += values[a] == values[b]
        observed = pair_same / max(pair_total, 1)

        # Permutation null: shuffle each language's wrong answers across the
        # problems where that language was wrong (preserves per-language error
        # counts and answer distribution, destroys per-problem alignment).
        rng = random.Random(20260805)
        by_lang: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for problem, langs in wrong_by_problem.items():
            for lang, value in langs.items():
                by_lang[lang].append((problem, value))
        null_rates = []
        for _ in range(200):
            shuffled: dict[str, dict[str, str]] = defaultdict(dict)
            for lang, pairs in by_lang.items():
                values = [v for _, v in pairs]
                rng.shuffle(values)
                for (problem, _), value in zip(pairs, values):
                    shuffled[problem][lang] = value
            total = same = 0
            for langs in shuffled.values():
                values = list(langs.values())
                for a in range(len(values)):
                    for b in range(a + 1, len(values)):
                        total += 1
                        same += values[a] == values[b]
            null_rates.append(same / max(total, 1))
        null_mean = st.mean(null_rates)
        null_hi = sorted(null_rates)[int(0.99 * len(null_rates))]
        d2[level] = {
            "wrong_pairs": pair_total, "same_answer_rate": round(observed, 4),
            "null_mean": round(null_mean, 4), "null_p99": round(null_hi, 4),
            "lift_over_null": round(observed / max(null_mean, 1e-9), 2),
            "mean_max_share_per_problem": round(st.mean(share_max), 3) if share_max else None,
            "problems_with_multi_lang_errors": len(share_max),
        }
        print(f"[{level}] wrong-language pairs sharing the SAME wrong answer: "
              f"{observed:.1%}  (null {null_mean:.1%}, p99 {null_hi:.1%}, "
              f"lift {observed/max(null_mean,1e-9):.1f}x)")
        if share_max:
            print(f"          per-problem dominant wrong answer covers "
                  f"{st.mean(share_max):.1%} of erring languages "
                  f"({len(share_max)} problems with >=2 erring languages)")
    results["D2"] = d2

    # ---------------- D3: confidence calibration ----------------
    print()
    print("=" * 72)
    print("D3 — CONFIDENCE CALIBRATION (selection_confidence of positives)")
    print("=" * 72)
    d3 = {}
    scored = []
    for i in instances:
        if i["outcome"] == "abstain":
            continue
        positives = [c for c in i["pool"] if c["assigned_label"] == 1]
        conf = max((c["selection_confidence"] or 0) for c in positives) if positives else 0.0
        scored.append((conf, i["outcome"] == "correct", i["tier"]))
    scored.sort(key=lambda t: t[0])
    print(f"{'decile':<8}{'conf range':>20}{'n':>7}{'precision':>11}")
    deciles = []
    for k in range(10):
        chunk = scored[k * len(scored) // 10:(k + 1) * len(scored) // 10]
        if not chunk:
            continue
        precision = sum(c for _, c, _ in chunk) / len(chunk)
        deciles.append({
            "lo": round(chunk[0][0], 4), "hi": round(chunk[-1][0], 4),
            "n": len(chunk), "precision": round(precision, 4),
        })
        print(f"{k+1:<8}{chunk[0][0]:>9.3f}-{chunk[-1][0]:<9.3f}"
              f"{len(chunk):>6}{precision:>10.1%}")
    d3["deciles"] = deciles
    d3["max_confidence_seen"] = round(scored[-1][0], 4) if scored else None
    # Frontier: precision at coverage if we re-thresholded on this ordering
    # (committed set only; abstentions are already out).
    frontier = []
    for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        chunk = scored[int((1 - keep) * len(scored)):]
        if chunk:
            frontier.append({"keep_frac": keep,
                             "precision": round(sum(c for _, c, _ in chunk) / len(chunk), 4)})
    d3["frontier"] = frontier
    print("re-threshold frontier (drop lowest-confidence committed answers):")
    for f in frontier:
        print(f"  keep {f['keep_frac']:.0%} of committed -> precision {f['precision']:.1%}")
    results["D3"] = d3

    # ---------------- D4: signal autopsy ----------------
    print()
    print("=" * 72)
    print("D4 — SIGNAL AUTOPSY (wrong vs right commitments; AUC per feature)")
    print("=" * 72)

    def features(i: dict) -> dict[str, float | None]:
        pool = i["pool"]
        positives = [c for c in pool if c["assigned_label"] == 1]
        supporting = [c for c in pool if c["canonical_answer"] == i["committed"]]
        counts = i["answers"]
        total = sum(counts.values())
        entropy = -sum((v / total) * math.log2(v / total) for v in counts.values()) if total else 0.0
        try:
            committed_decimal = Decimal(i["committed"])
            roundness = float(committed_decimal % 10 == 0)
            is_integer = float(committed_decimal == committed_decimal.to_integral_value())
        except (InvalidOperation, TypeError):
            roundness = is_integer = None
        def agg(rows, field, fn=st.mean):
            values = [r[field] for r in rows if r[field] is not None]
            return fn(values) if values else None
        return {
            "pool_size": total,
            "n_distinct_answers": len(counts),
            "pool_entropy": entropy,
            "committed_share_of_pool": len(supporting) / max(total, 1),
            "seed_agreement": i["seed_agreement"],
            "max_positive_confidence": agg(positives, "selection_confidence", max),
            "mean_positive_margin": agg(positives, "mutual_predictability_margin"),
            "mean_positive_stability": agg(positives, "prompt_stability"),
            "supporting_arithmetic_invalid": agg(supporting, "arithmetic_invalid", sum),
            "supporting_arithmetic_valid": agg(supporting, "arithmetic_valid", sum),
            "mean_supporting_chain_len": agg(supporting, "chain_len"),
            "problem_energy": i["problem_energy"],
            "answer_is_round": roundness,
            "answer_is_integer": is_integer,
        }

    wrong_rows = [features(i) for i in instances if i["outcome"] == "wrong"]
    right_rows = [features(i) for i in instances if i["outcome"] == "correct"]
    d4 = {"n_wrong": len(wrong_rows), "n_right": len(right_rows), "features": {}}
    print(f"wrong commitments: {len(wrong_rows)} | correct: {len(right_rows)}")
    print(f"{'feature':<32}{'AUC(wrong>right)':>17}{'wrong mean':>12}{'right mean':>12}")
    for name in wrong_rows[0]:
        wrong_values = [r[name] for r in wrong_rows if r[name] is not None]
        right_values = [r[name] for r in right_rows if r[name] is not None]
        score = auc(wrong_values, right_values)
        d4["features"][name] = {
            "auc": round(score, 3) if score is not None else None,
            "wrong_mean": round(st.mean(wrong_values), 4) if wrong_values else None,
            "right_mean": round(st.mean(right_values), 4) if right_values else None,
        }
        if score is not None:
            print(f"{name:<32}{score:>13.3f}    "
                  f"{st.mean(wrong_values):>11.3f} {st.mean(right_values):>11.3f}")

    # Runner-up check: when wrong, was gold sitting in second place?
    wrong_instances = [i for i in instances if i["outcome"] == "wrong"]
    runner_gold = sum(i["runner_up"] == i["gold"] for i in wrong_instances
                      if i["runner_up"] is not None)
    unanimous_wrong = sum(i["seed_agreement"] == 1.0 for i in wrong_instances)
    d4["wrong_with_gold_as_runner_up"] = runner_gold
    d4["unanimous_wrong"] = unanimous_wrong
    print(f"\nwrong commitments where gold was the RUNNER-UP: "
          f"{runner_gold}/{len(wrong_instances)}")
    print(f"wrong commitments with perfect seed agreement (unanimous): "
          f"{unanimous_wrong}/{len(wrong_instances)}")
    results["D4"] = d4

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
