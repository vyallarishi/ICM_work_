"""Experiments 1 & 3 on the frozen 37x200 data. Zero GPU.

EXP 1 — WORLD-CHECK ON LANDSLIDES.
The landslide tier commits 5,643 items at 93.9% -> ~344 confidently wrong
answers that nothing detects (the old unanimous-wrong problem). Test: flag a
landslide when the OTHER languages' majority disagrees with the local commit.
  * flag -> abstain   (precision/coverage trade)
  * flag -> override with the world majority IF present in the local pool
Swept over world-majority strength. Reports error capture (share of the wrong
landslides caught) and false-alarm rate (right landslides flagged).

EXP 3 — ENSEMBLE-SIZE ABLATION.
M2 currently consults all 36 other languages. How many are actually needed?
Accuracy of the split-pool tie-break with k sampled reference languages,
k in {1,2,3,5,8,12,16,24,36}, averaged over 20 seeded samples; plus a
"high-tier references only" variant to test whether reference QUALITY beats
reference QUANTITY.

Leakage discipline: the target language is never in its own reference; gold
enters only in the report.
"""

from __future__ import annotations

import glob
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
OUT = RUN_DIR / "rung6_results.json"

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
K_VALUES = (1, 2, 3, 5, 8, 12, 16, 24, 36)
TRIALS = 20
SEED = 20260810


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
    votes = defaultdict(Counter)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(
                path, columns=["instance_id", "canonical_answer", "parse_status"]
        ).to_pylist():
            answer = norm(row["canonical_answer"])
            if row["parse_status"] == "parsed" and answer is not None:
                votes[row["instance_id"]][answer] += 1
    plur = defaultdict(dict)
    for instance_id, counts in votes.items():
        if not counts:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        plur[problem][lang] = max(counts, key=lambda a: (counts[a], a))
    return votes, gold, plur


def main() -> None:
    votes, gold, plur = load()
    results: dict = {}

    landslides, splits = [], []
    for instance_id, counts in votes.items():
        g = gold.get(instance_id)
        total = sum(counts.values())
        if g is None or not total:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        top = max(counts, key=lambda a: (counts[a], a))
        others = {l: p for l, p in plur[problem].items() if l != lang}
        row = {"tier": TIER[lang], "lang": lang, "gold": g, "top": top,
               "pool": counts, "others": others}
        (landslides if counts[top] / total >= LANDSLIDE else splits).append(row)

    # ---------------- EXP 1: world-check on landslides -------------------
    print("=" * 78)
    print(f"EXP 1 — world-check on {len(landslides)} landslides "
          f"({sum(match(r['top'], r['gold']) for r in landslides)} right / "
          f"{sum(not match(r['top'], r['gold']) for r in landslides)} wrong)")
    print("=" * 78)
    exp1 = {}
    print(f"{'world>=':>8}{'flagged':>9}{'errors caught':>15}{'false alarms':>14}"
          f"{'kept prec':>11}{'override acc':>14}")
    for strength in (0.5, 0.6, 0.7, 0.8, 0.9):
        flagged, kept = [], []
        for row in landslides:
            majority = None
            if row["others"]:
                counts = Counter(row["others"].values())
                answer, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                if n / len(row["others"]) >= strength and not match(answer, row["top"]):
                    majority = answer
            (flagged if majority is not None else kept).append((row, majority))
        wrong_total = sum(not match(r["top"], r["gold"]) for r in landslides)
        caught = sum(not match(r["top"], r["gold"]) for r, _ in flagged)
        false_alarm = sum(match(r["top"], r["gold"]) for r, _ in flagged)
        kept_precision = (sum(match(r["top"], r["gold"]) for r, _ in kept)
                          / max(len(kept), 1))
        # override variant: replace flagged commit with world majority if in pool
        override_hits = override_n = 0
        for row, majority in flagged:
            candidate = next((a for a in row["pool"] if match(a, majority)), None)
            pick = candidate if candidate is not None else row["top"]
            override_n += 1
            override_hits += match(pick, row["gold"])
        exp1[strength] = {
            "flagged": len(flagged), "errors_caught": caught,
            "wrong_total": wrong_total, "false_alarms": false_alarm,
            "kept_precision": round(kept_precision, 4),
            "override_accuracy": round(override_hits / max(override_n, 1), 4),
        }
        print(f"{strength:>8.1f}{len(flagged):>9}{caught:>8}/{wrong_total:<6}"
              f"{false_alarm:>10}{100*kept_precision:>10.1f}%"
              f"{100*override_hits/max(override_n,1):>13.1f}%")
    results["exp1"] = exp1

    # tier detail at the operating point 0.7
    print("\nper-tier at world>=0.7 (flag -> abstain):")
    print(f"{'tier':<6}{'landslides':>11}{'old prec':>10}{'flagged':>9}"
          f"{'new prec':>10}{'errors caught':>15}")
    tier_block = {}
    for tier in TIERS:
        rows = [r for r in landslides if r["tier"] == tier]
        flagged = []
        kept = []
        for row in rows:
            block = None
            if row["others"]:
                counts = Counter(row["others"].values())
                answer, n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                if n / len(row["others"]) >= 0.7 and not match(answer, row["top"]):
                    block = answer
            (flagged if block is not None else kept).append(row)
        old = sum(match(r["top"], r["gold"]) for r in rows) / len(rows)
        new = sum(match(r["top"], r["gold"]) for r in kept) / max(len(kept), 1)
        caught = sum(not match(r["top"], r["gold"]) for r in flagged)
        wrong = sum(not match(r["top"], r["gold"]) for r in rows)
        tier_block[tier] = {"n": len(rows), "old_precision": round(old, 4),
                            "flagged": len(flagged), "new_precision": round(new, 4),
                            "caught": caught, "wrong": wrong}
        print(f"{tier:<6}{len(rows):>11}{100*old:>9.1f}%{len(flagged):>9}"
              f"{100*new:>9.1f}%{caught:>9}/{wrong:<6}")
    results["exp1_tiers"] = tier_block

    # ---------------- EXP 3: ensemble-size ablation ----------------------
    print()
    print("=" * 78)
    print(f"EXP 3 — ensemble size (split pools, n={len(splits)}; "
          f"{TRIALS} samples per k)")
    print("=" * 78)
    rng = random.Random(SEED)
    exp3 = {"random": {}, "high_tier_only": {}}
    for index, row in enumerate(splits):
        row["index"] = index

    def m2_accuracy(reference_langs_for) -> float:
        hits = 0
        for row in splits:
            langs = reference_langs_for(row)
            if langs:
                counts = Counter(row["others"][l] for l in langs)
                answer, _ = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
                candidate = next((a for a in row["pool"] if match(a, answer)), None)
                pick = candidate if candidate is not None else row["top"]
            else:
                pick = row["top"]
            hits += match(pick, row["gold"])
        return hits / len(splits)

    baseline = m2_accuracy(lambda row: list(row["others"]))
    print(f"all 36 references: {100*baseline:.1f}%   "
          f"(local-only Rung 0: "
          f"{100*sum(match(r['top'], r['gold']) for r in splits)/len(splits):.1f}%)")
    print(f"{'k':>4}{'random refs':>13}{'high-tier refs':>16}")
    for k in K_VALUES:
        random_scores, high_scores = [], []
        for trial in range(TRIALS):
            seed_a, seed_b = rng.random(), rng.random()

            def pick_random(row, _s=seed_a, _k=k):
                langs = sorted(row["others"])
                local = random.Random(f"{_s}:{row['index']}")
                return local.sample(langs, min(_k, len(langs)))

            def pick_high(row, _s=seed_b, _k=k):
                langs = sorted(l for l in row["others"] if TIER[l] == "high")
                local = random.Random(f"{_s}:{row['index']}")
                return local.sample(langs, min(_k, len(langs)))

            random_scores.append(m2_accuracy(pick_random))
            high_scores.append(m2_accuracy(pick_high))
        exp3["random"][k] = round(st.mean(random_scores), 4)
        exp3["high_tier_only"][k] = round(st.mean(high_scores), 4)
        print(f"{k:>4}{100*st.mean(random_scores):>12.1f}%"
              f"{100*st.mean(high_scores):>15.1f}%")
    exp3["all_36"] = round(baseline, 4)
    results["exp3"] = exp3

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
