"""Rung 5 — the Anna Karenina test: is the CORRECT computation the one that is
replicated tightly across languages, while wrong ones are loose copies?

Why this is different from Rung 4 (which failed): Rung 4 asked "which local
candidate matches what other languages computed?" — a MATCHING question, and
the reference was contaminated (wrong 32% on hard problems). But Rung 3 left
an unexploited asymmetry:

    both-right pairs   Jaccard 0.859   <- one way to be right, copied exactly
    both-wrong pairs   Jaccard 0.561   <- errors correlate, but LOOSELY

Correct computations are near-identical across languages; wrong ones agree on
the answer but vary in how they get there. So instead of matching against a
majority, score each candidate answer by the COHERENCE of its cross-lingual
support: mean pairwise structural similarity among supporting chains from
DIFFERENT languages, and the BREADTH of that support. If the asymmetry holds
at the candidate level, the gold answer's support is tighter than a shared
wrong answer's support — even when the wrong answer has more votes.

Three measurements, all frozen data, zero GPU:
  M1  signal existence: AUC of coherence / breadth for gold vs non-gold
      candidates on split pools (leakage-safe: target language excluded)
  M2  the simple control Rung 4 skipped: tie-break by cross-lingual ANSWER
      majority alone (no structure) — must be reported so any M3 gain is
      attributable to structure, not to borrowing votes
  M3  decision rule: pick the local candidate with the highest
      coherence x breadth; fall back to Rung 0 when evidence is thin

Usage: --self-test | --dry-run | (full)
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .rung3_reasoning_structure import (
    SPLIT_THRESHOLD,
    TIER,
    TIERS,
    answers_match,
    jaccard,
    normalize_number,
    plurality_answer,
    signature,
)
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
OUT = RUN_DIR / "rung5_results.json"

SIGNATURE_KIND = "values"
MAX_PAIRS = 120            # sampled cross-language pairs per candidate
MIN_SUPPORT_LANGS = 2      # a candidate needs support from >=2 other languages
MIN_REFERENCE_LANGS = 3    # else fall back to Rung 0
RNG_SEED = 20260809


def coherence_and_breadth(supporters: list[tuple[str, frozenset[str]]],
                          rng: random.Random) -> tuple[float | None, int]:
    """supporters: [(lang, signature)]. Coherence = mean Jaccard over sampled
    pairs from DIFFERENT languages; breadth = number of distinct languages."""
    langs = {lang for lang, _ in supporters}
    cross = [(a, b) for i, a in enumerate(supporters)
             for b in supporters[i + 1:] if a[0] != b[0]]
    if not cross:
        return None, len(langs)
    if len(cross) > MAX_PAIRS:
        cross = rng.sample(cross, MAX_PAIRS)
    return st.mean(jaccard(a[1], b[1]) for a, b in cross), len(langs)


def candidate_scores(all_chains: list[dict], exclude_lang: str,
                     candidates: set[str], rng: random.Random) -> dict[str, dict]:
    """Cross-lingual coherence/breadth for each candidate answer, computed
    ONLY from chains outside the target language (leakage discipline)."""
    supporters: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for chain in all_chains:
        if chain["lang"] == exclude_lang or not chain["signature"]:
            continue
        if chain["answer"] in candidates:
            supporters[chain["answer"]].append((chain["lang"], chain["signature"]))
    scores: dict[str, dict] = {}
    for answer, rows in supporters.items():
        coherence, breadth = coherence_and_breadth(rows, rng)
        scores[answer] = {"coherence": coherence, "breadth": breadth,
                          "n_chains": len(rows)}
    return scores


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = [(v, 0) for v in neg] + [(v, 1) for v in pos]
    ranked.sort(key=lambda t: t[0])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop][0] == ranked[index][0]:
            stop += 1
        rank_sum += (index + stop + 1) / 2 * sum(f for _, f in ranked[index:stop])
        index = stop
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def decide(local_counts: Counter, scores: dict[str, dict],
           n_reference_langs: int) -> dict:
    """M3 rule: coherence x breadth-fraction over local candidates."""
    fallback = max(local_counts, key=lambda a: (local_counts[a], a))
    if n_reference_langs < MIN_REFERENCE_LANGS:
        return {"answer": fallback, "route": "no_reference", "changed": False}
    ranked = []
    for answer in local_counts:
        block = scores.get(answer)
        if (block is None or block["coherence"] is None
                or block["breadth"] < MIN_SUPPORT_LANGS):
            continue
        score = block["coherence"] * (block["breadth"] / n_reference_langs)
        ranked.append((answer, score))
    if not ranked:
        return {"answer": fallback, "route": "no_evidence", "changed": False}
    ranked.sort(key=lambda row: (-row[1], row[0]))
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] <= 0:
        return {"answer": fallback, "route": "indecisive", "changed": False}
    winner, best = ranked[0]
    margin = best - ranked[1][1] if len(ranked) > 1 else best
    return {"answer": winner, "route": "structural", "changed": winner != fallback,
            "margin": round(margin, 4)}


# ------------------------------------------------------------ data loading

def load(languages: set[str] | None = None):
    import pyarrow.parquet as pq

    keep = languages or set(TIER)
    gold: dict[str, str] = {}
    for lang in keep:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = normalize_number(row.get("gold_number"))

    by_problem: dict[str, list[dict]] = defaultdict(list)
    pools: dict[str, Counter] = defaultdict(Counter)
    columns = ["instance_id", "canonical_answer", "parse_status", "chain_text"]
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=columns).to_pylist():
            lang = "_".join(row["instance_id"].split("_")[1:-1])
            answer = normalize_number(row["canonical_answer"])
            if lang not in keep or row["parse_status"] != "parsed" or answer is None:
                continue
            problem = row["instance_id"].rsplit("_", 1)[-1]
            by_problem[problem].append({
                "lang": lang, "answer": answer,
                "signature": signature(row["chain_text"], SIGNATURE_KIND),
            })
            pools[row["instance_id"]][answer] += 1
    return by_problem, pools, gold


# ------------------------------------------------------------------ main

def run(dry: bool = False) -> None:
    rng = random.Random(RNG_SEED)
    by_problem, pools, gold = load()
    print(f"problems {len(by_problem)}, pools {len(pools)}")

    # per-language plurality per problem, for the M2 control
    plur_by_problem: dict[str, dict[str, str]] = defaultdict(dict)
    for instance_id, counts in pools.items():
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        plur_by_problem[problem][lang] = max(counts, key=lambda a: (counts[a], a))

    items: list[dict] = []
    m1_gold, m1_rest = defaultdict(list), defaultdict(list)
    for instance_id, counts in pools.items():
        g = gold.get(instance_id)
        total = sum(counts.values())
        if g is None or not total:
            continue
        if counts.most_common(1)[0][1] / total >= SPLIT_THRESHOLD:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        chains = by_problem[problem]
        n_reference = len({c["lang"] for c in chains if c["lang"] != lang})
        scores = candidate_scores(chains, lang, set(counts), rng)

        # ---- M1: does coherence/breadth separate gold from non-gold?
        for answer, block in scores.items():
            if block["coherence"] is None:
                continue
            bucket = m1_gold if answers_match(answer, g) else m1_rest
            bucket["coherence"].append(block["coherence"])
            bucket["breadth"].append(block["breadth"] / max(n_reference, 1))
            bucket["product"].append(
                block["coherence"] * block["breadth"] / max(n_reference, 1))

        # ---- M2 control: cross-lingual answer majority, no structure
        other_plur = [p for l, p in plur_by_problem[problem].items() if l != lang]
        majority = (Counter(other_plur).most_common(1)[0][0]
                    if other_plur else None)
        rung0_pick = max(counts, key=lambda a: (counts[a], a))
        m2_pick = majority if (majority is not None and majority in counts) else rung0_pick

        items.append({
            "instance_id": instance_id, "tier": TIER[lang], "gold": g,
            "rung0": rung0_pick, "m2": m2_pick,
            "decision": decide(counts, scores, n_reference),
        })

    results: dict = {"n_split_pools": len(items)}

    print()
    print("=" * 78)
    print("M1 — signal existence: gold vs non-gold candidates (split pools)")
    print("=" * 78)
    m1: dict = {}
    for feature in ("coherence", "breadth", "product"):
        score = auc(m1_gold[feature], m1_rest[feature])
        m1[feature] = {
            "auc": round(score, 3) if score is not None else None,
            "gold_mean": round(st.mean(m1_gold[feature]), 3) if m1_gold[feature] else None,
            "rest_mean": round(st.mean(m1_rest[feature]), 3) if m1_rest[feature] else None,
            "n_gold": len(m1_gold[feature]), "n_rest": len(m1_rest[feature]),
        }
        print(f"  {feature:<11} AUC {score:.3f}   gold mean "
              f"{m1[feature]['gold_mean']}   non-gold mean {m1[feature]['rest_mean']}")
    results["m1"] = m1

    print()
    print("=" * 78)
    print("M2/M3 — head-to-head on identical split pools")
    print("=" * 78)
    print(f"{'tier':<6}{'n':>7}{'Rung0':>9}{'M2 answer-maj':>15}{'M3 coherence':>14}")
    for tier in TIERS + ("ALL",):
        rows = [i for i in items if tier in ("ALL", i["tier"])]
        if not rows:
            continue
        base = sum(answers_match(i["rung0"], i["gold"]) for i in rows) / len(rows)
        m2 = sum(answers_match(i["m2"], i["gold"]) for i in rows) / len(rows)
        m3 = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in rows) / len(rows)
        results[tier] = {"n": len(rows), "rung0": round(base, 4),
                         "m2_answer_majority": round(m2, 4),
                         "m3_coherence": round(m3, 4)}
        print(f"{tier:<6}{len(rows):>7}{100*base:>8.1f}%{100*m2:>14.1f}%"
              f"{100*m3:>13.1f}%")

    for label, key in (("M2", "m2"),):
        changed = [i for i in items if i[key] != i["rung0"]]
        now = sum(answers_match(i[key], i["gold"]) for i in changed)
        was = sum(answers_match(i["rung0"], i["gold"]) for i in changed)
        results[f"{key}_changed"] = {"n": len(changed), "now": now, "was": was}
        print(f"\n{label} changed {len(changed)} decisions: {was} right -> {now} right "
              f"(net {now-was:+d})")
    changed = [i for i in items if i["decision"]["changed"]]
    now = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in changed)
    was = sum(answers_match(i["rung0"], i["gold"]) for i in changed)
    results["m3_changed"] = {"n": len(changed), "now": now, "was": was}
    print(f"M3 changed {len(changed)} decisions: {was} right -> {now} right "
          f"(net {now-was:+d})")
    routes = Counter(i["decision"]["route"] for i in items)
    results["m3_routes"] = dict(routes)
    print(f"M3 routes: {dict(routes)}")

    if not dry:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {OUT}")


# --------------------------------------------------------------- self-test

def self_test() -> None:
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    rng = random.Random(0)
    print("--- coherence_and_breadth ---")
    tight = [("en", frozenset({"a", "b"})), ("de", frozenset({"a", "b"})),
             ("hi", frozenset({"a", "b"}))]
    coherence, breadth = coherence_and_breadth(tight, rng)
    check("identical cross-lingual support -> coherence 1.0", coherence, 1.0)
    check("breadth counts languages", breadth, 3)
    loose = [("en", frozenset({"a"})), ("de", frozenset({"b"}))]
    check("disjoint support -> coherence 0.0",
          coherence_and_breadth(loose, rng)[0], 0.0)
    same_lang = [("en", frozenset({"a"})), ("en", frozenset({"a"}))]
    coherence, breadth = coherence_and_breadth(same_lang, rng)
    check("same-language pairs excluded", coherence, None)
    check("same-language breadth", breadth, 1)

    print("--- candidate_scores (leakage discipline) ---")
    chains = [
        {"lang": "en", "answer": "10", "signature": frozenset({"x"})},
        {"lang": "de", "answer": "10", "signature": frozenset({"x"})},
        {"lang": "hi", "answer": "10", "signature": frozenset({"x"})},
        {"lang": "hi", "answer": "20", "signature": frozenset({"y"})},
    ]
    scores = candidate_scores(chains, "hi", {"10", "20"}, rng)
    check("target language excluded from support",
          scores["10"]["n_chains"], 2)
    check("candidate supported only by target language has no score",
          "20" in scores, False)
    check("cross-lingual coherence computed", scores["10"]["coherence"], 1.0)

    print("--- auc sanity ---")
    check("separable", auc([1.0, 0.9], [0.1, 0.2]), 1.0)
    check("chance", auc([0.5, 0.5], [0.5, 0.5]), 0.5)

    print("--- decide ---")
    counts = Counter({"1825": 2, "2190": 2})
    scores = {"2190": {"coherence": 0.9, "breadth": 10, "n_chains": 20},
              "1825": {"coherence": 0.4, "breadth": 8, "n_chains": 20}}
    picked = decide(counts, scores, n_reference_langs=20)
    check("tight broad support wins", picked["answer"], "2190")
    check("route", picked["route"], "structural")
    thin = decide(counts, scores, n_reference_langs=2)
    check("thin reference falls back", thin["route"], "no_reference")
    check("fallback is Rung 0 plurality", thin["answer"],
          max(counts, key=lambda a: (counts[a], a)))
    no_evidence = decide(counts, {}, n_reference_langs=20)
    check("no evidence falls back", no_evidence["route"], "no_evidence")
    tied = decide(counts, {"2190": {"coherence": 0.5, "breadth": 5, "n_chains": 5},
                           "1825": {"coherence": 0.5, "breadth": 5, "n_chains": 5}},
                  n_reference_langs=10)
    check("exact tie is indecisive", tied["route"], "indecisive")

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(dry=args.dry_run)
