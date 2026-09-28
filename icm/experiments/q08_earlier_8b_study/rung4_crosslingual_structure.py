"""Rung 4 — Rung 0 + a cross-lingual STRUCTURAL tie-breaker on split pools.

Rung 3 (measurement) established, on 7,392 pools:
  * structure agreement across languages separates right from wrong pairs
    (Jaccard 0.859 both-right vs 0.561 both-wrong vs 0.031 permutation null),
    which answer agreement provably cannot (D2: wrong languages give the same
    wrong answer 54-57% of the time).
  * BUT structure-weighted voting within a language loses to plain counting
    (Q1), so structure must not replace the vote.

So Rung 4 uses structure the only way the evidence supports: as a REFERENCE
built from OTHER languages, consulted only where Rung 0 abstains.

For a split pool in language L on problem P:
  1. build a structural reference for P from every language except L
     (values-signature; each language contributes the computation supporting
     its own plurality answer)
  2. for each candidate answer in L's pool, score its supporting chains by
     structural similarity to that reference
  3. commit the candidate whose computation best matches how the rest of the
     world computed this problem

Leakage discipline: the reference for (P, L) never includes L, and gold is
used only in the final report. The reference is built from other languages'
PLURALITY answers, which are themselves label-free.

Baselines reported side by side on identical items:
  rung0     plurality of the pool (what Rung 0 would answer if forced)
  rung4     structural tie-break as above
  oracle    best answer present in the pool (the ceiling for any selector)

Usage:
  python rung4_crosslingual_structure.py --self-test
  python rung4_crosslingual_structure.py --dry-run
  python rung4_crosslingual_structure.py
"""

from __future__ import annotations

import argparse
import glob
import json
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
    parse_equations,
    plurality_answer,
    signature,
)
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
OUT = RUN_DIR / "rung4_results.json"

SIGNATURE_KIND = "values"   # best right/wrong separation in Rung 3
MIN_REFERENCE_LANGS = 3     # below this the reference is too thin to trust
MIN_MARGIN = 0.0            # commit only if the winner beats the runner-up by this


def build_reference(problem_entries: dict[str, dict], exclude: str) -> frozenset[str]:
    """Union of the computations other languages used for their own plurality."""
    reference: set[str] = set()
    for lang, entry in problem_entries.items():
        if lang == exclude:
            continue
        reference |= entry["signature"]
    return frozenset(reference)


def reference_language_count(problem_entries: dict[str, dict], exclude: str) -> int:
    return sum(1 for lang in problem_entries if lang != exclude)


def score_candidates(chains: list[dict], reference: frozenset[str]
                     ) -> list[tuple[str, float, int]]:
    """(answer, mean structural similarity to reference, n supporting chains)."""
    by_answer: dict[str, list[float]] = defaultdict(list)
    for chain in chains:
        if not chain["signature"]:
            continue
        by_answer[chain["answer"]].append(jaccard(chain["signature"], reference))
    scored = [(answer, st.mean(scores), len(scores))
              for answer, scores in by_answer.items() if scores]
    scored.sort(key=lambda row: (-row[1], -row[2], row[0]))
    return scored


def decide(chains: list[dict], reference: frozenset[str], n_reference_langs: int
           ) -> dict:
    """Rung 4's decision for one split pool. Falls back to Rung 0 when the
    structural evidence is absent, thin, or indecisive."""
    fallback = plurality_answer(chains)
    if n_reference_langs < MIN_REFERENCE_LANGS or not reference:
        return {"answer": fallback, "route": "no_reference", "changed": False,
                "margin": 0.0}
    scored = score_candidates(chains, reference)
    if not scored:
        return {"answer": fallback, "route": "no_equations", "changed": False,
                "margin": 0.0}
    if len(scored) == 1:
        winner, best, _ = scored[0]
        margin = best
    else:
        winner, best, _ = scored[0]
        margin = best - scored[1][1]
    if margin <= MIN_MARGIN:
        return {"answer": fallback, "route": "indecisive", "changed": False,
                "margin": round(margin, 4)}
    return {"answer": winner, "route": "structural", "margin": round(margin, 4),
            "changed": winner != fallback, "top_similarity": round(best, 4)}


def load(languages: set[str] | None = None):
    import pyarrow.parquet as pq

    keep = languages or set(TIER)
    gold: dict[str, str] = {}
    for lang in keep:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = normalize_number(row.get("gold_number"))

    pools: dict[str, list[dict]] = defaultdict(list)
    columns = ["instance_id", "canonical_answer", "parse_status", "chain_text"]
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=columns).to_pylist():
            lang = "_".join(row["instance_id"].split("_")[1:-1])
            answer = normalize_number(row["canonical_answer"])
            if lang not in keep or row["parse_status"] != "parsed" or answer is None:
                continue
            pools[row["instance_id"]].append({
                "answer": answer,
                "signature": signature(row["chain_text"], SIGNATURE_KIND),
                "has_equations": bool(parse_equations(row["chain_text"])),
            })

    # Per-problem structural entries: each language's plurality computation.
    problems: dict[str, dict[str, dict]] = defaultdict(dict)
    for instance_id, chains in pools.items():
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        pick = plurality_answer(chains)
        union: set[str] = set()
        for chain in chains:
            if chain["answer"] == pick:
                union |= chain["signature"]
        if union:
            problems[problem][lang] = {"answer": pick, "signature": frozenset(union)}
    return pools, gold, problems


def report(items: list[dict]) -> dict:
    results: dict = {}
    print("=" * 80)
    print("RUNG 4 — cross-lingual structural tie-break on Rung 0's abstained bucket")
    print("=" * 80)
    print(f"{'tier':<6}{'n':>7}{'Rung0':>9}{'Rung4':>9}{'delta':>8}"
          f"{'oracle':>9}{'changed':>9}{'structural%':>13}")
    for tier in TIERS + ("ALL",):
        rows = [i for i in items if tier in ("ALL", i["tier"])]
        if not rows:
            continue
        base = sum(answers_match(i["rung0"], i["gold"]) for i in rows)
        rung4 = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in rows)
        oracle = sum(i["gold_in_pool"] for i in rows)
        changed = sum(i["decision"]["changed"] for i in rows)
        structural = sum(i["decision"]["route"] == "structural" for i in rows)
        results[tier] = {
            "n": len(rows),
            "rung0": round(base / len(rows), 4),
            "rung4": round(rung4 / len(rows), 4),
            "delta_points": round(100 * (rung4 - base) / len(rows), 2),
            "oracle_ceiling": round(oracle / len(rows), 4),
            "changed": changed,
            "structural_route_share": round(structural / len(rows), 4),
        }
        print(f"{tier:<6}{len(rows):>7}{100*base/len(rows):>8.1f}%"
              f"{100*rung4/len(rows):>8.1f}%{100*(rung4-base)/len(rows):>+7.1f}"
              f"{100*oracle/len(rows):>8.1f}%{changed:>9}"
              f"{100*structural/len(rows):>12.1f}%")

    changed_items = [i for i in items if i["decision"]["changed"]]
    if changed_items:
        now = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in changed_items)
        was = sum(answers_match(i["rung0"], i["gold"]) for i in changed_items)
        results["changed_items"] = {"n": len(changed_items), "rung4_correct": now,
                                    "rung0_correct": was, "net": now - was}
        print(f"\nitems where structure overruled the vote: {len(changed_items)}")
        print(f"  Rung 0 had {was} right -> Rung 4 has {now} right (net {now-was:+d})")

    # Is the structural margin a usable confidence signal? (selective answering)
    structural_items = [i for i in items if i["decision"]["route"] == "structural"]
    if structural_items:
        structural_items.sort(key=lambda i: -i["decision"]["margin"])
        print("\nselective answering on the structural route "
              f"({len(structural_items)} items), by margin:")
        frontier = []
        for keep in (1.0, 0.75, 0.5, 0.25):
            top = structural_items[:max(1, int(keep * len(structural_items)))]
            hits = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in top)
            base = sum(answers_match(i["rung0"], i["gold"]) for i in top)
            frontier.append({"keep": keep, "n": len(top),
                             "rung4": round(hits / len(top), 4),
                             "rung0": round(base / len(top), 4)})
            print(f"  top {keep:>4.0%} by margin ({len(top):>4} items): "
                  f"Rung 4 {100*hits/len(top):>5.1f}%  vs Rung 0 {100*base/len(top):>5.1f}%")
        results["margin_frontier"] = frontier

    routes = Counter(i["decision"]["route"] for i in items)
    results["routes"] = dict(routes)
    print(f"\nroutes: {dict(routes)}")
    return results


def self_test() -> None:
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- build_reference (leakage discipline) ---")
    entries = {
        "en": {"answer": "2190", "signature": frozenset({"2190", "1800"})},
        "de": {"answer": "2190", "signature": frozenset({"2190", "390"})},
        "hi": {"answer": "1825", "signature": frozenset({"1825"})},
    }
    check("excludes the target language",
          build_reference(entries, "hi"), frozenset({"2190", "1800", "390"}))
    check("target's own signature absent",
          "1825" in build_reference(entries, "hi"), False)
    check("reference count", reference_language_count(entries, "hi"), 2)
    check("single-language problem yields empty reference",
          build_reference({"en": entries["en"]}, "en"), frozenset())

    print("--- score_candidates ---")
    chains = [
        {"answer": "2190", "signature": frozenset({"2190", "1800"})},
        {"answer": "2190", "signature": frozenset({"2190", "390"})},
        {"answer": "1825", "signature": frozenset({"1825", "999"})},
    ]
    reference = frozenset({"2190", "1800", "390"})
    scored = score_candidates(chains, reference)
    check("winner is the reference-aligned answer", scored[0][0], "2190")
    check("both candidates scored", len(scored), 2)
    check("loser scores zero", scored[1][1], 0.0)
    check("chains without equations ignored",
          score_candidates([{"answer": "5", "signature": frozenset()}], reference), [])

    print("--- decide (fallback discipline) ---")
    thin = decide(chains, reference, n_reference_langs=1)
    check("thin reference falls back to Rung 0", thin["answer"], "2190")
    check("thin reference route", thin["route"], "no_reference")
    check("thin reference never flags a change", thin["changed"], False)
    empty = decide(chains, frozenset(), n_reference_langs=5)
    check("empty reference falls back", empty["route"], "no_reference")
    no_eq = decide([{"answer": "7", "signature": frozenset()}], reference, 5)
    check("no equations falls back", no_eq["route"], "no_equations")
    check("no equations keeps plurality", no_eq["answer"], "7")

    # A pool where the vote says 1825 but the other languages computed 2190.
    contested = [
        {"answer": "1825", "signature": frozenset({"1825", "999"})},
        {"answer": "1825", "signature": frozenset({"1825", "998"})},
        {"answer": "2190", "signature": frozenset({"2190", "1800", "390"})},
    ]
    check("plain vote picks 1825", plurality_answer(contested), "1825")
    overruled = decide(contested, reference, n_reference_langs=5)
    check("structure overrules the vote", overruled["answer"], "2190")
    check("change is flagged", overruled["changed"], True)
    check("route is structural", overruled["route"], "structural")

    # Ties must not be broken by structure.
    tied = [
        {"answer": "10", "signature": frozenset({"5"})},
        {"answer": "20", "signature": frozenset({"5"})},
    ]
    tie = decide(tied, frozenset({"5"}), n_reference_langs=5)
    check("equal similarity is indecisive", tie["route"], "indecisive")
    check("indecisive keeps Rung 0's answer", tie["answer"], plurality_answer(tied))

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    pools, gold, problems = load()
    print(f"pools {len(pools)}, problems {len(problems)}")

    items: list[dict] = []
    for instance_id, chains in pools.items():
        g = gold.get(instance_id)
        if g is None or not chains:
            continue
        counts = Counter(c["answer"] for c in chains)
        if counts.most_common(1)[0][1] / len(chains) >= SPLIT_THRESHOLD:
            continue  # Rung 0 commits; Rung 4 never touches these
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        entries = problems.get(problem, {})
        reference = build_reference(entries, lang)
        n_reference = reference_language_count(entries, lang)
        items.append({
            "instance_id": instance_id, "lang": lang, "tier": TIER[lang],
            "gold": g, "rung0": plurality_answer(chains),
            "gold_in_pool": any(answers_match(c["answer"], g) for c in chains),
            "decision": decide(chains, reference, n_reference),
        })

    if args.dry_run:
        print(f"split pools: {len(items)}")
        print(f"routes: {Counter(i['decision']['route'] for i in items)}")
        sample = next(i for i in items if i["decision"]["route"] == "structural")
        print(f"example: {sample['instance_id']} gold={sample['gold']} "
              f"rung0={sample['rung0']} rung4={sample['decision']}")
        return

    results = report(items)
    OUT.write_text(json.dumps(results, indent=2))
    detail = OUT.with_name("rung4_items.jsonl")
    with detail.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT} and {detail}")


if __name__ == "__main__":
    main()
