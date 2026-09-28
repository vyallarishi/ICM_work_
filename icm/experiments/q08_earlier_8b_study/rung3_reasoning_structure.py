"""Rung 3 (measurement) — is reasoning STRUCTURE more informative than the answer?

Every mechanism tested so far compared chains by their final answer. Rung 2
showed why that fails: the errors are setup errors, so a chain can compute
flawlessly and still be wrong. Chains carry their computation explicitly as
<<a op b=c>> annotations, so we can compare chains by WHAT THEY COMPUTED
instead of what they concluded.

Two questions, both answered from frozen data (no GPU, no model):

  Q1 (within-language): on split pools, does setup consensus predict the gold
     answer better than answer consensus does?
  Q2 (cross-lingual): equations are language-independent. Answer agreement
     across languages was shown uninformative (D2: wrong languages agree 54%
     of the time vs ~2% null). Is STRUCTURE agreement across languages more
     informative than that? Same permutation-null discipline as D2.

Three signature granularities are compared, since which one carries signal is
itself a finding:
  ops     multiset of normalized operations      {"6*365", "1800+390"}
  values  multiset of intermediate results       {"1800", "390", "2190"}
  nums    set of all numbers used in computation {"6","365","1800","390","2190"}

Usage:
  python rung3_equations.py --self-test     # logic checks, seconds
  python rung3_equations.py --dry-run       # shapes/costs on 2 languages
  python rung3_equations.py                 # full measurement
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from ...paths import RUN_DIR

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
OUT = RUN_DIR / "rung3_results.json"

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
SPLIT_THRESHOLD = 0.70
SIGNATURES = ("ops", "values", "nums")

_EQUATION_RE = re.compile(r"<<([^>]{1,120}?)>>")
_OP_SPLIT_RE = re.compile(r"[+\-*/×÷]")


def normalize_number(value) -> str | None:
    try:
        number = Decimal(str(value).replace(",", "").replace(" ", ""))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not number.is_finite():
        return None
    number = number.normalize()
    if number == number.to_integral_value():
        number = number.quantize(Decimal(1))
    return f"{number:f}"


def parse_equations(chain_text: str) -> list[tuple[str, str]]:
    """[(lhs, rhs)] from <<a op b=c>> annotations, numerically normalized."""
    parsed: list[tuple[str, str]] = []
    for body in _EQUATION_RE.findall(chain_text or ""):
        if "=" not in body:
            continue
        lhs, _, rhs = body.rpartition("=")
        lhs = lhs.replace("×", "*").replace("÷", "/").replace(" ", "")
        right = normalize_number(rhs)
        if not lhs or right is None:
            continue
        # normalize each operand so 6*365 and 6.0*365 collapse
        operands = [normalize_number(token) for token in _OP_SPLIT_RE.split(lhs)]
        if any(operand is None for operand in operands):
            continue
        operators = _OP_SPLIT_RE.findall(lhs)
        canonical = operands[0]
        for operator, operand in zip(operators, operands[1:]):
            canonical += operator + operand
        parsed.append((canonical, right))
    return parsed


def commutative_key(lhs: str) -> str:
    """`365*6` and `6*365` are the same computation; `9-4` and `4-9` are not."""
    if "*" in lhs and not any(op in lhs for op in "+-/"):
        return "*".join(sorted(lhs.split("*")))
    if "+" in lhs and not any(op in lhs for op in "*-/"):
        return "+".join(sorted(lhs.split("+")))
    return lhs


def signature(chain_text: str, kind: str) -> frozenset[str]:
    equations = parse_equations(chain_text)
    if kind == "ops":
        return frozenset(commutative_key(lhs) for lhs, _ in equations)
    if kind == "values":
        return frozenset(rhs for _, rhs in equations)
    if kind == "nums":
        numbers: set[str] = set()
        for lhs, rhs in equations:
            numbers.update(token for token in _OP_SPLIT_RE.split(lhs) if token)
            numbers.add(rhs)
        return frozenset(numbers)
    raise ValueError(f"unknown signature kind: {kind}")


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def answers_match(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return False
    try:
        a, b = Decimal(left), Decimal(right)
    except (InvalidOperation, TypeError):
        return False
    if not (a.is_finite() and b.is_finite()):
        return False
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= Decimal("0.005")


# ------------------------------------------------------------- consensus

def structure_consensus_answer(chains: list[dict], kind: str) -> str | None:
    """Pick the answer whose chains form the largest STRUCTURAL cluster.

    Each chain votes with the size of its structural neighbourhood (chains
    sharing >= 0.5 Jaccard on the signature), so an answer supported by many
    chains that computed the same way outranks an answer supported by the same
    number of chains that each computed differently.
    """
    usable = [c for c in chains if c["signature"][kind]]
    if not usable:
        return None
    weights: dict[str, float] = defaultdict(float)
    for chain in usable:
        neighbours = sum(
            1 for other in usable
            if jaccard(chain["signature"][kind], other["signature"][kind]) >= 0.5
        )
        weights[chain["answer"]] += neighbours
    return max(weights, key=lambda answer: (weights[answer], answer))


def plurality_answer(chains: list[dict]) -> str | None:
    counts = Counter(c["answer"] for c in chains)
    if not counts:
        return None
    return max(counts, key=lambda answer: (counts[answer], answer))


# ------------------------------------------------------------ data loading

def load_pools(languages: set[str] | None = None) -> tuple[dict[str, list[dict]], dict[str, str]]:
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
                "signature": {kind: signature(row["chain_text"], kind)
                              for kind in SIGNATURES},
                "n_equations": len(parse_equations(row["chain_text"])),
            })
    return pools, gold


# ----------------------------------------------------------------- Q1 / Q2

def q1_within_language(pools: dict[str, list[dict]], gold: dict[str, str]) -> dict:
    """On split pools: does structure consensus beat answer consensus?"""
    print("=" * 78)
    print("Q1 — within-language: structure consensus vs answer consensus")
    print("=" * 78)
    rows: list[dict] = []
    for instance_id, chains in pools.items():
        g = gold.get(instance_id)
        if g is None or not chains:
            continue
        counts = Counter(c["answer"] for c in chains)
        top_n = counts.most_common(1)[0][1]
        if top_n / len(chains) >= SPLIT_THRESHOLD:
            continue  # Rung 0 already commits
        lang = "_".join(instance_id.split("_")[1:-1])
        rows.append({
            "tier": TIER[lang], "gold": g, "chains": chains,
            "plurality": plurality_answer(chains),
            **{kind: structure_consensus_answer(chains, kind) for kind in SIGNATURES},
        })

    results: dict = {"n_split_pools": len(rows)}
    print(f"split pools: {len(rows)}")
    coverage = st.mean([sum(1 for c in r["chains"] if c["n_equations"] > 0) / len(r["chains"])
                        for r in rows]) if rows else 0.0
    results["chains_with_equations"] = round(coverage, 4)
    print(f"chains carrying >=1 equation: {100*coverage:.1f}%")
    header = f"{'tier':<6}{'n':>7}{'answer-vote':>13}" + "".join(f"{kind:>10}" for kind in SIGNATURES)
    print(header)
    for tier in TIERS + ("ALL",):
        subset = [r for r in rows if tier in ("ALL", r["tier"])]
        if not subset:
            continue
        base = sum(answers_match(r["plurality"], r["gold"]) for r in subset)
        cells = {}
        for kind in SIGNATURES:
            hits = sum(answers_match(r[kind], r["gold"]) for r in subset)
            cells[kind] = hits / len(subset)
        results[tier] = {"n": len(subset),
                         "answer_vote": round(base / len(subset), 4),
                         **{kind: round(value, 4) for kind, value in cells.items()}}
        print(f"{tier:<6}{len(subset):>7}{100*base/len(subset):>12.1f}%"
              + "".join(f"{100*cells[kind]:>9.1f}%" for kind in SIGNATURES))
    return results


def q2_cross_lingual(pools: dict[str, list[dict]], gold: dict[str, str],
                     kind: str = "ops") -> dict:
    """Does STRUCTURE agree across languages more than answers do (vs null)?

    D2 measured: when two languages are both wrong, they give the same wrong
    ANSWER 54% of the time (null ~2%). Here: when two languages are both
    wrong, how similar are their computations — and is structural similarity
    higher when languages are RIGHT than when they are WRONG? If structure
    similarity separates right from wrong pairs, it is informative in a way
    answer agreement is not.
    """
    print()
    print("=" * 78)
    print(f"Q2 — cross-lingual structure agreement (signature: {kind})")
    print("=" * 78)

    by_problem: dict[str, dict[str, dict]] = defaultdict(dict)
    for instance_id, chains in pools.items():
        g = gold.get(instance_id)
        if g is None or not chains:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        pick = plurality_answer(chains)
        # pool-level structural signature: union over the chains supporting
        # that language's plurality answer
        supporting = [c for c in chains if c["answer"] == pick and c["signature"][kind]]
        if not supporting:
            continue
        union: set[str] = set()
        for chain in supporting:
            union |= chain["signature"][kind]
        by_problem[problem][lang] = {
            "answer": pick, "signature": frozenset(union),
            "correct": answers_match(pick, g),
        }

    both_right, both_wrong, mixed = [], [], []
    same_answer_wrong = total_wrong_pairs = 0
    for langs in by_problem.values():
        entries = list(langs.values())
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                score = jaccard(a["signature"], b["signature"])
                if a["correct"] and b["correct"]:
                    both_right.append(score)
                elif not a["correct"] and not b["correct"]:
                    both_wrong.append(score)
                    total_wrong_pairs += 1
                    same_answer_wrong += answers_match(a["answer"], b["answer"])
                else:
                    mixed.append(score)

    # Permutation null: shuffle which problem each language's signature belongs
    # to, preserving per-language signature distributions.
    rng = random.Random(20260808)
    by_lang: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for problem, langs in by_problem.items():
        for lang, entry in langs.items():
            by_lang[lang].append((problem, entry["signature"]))
    null_scores: list[float] = []
    for _ in range(20):
        shuffled: dict[str, dict[str, frozenset[str]]] = defaultdict(dict)
        for lang, pairs in by_lang.items():
            signatures = [s for _, s in pairs]
            rng.shuffle(signatures)
            for (problem, _), sig in zip(pairs, signatures):
                shuffled[problem][lang] = sig
        for langs in shuffled.values():
            values = list(langs.values())
            for i in range(0, len(values) - 1, 2):  # sample pairs, keep it cheap
                null_scores.append(jaccard(values[i], values[i + 1]))

    def summarize(name: str, values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        return {"n": len(values), "mean_jaccard": round(st.mean(values), 4),
                "median": round(st.median(values), 4)}

    results = {
        "signature": kind,
        "both_right": summarize("both right", both_right),
        "both_wrong": summarize("both wrong", both_wrong),
        "one_right_one_wrong": summarize("mixed", mixed),
        "permutation_null": summarize("null", null_scores),
        "same_wrong_answer_rate": round(same_answer_wrong / max(total_wrong_pairs, 1), 4),
    }
    print(f"{'pair type':<24}{'n':>8}{'mean Jaccard':>15}")
    for label, key in (("both languages right", "both_right"),
                       ("both wrong", "both_wrong"),
                       ("one right, one wrong", "one_right_one_wrong"),
                       ("permutation null", "permutation_null")):
        block = results[key]
        if block.get("n"):
            print(f"{label:<24}{block['n']:>8}{block['mean_jaccard']:>15.3f}")
    print(f"\n(reference) same wrong ANSWER rate among wrong pairs: "
          f"{100*results['same_wrong_answer_rate']:.1f}%")
    separation = (results["both_right"].get("mean_jaccard", 0)
                  - results["both_wrong"].get("mean_jaccard", 0))
    results["right_minus_wrong_separation"] = round(separation, 4)
    print(f"separation (right-pairs minus wrong-pairs similarity): {separation:+.3f}")
    return results


# --------------------------------------------------------------- self-test

def self_test() -> None:
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- parse_equations ---")
    chain = ("Sy verdien 6 * 365 = $<<6*365=2190>>2190 per jaar.\n"
             "Sy spandeer <<2190-190=2000>>2000.\n#### 2000")
    check("two equations", parse_equations(chain),
          [("6*365", "2190"), ("2190-190", "2000")])
    check("no annotations", parse_equations("plain text #### 5"), [])
    check("malformed skipped", parse_equations("<<not an equation>>"), [])
    check("unicode operators", parse_equations("<<6×365=2190>>"), [("6*365", "2190")])
    check("decimal normalized", parse_equations("<<1.50+5.50=7.00>>"),
          [("1.5+5.5", "7")])
    check("comma thousands", parse_equations("<<1,200*2=2,400>>"),
          [("1200*2", "2400")])
    check("empty input", parse_equations(""), [])
    check("none-safe", parse_equations(None), [])

    print("--- commutative_key ---")
    check("multiplication sorted", commutative_key("365*6"), commutative_key("6*365"))
    check("addition sorted", commutative_key("24+48"), commutative_key("48+24"))
    check("subtraction NOT sorted", commutative_key("9-4") == commutative_key("4-9"), False)
    check("mixed left alone", commutative_key("6*365+10"), "6*365+10")

    print("--- signature ---")
    # operands of a commutative op are sorted, so 6*365 canonicalizes to 365*6
    check("ops", signature(chain, "ops"), frozenset({"365*6", "2190-190"}))
    check("values", signature(chain, "values"), frozenset({"2190", "2000"}))
    check("nums", signature(chain, "nums"),
          frozenset({"6", "365", "2190", "190", "2000"}))
    check("commutative equivalence",
          signature("<<365*6=2190>>", "ops"), signature("<<6*365=2190>>", "ops"))
    check("empty chain", signature("no equations", "ops"), frozenset())

    print("--- jaccard ---")
    check("identical", jaccard(frozenset({"a"}), frozenset({"a"})), 1.0)
    check("disjoint", jaccard(frozenset({"a"}), frozenset({"b"})), 0.0)
    check("half", jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})), 1 / 3)
    check("both empty", jaccard(frozenset(), frozenset()), 1.0)
    check("one empty", jaccard(frozenset({"a"}), frozenset()), 0.0)

    print("--- structure_consensus_answer ---")
    # 2 chains agree on answer 1825 but computed differently;
    # 2 chains agree on 2190 AND computed identically -> structure prefers 2190.
    pool = [
        {"answer": "2190", "chain": "<<6*365=2190>>"},
        {"answer": "2190", "chain": "<<6*365=2190>>"},
        {"answer": "1825", "chain": "<<5*365=1825>>"},
        {"answer": "1825", "chain": "<<1825*1=1825>>"},
        {"answer": "1825", "chain": "<<1000+825=1825>>"},
    ]
    for entry in pool:
        entry["signature"] = {kind: signature(entry["chain"], kind) for kind in SIGNATURES}
    check("answer vote picks 1825", plurality_answer(pool), "1825")
    check("structure picks 2190", structure_consensus_answer(pool, "ops"), "2190")
    check("no equations -> None",
          structure_consensus_answer(
              [{"answer": "5", "signature": {k: frozenset() for k in SIGNATURES}}],
              "ops"), None)

    print("--- answers_match ---")
    check("exact", answers_match("2190", "2190"), True)
    check("tolerance", answers_match("9.9996", "10"), True)
    check("different", answers_match("2190", "1825"), False)
    check("none", answers_match(None, "1"), False)

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

    languages = {"en", "hi", "am"} if args.dry_run else None
    pools, gold = load_pools(languages)
    print(f"pools loaded: {len(pools)} "
          f"({sum(len(v) for v in pools.values())} chains)\n")
    if args.dry_run:
        sample = next(iter(pools.values()))[0]
        print(f"example signatures: {dict(sample['signature'])}")
        print(f"equations in that chain: {sample['n_equations']}\n")

    results = {"q1": q1_within_language(pools, gold)}
    for kind in SIGNATURES:
        results[f"q2_{kind}"] = q2_cross_lingual(pools, gold, kind)
    if not args.dry_run:
        OUT.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
