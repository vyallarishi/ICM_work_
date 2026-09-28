"""Rung 2 — Rung 0 + one execution check on split pools.

Rung 0 abstains on 1,749 pools whose top answer holds <70% of the vote; forced
plurality there scores 52.5% (low tier 46.5%). Rung 2 adds exactly one part:
for the chains in those pools, ask the model to transcribe the chain's
arithmetic into a short Python program, EXECUTE it, and keep only chains whose
executed result matches their own stated answer. Re-vote among survivors.

The interpreter is the new evidence channel: unlike every mechanism tested in
Phase 1, its verdict does not come from the model's own opinion.

Design decisions that keep this honest:
  * label-blind: nothing here sees gold; gold enters only in the report.
  * transcription-only prompt: the model is asked to TRANSCRIBE the chain's
    computation, not to re-solve the problem (re-solving would just be another
    sample, which is Rung 3, not Rung 2).
  * a chain whose program fails to parse/run/print is UNVERIFIED, never
    "wrong" — unverified chains keep their vote at reduced weight so the
    mechanism can only add evidence, never destroy the Rung 0 baseline.
  * the fallback ladder is pre-registered: verified-majority -> weighted vote
    -> plain plurality (= Rung 0 forced), so precision is measured against a
    fixed comparator on identical items.

Run `--self-test` (no GPU, no model) to exercise every non-GPU component.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import multiprocessing as mp
import re
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
MODEL_DIR = "/lustre10/scratch/rvyalla/sm-icm/assets/models/Qwen3-8B-Base"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent

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
SPLIT_THRESHOLD = 0.70          # Rung 0's commit threshold; below it = split pool
MAX_CHAINS_PER_POOL = 10        # cap on GPU work per pool
EXEC_TIMEOUT_SECONDS = 5
RELATIVE_TOLERANCE = Decimal("0.005")

# Rung 2 ADJUSTS Rung 0's vote; it never replaces it. A chain's vote is scaled
# by the execution verdict, so with no verdicts at all the decision is exactly
# Rung 0's plurality. Weights are pre-registered, not tuned on gold.
VOTE_WEIGHT = {
    True: 1.5,    # program reproduced the chain's own computation
    False: 0.5,   # program contradicted the chain's stated answer
    None: 1.0,    # no usable program -> chain keeps its plain Rung 0 vote
}
# A chain also has to be transcribed FAITHFULLY to earn the bonus: the program
# must reproduce the chain's intermediate results, not merely print its answer.
MIN_STEP_COVERAGE = 0.5

# ---------------------------------------------------------------- helpers

def normalize_answer(value) -> str | None:
    """Canonical numeric string. `Decimal.normalize()` alone yields scientific
    notation for round numbers ('2190' -> '2.19E+3'), so re-expand integers."""
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


def answers_match(left: str | None, right: str | None) -> bool:
    """Numeric equality with a small relative tolerance (rounding in chains)."""
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
    return scale > 0 and abs(a - b) / scale <= RELATIVE_TOLERANCE


# ------------------------------------------------------- program extraction

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Store,
    ast.Load, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Call, ast.Tuple, ast.List,
)
_ALLOWED_CALLS = {"round", "min", "max", "abs", "int", "float", "sum", "len", "print"}


def extract_program(text: str) -> str | None:
    """Pull the program out of model output.

    The prompt ends with an opening ```python fence, so the model's completion
    is bare code terminated by a closing fence (or by a stop sequence). Handle
    both that shape and a fully fenced block, and stop at any prompt-echo.
    """
    match = _CODE_FENCE_RE.search(text)
    body = match.group(1) if match else text
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            break  # closing fence of an unmatched opening fence
        if not lines and not stripped:
            continue
        if stripped.startswith(("[Q]", "[R]", "[V]", "Question:", "<example>",
                                "Solution:")):
            break
        lines.append(line)
    program = "\n".join(lines).strip()
    return program or None


def is_program_safe(program: str) -> bool:
    """Whitelist: arithmetic, assignments, and a few numeric builtins only."""
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_CALLS:
                return False
    return True


def _exec_worker(program: str, queue) -> None:  # pragma: no cover - subprocess
    import io
    import contextlib

    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            namespace: dict = {"__builtins__": {name: __builtins__[name]  # type: ignore[index]
                                                if isinstance(__builtins__, dict)
                                                else getattr(__builtins__, name)
                                                for name in _ALLOWED_CALLS}}
            exec(compile(program, "<chain>", "exec"), namespace)  # noqa: S102
        queue.put(("ok", buffer.getvalue()))
    except Exception as error:  # noqa: BLE001 - any failure is "unverified"
        queue.put(("error", f"{type(error).__name__}: {error}"))


def run_program(program: str, timeout: float = EXEC_TIMEOUT_SECONDS
                ) -> tuple[str | None, str, list[str]]:
    """Execute in a sandboxed subprocess.

    Returns (last printed number, status, all printed numbers in order).
    """
    if not is_program_safe(program):
        return None, "unsafe", []
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_exec_worker, args=(program, queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return None, "timeout", []
    if queue.empty():
        return None, "crashed", []
    status, payload = queue.get()
    if status == "error":
        return None, "error", []
    raw = re.findall(r"-?\d+(?:\.\d+)?", payload.replace(",", ""))
    values = [v for v in (normalize_answer(number) for number in raw) if v is not None]
    if not values:
        return None, "no_output", []
    return values[-1], "ok", values


# ------------------------------------------------------------ the decision

def decide_pool(chains: list[dict]) -> dict:
    """Rung 2's decision for one split pool: Rung 0's vote, execution-reweighted.

    chains: [{"answer": str, "verified": bool|None}] where verified is
    True (program reproduced the chain), False (program contradicted it), or
    None (no usable program).

    Each chain contributes VOTE_WEIGHT[verified] to its answer. With every
    verdict None this is exactly Rung 0's plurality, so the mechanism can only
    perturb the baseline in proportion to how much evidence it actually
    gathered. Ties fall back to the plain (unweighted) Rung 0 count.
    """
    if not chains:
        return {"answer": None, "route": "empty", "support": 0.0,
                "n_verified": 0, "n_refuted": 0, "changed": False}
    plain: Counter = Counter(c["answer"] for c in chains)
    rung0_pick = max(plain, key=lambda a: (plain[a], a))

    weights: dict[str, float] = defaultdict(float)
    for chain in chains:
        weights[chain["answer"]] += VOTE_WEIGHT[chain.get("verified")]
    best = max(weights.values())
    leaders = [answer for answer, weight in weights.items()
               if math.isclose(weight, best, rel_tol=1e-9)]
    # tie-break on the plain vote, then deterministically
    pick = max(leaders, key=lambda a: (plain[a], a))

    n_verified = sum(c.get("verified") is True for c in chains)
    n_refuted = sum(c.get("verified") is False for c in chains)
    route = ("no_evidence" if n_verified == 0 and n_refuted == 0
             else "reweighted")
    return {"answer": pick, "route": route, "support": round(best, 3),
            "n_verified": n_verified, "n_refuted": n_refuted,
            "rung0_pick": rung0_pick, "changed": pick != rung0_pick}


_STEP_RE = re.compile(r"<<([^>]*?)=([^>]*?)>>")


def chain_steps(chain_text: str) -> list[str]:
    """Intermediate results the chain itself claims, from its <<a*b=c>> marks."""
    steps: list[str] = []
    for _, right in _STEP_RE.findall(chain_text):
        value = normalize_answer(right)
        if value is not None:
            steps.append(value)
    return steps


def step_coverage(printed_values: list[str], steps: list[str]) -> float:
    """Fraction of the chain's claimed intermediates the program reproduced."""
    if not steps:
        return 1.0  # nothing to corroborate; final-answer match is all we have
    remaining = list(printed_values)
    hits = 0
    for step in steps:
        for index, value in enumerate(remaining):
            if answers_match(value, step):
                hits += 1
                remaining.pop(index)
                break
    return hits / len(steps)


TRANSCRIBE_PROMPT = """\
Convert the calculation in the solution below into a short Python program.
Transcribe only the arithmetic that the solution actually performs. Do not
solve the problem yourself and do not correct the solution. print() every
intermediate result the solution computes, in order, and print the final
value last.

Solution:
{chain}

```python
"""


def build_prompt(chain_text: str, max_chars: int = 1800) -> str:
    return TRANSCRIBE_PROMPT.format(chain=chain_text.strip()[:max_chars])


def verdict_for_chain(chain: dict, generated_text: str) -> dict:
    """Model output -> program -> execution -> verified/refuted/unverified.

    A verdict requires the transcription to be FAITHFUL: the program must
    reproduce at least MIN_STEP_COVERAGE of the chain's claimed INTERMEDIATE
    results (the final answer is excluded, so "print(<the answer>)" earns
    nothing). Given faithfulness, the program's final value either corroborates
    the chain's stated answer (True) or contradicts it (False). Anything
    unusable is None — never False — so a failed transcription can only leave
    Rung 0's vote untouched.
    """
    program = extract_program(generated_text)
    result = {"program": program, "verified": None, "exec_status": "no_program",
              "executed_value": None, "step_coverage": None}
    if program is None:
        return result
    value, status, printed = run_program(program)
    result.update(exec_status=status, executed_value=value)
    if status != "ok":
        return result
    steps = chain_steps(chain["chain_text"])
    # Corroboration must come from intermediates, not from the answer itself.
    intermediates = [s for s in steps if not answers_match(s, chain["answer"])]
    if not intermediates:
        result["step_coverage"] = None  # nothing independent to check against
        return result
    coverage = step_coverage(printed, intermediates)
    result["step_coverage"] = round(coverage, 3)
    if coverage < MIN_STEP_COVERAGE:
        return result  # transcription not faithful enough to carry a verdict
    result["verified"] = answers_match(value, chain["answer"])
    return result


# ------------------------------------------------------------- data loading

def load_split_pools(limit_per_tier: int | None = None) -> tuple[list[dict], dict[str, str]]:
    """Rung 0's abstained bucket: pools whose top answer holds < threshold."""
    import pyarrow.parquet as pq

    gold: dict[str, str] = {}
    for lang in TIER:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = normalize_answer(row.get("gold_number"))

    pools: dict[str, list[dict]] = defaultdict(list)
    columns = ["instance_id", "canonical_answer", "parse_status", "chain_text",
               "candidate_id"]
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=columns).to_pylist():
            answer = normalize_answer(row["canonical_answer"])
            if row["parse_status"] != "parsed" or answer is None:
                continue
            pools[row["instance_id"]].append(
                {"candidate_id": row["candidate_id"], "answer": answer,
                 "chain_text": row["chain_text"]})

    split: list[dict] = []
    for instance_id, chains in pools.items():
        g = gold.get(instance_id)
        if g is None or not chains:
            continue
        counts = Counter(c["answer"] for c in chains)
        top, top_n = counts.most_common(1)[0]
        if top_n / len(chains) >= SPLIT_THRESHOLD:
            continue  # Rung 0 already commits here; Rung 2 never touches it
        lang = "_".join(instance_id.split("_")[1:-1])
        # keep a diverse subset: cover every distinct answer before duplicates
        by_answer: dict[str, list[dict]] = defaultdict(list)
        for chain in chains:
            by_answer[chain["answer"]].append(chain)
        ordered: list[dict] = []
        while len(ordered) < MAX_CHAINS_PER_POOL and any(by_answer.values()):
            for answer in sorted(by_answer, key=lambda a: -counts[a]):
                if by_answer[answer] and len(ordered) < MAX_CHAINS_PER_POOL:
                    ordered.append(by_answer[answer].pop())
        split.append({
            "instance_id": instance_id, "lang": lang, "tier": TIER[lang],
            "gold": g, "chains": ordered,
            "rung0_plurality": top,
            "pool_share": top_n / len(chains),
        })

    if limit_per_tier is not None:
        capped: list[dict] = []
        for tier in TIERS:
            rows = [item for item in split if item["tier"] == tier]
            rows.sort(key=lambda item: item["instance_id"])
            capped.extend(rows[:limit_per_tier])
        split = capped
    return split, gold


# ------------------------------------------------------------------ report

def report(items: list[dict]) -> dict:
    results: dict = {}
    print("=" * 78)
    print("RUNG 2 — execution check on Rung 0's abstained bucket")
    print("=" * 78)
    print(f"{'tier':<6}{'n':>7}{'Rung0':>9}{'Rung2':>9}{'delta':>8}"
          f"{'changed':>9}{'transcribed%':>14}{'verified%':>11}")
    for tier in TIERS + ("ALL",):
        rows = [i for i in items if tier in ("ALL", i["tier"])]
        if not rows:
            continue
        base = sum(answers_match(i["rung0_plurality"], i["gold"]) for i in rows)
        rung2 = sum(answers_match(i["decision"]["answer"], i["gold"]) for i in rows)
        changed = sum(i["decision"]["changed"] for i in rows)
        chains = [c for i in rows for c in i["chains"]]
        ran = sum(c.get("exec_status") == "ok" for c in chains)
        verified = sum(c.get("verified") is True for c in chains)
        results[tier] = {
            "n": len(rows),
            "rung0_forced_accuracy": round(base / len(rows), 4),
            "rung2_accuracy": round(rung2 / len(rows), 4),
            "delta_points": round(100 * (rung2 - base) / len(rows), 2),
            "decisions_changed": changed,
            "exec_success_rate": round(ran / max(len(chains), 1), 4),
            "verified_rate": round(verified / max(len(chains), 1), 4),
        }
        print(f"{tier:<6}{len(rows):>7}{100*base/len(rows):>8.1f}%"
              f"{100*rung2/len(rows):>8.1f}%"
              f"{100*(rung2-base)/len(rows):>+7.1f}{changed:>9}"
              f"{100*ran/max(len(chains),1):>13.1f}%"
              f"{100*verified/max(len(chains),1):>10.1f}%")

    # The decisive diagnostic: on items where execution CHANGED the answer,
    # was the change an improvement?
    changed_items = [i for i in items if i["decision"]["changed"]]
    if changed_items:
        now_right = sum(answers_match(i["decision"]["answer"], i["gold"])
                        for i in changed_items)
        was_right = sum(answers_match(i["rung0_plurality"], i["gold"])
                        for i in changed_items)
        results["changed_items"] = {
            "n": len(changed_items), "rung2_correct": now_right,
            "rung0_correct": was_right, "net": now_right - was_right,
        }
        print(f"\nitems where execution changed the answer: {len(changed_items)}")
        print(f"  Rung 0 had {was_right} right -> Rung 2 has {now_right} right "
              f"(net {now_right - was_right:+d})")

    # Does self-consistency correlate with correctness at all?
    verified_chains = [c for i in items for c in i["chains"] if c.get("verified") is True]
    refuted_chains = [c for i in items for c in i["chains"] if c.get("verified") is False]
    for label, group in (("verified", verified_chains), ("refuted", refuted_chains)):
        if not group:
            continue
        gold_by_instance = {i["instance_id"]: i["gold"] for i in items}
        hits = sum(answers_match(c["answer"], gold_by_instance[c["instance_id"]])
                   for c in group)
        results[f"{label}_chain_correctness"] = round(hits / len(group), 4)
        print(f"  chains {label} by execution: {100*hits/len(group):.1f}% "
              f"actually end in the gold answer ({len(group)} chains)")

    statuses = Counter(c.get("exec_status", "not_attempted")
                       for i in items for c in i["chains"])
    results["exec_status_counts"] = dict(statuses)
    print(f"\nexecution outcomes across all chains: {dict(statuses)}")
    return results


# --------------------------------------------------------------- self-test

def self_test() -> None:
    """Exercise every non-GPU component with known inputs."""
    failures: list[str] = []

    def check(name: str, got, want) -> None:
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- normalize_answer / answers_match ---")
    check("normalize 2,190", normalize_answer("2,190"), "2190")
    check("normalize 2190.0", normalize_answer("2190.0"), "2190")
    check("normalize NaN", normalize_answer("NaN"), None)
    check("normalize junk", normalize_answer("abc"), None)
    check("match exact", answers_match("2190", "2190"), True)
    check("match tolerance", answers_match("9.9996", "10"), True)
    check("match distinct", answers_match("2190", "1825"), False)
    check("match none", answers_match(None, "10"), False)

    print("--- extract_program ---")
    fenced = "Sure!\n```python\nx = 6 * 365\nprint(x)\n```\nDone."
    check("fenced block", extract_program(fenced), "x = 6 * 365\nprint(x)")
    bare = "x = 2 + 3\nprint(x)\n[Q]\nnext prompt leaked"
    check("bare + stop marker", extract_program(bare), "x = 2 + 3\nprint(x)")
    # the real shape: prompt ends with an opening fence, completion closes it
    check("completion closed by fence",
          extract_program("print(6*365)\n```"), "print(6*365)")
    check("completion + trailing prose",
          extract_program("a = 2\nprint(a)\n```\nThat computes it."),
          "a = 2\nprint(a)")
    check("echoed next solution stops",
          extract_program("print(1)\nSolution:\nblah"), "print(1)")
    check("empty", extract_program("   "), None)
    check("fence only", extract_program("```"), None)

    print("--- is_program_safe ---")
    check("arithmetic ok", is_program_safe("a = 6*365\nprint(a)"), True)
    check("round ok", is_program_safe("print(round(3.14159, 2))"), True)
    check("import blocked", is_program_safe("import os\nprint(1)"), False)
    check("open blocked", is_program_safe("print(open('/etc/passwd').read())"), False)
    check("dunder blocked", is_program_safe("print(().__class__)"), False)
    check("while blocked", is_program_safe("while True: pass"), False)
    check("syntax error", is_program_safe("x = = 5"), False)

    print("--- run_program (sandboxed subprocess) ---")
    check("simple", run_program("print(6*365)")[:2], ("2190", "ok"))
    check("multi-statement", run_program("a=48\nb=a/2\nprint(a+b)")[:2], ("72", "ok"))
    check("float", run_program("print(round(10/3, 2))")[:2], ("3.33", "ok"))
    check("last print wins", run_program("print(1)\nprint(2190)")[:2], ("2190", "ok"))
    check("all prints captured",
          run_program("print(24)\nprint(72)")[2], ["24", "72"])
    check("no output", run_program("x = 5")[1], "no_output")
    check("divide by zero", run_program("print(1/0)")[1], "error")
    check("unsafe rejected", run_program("import os")[1], "unsafe")
    check("infinite loop rejected", run_program("while True: pass")[1], "unsafe")

    print("--- chain_steps / step_coverage ---")
    chain_text = "48 / 2 = <<48/2=24>>24 clips\n48 + 24 = <<48+24=72>>72\n#### 72"
    check("steps extracted", chain_steps(chain_text), ["24", "72"])
    check("no annotations", chain_steps("just prose #### 5"), [])
    check("full coverage", step_coverage(["24", "72"], ["24", "72"]), 1.0)
    check("half coverage", step_coverage(["72"], ["24", "72"]), 0.5)
    check("answer-only gets no credit", step_coverage(["72"], ["24", "99"]), 0.0)
    check("no steps to check", step_coverage(["72"], []), 1.0)

    print("--- verdict_for_chain (faithfulness requirement) ---")
    chain = {"answer": "72", "chain_text": chain_text}
    faithful = verdict_for_chain(chain, "print(48/2)\nprint(48+24)\n```")
    check("faithful transcription verifies", faithful["verified"], True)
    shortcut = verdict_for_chain(chain, "print(72)\n```")
    check("answer-only earns no bonus", shortcut["verified"], None)
    contradiction = verdict_for_chain(
        chain, "print(48/2)\nprint(48+24+100)\n```")
    check("faithful but contradictory refutes", contradiction["verified"], False)
    check("unparseable is unverified",
          verdict_for_chain(chain, "I cannot do this")["verified"], None)
    check("unsafe is unverified",
          verdict_for_chain(chain, "import os\nprint(72)\n```")["verified"], None)

    print("--- decide_pool (reweighting, never replacing) ---")
    no_evidence = decide_pool([
        {"answer": "2190", "verified": None},
        {"answer": "1825", "verified": None},
        {"answer": "1825", "verified": None},
    ])
    check("no evidence == Rung 0 plurality", no_evidence["answer"], "1825")
    check("no evidence route", no_evidence["route"], "no_evidence")
    check("no evidence unchanged", no_evidence["changed"], False)
    flip = decide_pool([
        {"answer": "2190", "verified": True},
        {"answer": "2190", "verified": True},
        {"answer": "1825", "verified": False},
        {"answer": "1825", "verified": False},
        {"answer": "1825", "verified": None},
    ])
    check("evidence can overturn a plurality", flip["answer"], "2190")
    check("flip is flagged", flip["changed"], True)
    resists = decide_pool([
        {"answer": "1825", "verified": None},
        {"answer": "1825", "verified": None},
        {"answer": "1825", "verified": None},
        {"answer": "1825", "verified": None},
        {"answer": "2190", "verified": True},
    ])
    check("one verdict cannot beat a strong plurality", resists["answer"], "1825")
    tie_break = decide_pool([
        {"answer": "2190", "verified": None},
        {"answer": "2190", "verified": None},
        {"answer": "1825", "verified": None},
        {"answer": "1825", "verified": None},
    ])
    check("weight ties fall back to plain vote", tie_break["answer"] in {"2190", "1825"}, True)
    check("empty pool", decide_pool([])["answer"], None)

    print("--- end-to-end on a synthetic split pool ---")
    # Multi-step chains: an intermediate exists to corroborate against.
    good_chain = ("6 * 300 = <<6*300=1800>>1800 plus 6 * 65 = "
                  "<<6*65=390>>390 gives <<1800+390=2190>>2190\n#### 2190")
    bad_chain = ("5 * 300 = <<5*300=1500>>1500 plus 5 * 65 = "
                 "<<5*65=325>>325 gives <<1500+325=1825>>1825\n#### 1825")
    pool = [
        {"answer": "2190", "chain_text": good_chain,
         "output": "print(6*300)\nprint(6*65)\nprint(1800+390)\n```"},
        {"answer": "1825", "chain_text": bad_chain,
         # faithful to the intermediates, but the sum contradicts the answer
         "output": "print(5*300)\nprint(5*65)\nprint(1500+325+100)\n```"},
        {"answer": "1825", "chain_text": bad_chain,
         "output": "import os\n```"},      # unsafe -> unverified
    ]
    for entry in pool:
        entry.update(verdict_for_chain(entry, entry["output"]))
    check("chain 1 verified", pool[0]["verified"], True)
    check("chain 2 refuted", pool[1]["verified"], False)
    check("chain 3 unverified", pool[2]["verified"], None)
    decision = decide_pool(pool)
    # 2190: 1.5 (verified). 1825: 0.5 (refuted) + 1.0 (unverified) = 1.5.
    # A tie, broken by the plain vote -> 1825 holds. One verdict against two
    # votes is deliberately NOT enough; that is the safety property.
    check("single verdict does not overturn a 2-1 vote", decision["answer"], "1825")
    check("no change flagged", decision["changed"], False)
    # Refute the second vote as well and the pool flips.
    pool[2].update(verdict_for_chain(
        pool[2], "print(5*300)\nprint(5*65)\nprint(1500+325+100)\n```"))
    check("chain 3 now refuted", pool[2]["verified"], False)
    flipped = decide_pool(pool)
    check("two refutations overturn the vote", flipped["answer"], "2190")
    check("flip flagged", flipped["changed"], True)
    print("  (1.5 vs 1.5 holds with Rung 0; refute both and 2190 wins 1.5 to 1.0)")

    print("--- prompt construction ---")
    prompt = build_prompt("6 * 365 = 2190\n#### 2190")
    check("prompt ends in code fence", prompt.rstrip().endswith("```python"), True)
    check("prompt contains chain", "6 * 365 = 2190" in prompt, True)
    check("prompt truncates", len(build_prompt("x" * 5000)) < 3000, True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def dry_run(limit_per_tier: int) -> None:
    """Load real data and verify shapes/costs without touching a GPU."""
    items, _ = load_split_pools(limit_per_tier=limit_per_tier)
    print(f"split-pool items loaded: {len(items)}")
    by_tier = Counter(i["tier"] for i in items)
    print(f"by tier: {dict(by_tier)}")
    chains = sum(len(i["chains"]) for i in items)
    print(f"chains to transcribe: {chains} "
          f"(mean {chains/max(len(items),1):.1f} per pool)")
    lengths = [len(c["chain_text"]) for i in items for c in i["chains"]]
    print(f"chain chars: median {int(st.median(lengths))}, "
          f"p95 {int(sorted(lengths)[int(0.95*len(lengths))])}, max {max(lengths)}")
    sample = items[0]
    print(f"\nexample pool {sample['instance_id']} ({sample['tier']}, "
          f"share {sample['pool_share']:.2f}, gold {sample['gold']}):")
    print(f"  answers: {Counter(c['answer'] for c in sample['chains'])}")
    print(f"  rung0 plurality: {sample['rung0_plurality']}")
    print("\n--- example prompt (first 600 chars) ---")
    print(build_prompt(sample["chains"][0]["chain_text"])[:600])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-tier", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--out", default=str(OUT_DIR / "rung2_results.json"))
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.dry_run:
        dry_run(args.limit_per_tier or 20)
        return

    from vllm import LLM, SamplingParams

    items, _ = load_split_pools(limit_per_tier=args.limit_per_tier)
    prompts, index = [], []
    for item_index, item in enumerate(items):
        for chain_index, chain in enumerate(item["chains"]):
            prompts.append(build_prompt(chain["chain_text"]))
            index.append((item_index, chain_index))
    print(f"transcribing {len(prompts)} chains from {len(items)} split pools",
          flush=True)

    llm = LLM(model=MODEL_DIR, dtype="bfloat16", trust_remote_code=False,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                              stop=["```", "\n[Q]", "\nSolution:"])
    outputs = llm.generate(prompts, sampling)

    for (item_index, chain_index), output in zip(index, outputs):
        chain = items[item_index]["chains"][chain_index]
        chain["instance_id"] = items[item_index]["instance_id"]
        chain.update(verdict_for_chain(chain, output.outputs[0].text))

    for item in items:
        item["decision"] = decide_pool(item["chains"])

    results = report(items)
    Path(args.out).write_text(json.dumps(results, indent=2))
    detail = Path(args.out).with_name("rung2_items.jsonl")
    with detail.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({
                "instance_id": item["instance_id"], "tier": item["tier"],
                "gold": item["gold"], "rung0_plurality": item["rung0_plurality"],
                "decision": item["decision"],
                "chains": [{k: v for k, v in c.items() if k != "chain_text"}
                           for c in item["chains"]],
            }, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out} and {detail}")


if __name__ == "__main__":
    main()
