"""Stage 20 — VERIFICATION BY GENERATION: continuation consistency (14B, vLLM).

THE PIVOT (from six measured negatives, stages 8/18/19 + rungs 2/3/5 + 8B):
every judge so far was the FROZEN model READING one forward pass as a critic;
all converge to content-signal AUC ~0.6-0.7. The model's strong channel is
GENERATION (raw accuracy 80-90% high-resource; consensus +35 pts). So stop
asking "is this chain correct?" and ask the generator to FINISH the chain:

  score(chain) = fraction of k=4 sampled continuations of the chain's
                 60%-prefix that land on the chain's own claimed answer.

A chain whose reasoning is sound leads regenerations back to its answer; a
chain with a broken step does not reliably re-derive its own conclusion.
Label-blind: the target is the chain's own answer, never gold; demos are
label-blind landslide chains (format anchors). Gold enters only in AUC eval.

Scored on the exact stage-8 gate sample for comparability.

PRE-REGISTERED READS:
  * contested AUC >= 0.70 -> first judge to pass the standalone gate;
    generation channel beats the frozen-critic ceiling.
  * harvest genuine-vs-contaminated AUC > 0.677 (J3's number) -> better
    SFT filter than the verdict judge.
  * both below -> the critic ceiling extends to self-regeneration: the
    model cannot verify its own reasoning through ANY self-channel at 14B.
Reference points: J3 0.636 contested / 0.677 harvest; LR2 0.536; peer-judge
residual 0.6 (its 0.92 headline was answer-matching, computable exactly).

CONFOUND FLAGGED PER CHAIN: if the claimed answer's normalized value already
appears in the prefix, continuation may copy rather than re-derive
(leak=true); AUCs reported for all chains and for leak-free chains.

Usage: --self-test | --dry-run | full via sbatch (stage20_cont.sbatch).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from .verdict_judge_gate import (SEED, auc, build_demo_bank, gate_sample,
                               load_pools, match, norm)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
K_SAMPLES = 4
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 400
PREFIX_FRACTION = 0.6
PREFIX_MAX_CHARS = 1200
DEMO_MAX_CHARS = 700
MAX_MODEL_LEN = 4096
ANSWER_RE = re.compile(r"####\s*([^\n]+)")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def truncate_chain(text):
    """Prefix = first ~60% of body lines (#### lines excluded), always
    cutting at least one body line. None if the chain is too short (<2
    body lines) to leave both a prefix and something to regenerate."""
    body = [l for l in text.split("\n") if l.strip()
            and not l.lstrip().startswith("####")]
    if len(body) < 2:
        return None
    n_keep = min(len(body) - 1, max(1, round(PREFIX_FRACTION * len(body))))
    prefix = "\n".join(body[:n_keep])
    return prefix[:PREFIX_MAX_CHARS]


def extract_answer(continuation):
    """'#### N' if present, else the last number in the continuation."""
    m = ANSWER_RE.search(continuation)
    if m:
        return norm(m.group(1).strip().rstrip("."))
    numbers = NUMBER_RE.findall(continuation)
    return norm(numbers[-1]) if numbers else None


def leaks(prefix, answer):
    """Does the claimed answer's value already appear in the prefix?"""
    if answer is None:
        return False
    return any(match(norm(n), answer) for n in NUMBER_RE.findall(prefix))


def build_cont_prompt(question, prefix, demos):
    """Few-shot completion prompt: 2 label-blind full demos anchor the
    format (solutions end '#### N'), then the target question + prefix,
    ending mid-solution so the model continues."""
    blocks = []
    for demo, label in demos:
        if not label:
            continue                       # positives only: format anchors
        blocks.append("[Q]\n" + demo["question"].strip() + "\n[A]\n"
                      + demo["chain_text"].strip()[:DEMO_MAX_CHARS])
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n" + prefix + "\n")
    return "\n\n".join(blocks)


def consistency(continuations, claimed):
    """Fraction of parseable continuations matching the chain's claim."""
    answers = [extract_answer(c) for c in continuations]
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, 0
    return sum(match(a, claimed) for a in answers) / len(answers), len(answers)


SLICES = (("ALL", lambda s: True),
          ("contested", lambda s: s["contested"]),
          ("landslide", lambda s: not s["contested"]),
          ("high", lambda s: s["tier"] == "high"),
          ("mid", lambda s: s["tier"] == "mid"),
          ("low", lambda s: s["tier"] == "low"))


def report(sample, key="cc"):
    scored = [s for s in sample if s.get(key) is not None]
    mu = defaultdict(list)
    for s in scored:
        mu[s["lang"]].append(s[key])
    mu = {l: st.mean(v) for l, v in mu.items()}
    results = {}
    for subset, rows in (("all_chains", scored),
                         ("leak_free", [s for s in scored if not s["leak"]])):
        block = {}
        print("=" * 66)
        print(f"SUBSET {subset}  ({len(rows)} chains)")
        print(f"{'slice':<12}{'n_gold':>8}{'n_rest':>8}{'AUC raw':>10}{'AUC centered':>14}")
        for label, pred in SLICES:
            g = [s for s in rows if pred(s) and s["is_gold"]]
            r = [s for s in rows if pred(s) and not s["is_gold"]]
            raw = auc([s[key] for s in g], [s[key] for s in r])
            cen = auc([s[key] - mu[s["lang"]] for s in g],
                      [s[key] - mu[s["lang"]] for s in r])
            block[label] = {"auc": round(raw, 3) if raw is not None else None,
                            "auc_centered": round(cen, 3) if cen is not None else None,
                            "n_gold": len(g), "n_rest": len(r)}
            print(f"{label:<12}{len(g):>8}{len(r):>8}"
                  f"{raw if raw is not None else float('nan'):>10.3f}"
                  f"{cen if cen is not None else float('nan'):>14.3f}")
        results[subset] = block
    return results


def harvest_auc(pools, sample, key="cc"):
    """Genuine vs contaminated split on landslide pools (harvest_evidence
    logic): the filter target voting cannot see."""
    genuine, contaminated = [], []
    for s in sample:
        if s.get(key) is None or s["contested"]:
            continue
        pool = pools[s["instance_id"]]
        commit_right = match(pool["plurality"], pool["gold"])
        if commit_right and s["is_gold"]:
            genuine.append(s[key])
        elif not commit_right and not s["is_gold"]:
            contaminated.append(s[key])
    score = auc(genuine, contaminated)
    print(f"\nHARVEST: {len(genuine)} genuine vs {len(contaminated)} contaminated"
          f" -> AUC {score if score else float('nan'):.3f} (J3 reference 0.677)")
    return {"auc": round(score, 3) if score is not None else None,
            "n_genuine": len(genuine), "n_contaminated": len(contaminated)}


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- truncate_chain ---")
    five = "s1\ns2\ns3\ns4\ns5\n#### 9"
    check("60% of 5 lines", truncate_chain(five), "s1\ns2\ns3")
    check("2 lines -> keep 1", truncate_chain("a\nb\n#### 3"), "a")
    check("1 line -> None", truncate_chain("only\n#### 3"), None)
    check("#### excluded from body", truncate_chain("a\n#### 3"), None)

    print("--- extract_answer ---")
    check("marker wins", extract_answer("so 5+2=7\n#### 7"), "7")
    check("marker with comma", extract_answer("#### 1,234"), "1234")
    check("fallback last number", extract_answer("we get 12 then 15."), "15")
    check("no numbers -> None", extract_answer("no digits here"), None)

    print("--- leaks ---")
    check("leak detected", leaks("value is 42 so far", "42"), True)
    check("no leak", leaks("value is 41 so far", "42"), False)
    check("intermediate 4 vs 42 no leak", leaks("4 + 2", "42"), False)

    print("--- consistency ---")
    score, n = consistency(["x\n#### 7", "y\n#### 7", "z\n#### 8", "gibberish"],
                           "7")
    check("2/3 parseable match", round(score, 3), 0.667)
    check("parsed count", n, 3)
    check("none parseable", consistency(["??", "!!"], "7"), (None, 0))

    print("--- prompt ---")
    demos = [({"question": "DQ", "chain_text": "DC\n#### 1"}, True),
             ({"question": "NQ", "chain_text": "NC\n#### 2"}, False)]
    prompt = build_cont_prompt("Q?", "step1\nstep2", demos)
    check("positive demo included", "DC" in prompt, True)
    check("negative demo excluded", "NC" not in prompt, True)
    check("ends open after prefix", prompt.endswith("step2\n"), True)

    print("--- smoke pools produce prefixes ---")
    pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-smoke")
    some = next(iter(pools.values()))
    prefixes = [truncate_chain(c["text"]) for c in some["chains"]]
    check("most chains truncatable",
          sum(p is not None for p in prefixes) >= len(prefixes) // 2, True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load_pools(args.root)
    rng = random.Random(SEED)
    demos = build_demo_bank(pools, rng)      # same rng order -> same sample
    sample = [s for s in gate_sample(pools, rng)
              if s["instance_id"] not in
              {d["iid"] for ds in demos.values() for d, _ in ds}]
    for s in sample:                          # recover each chain's claim
        txt2ans = {c["text"]: c["answer"] for c in pools[s["instance_id"]]["chains"]}
        s["claim"] = txt2ans.get(s["text"])
        s["prefix"] = truncate_chain(s["text"])
        s["leak"] = (leaks(s["prefix"], s["claim"])
                     if s["prefix"] is not None else False)
    usable = [s for s in sample if s["prefix"] is not None
              and s["claim"] is not None]
    print(f"pools {len(pools)}; gate sample {len(sample)}; "
          f"usable (>=2 body lines) {len(usable)}; "
          f"leak-flagged {sum(s['leak'] for s in usable)} "
          f"({100*sum(s['leak'] for s in usable)/len(usable):.0f}%)")

    prompts = [build_cont_prompt(s["question"], s["prefix"],
                                 demos.get(s["lang"], [])) for s in usable]
    if args.dry_run:
        lengths = sorted(len(p) for p in prompts)
        print(f"prompt chars: median {lengths[len(lengths)//2]}, "
              f"p95 {lengths[int(0.95*len(lengths))]}, max {lengths[-1]}")
        print(f"\n--- sample prompt (first 700 chars) ---\n{prompts[0][:700]}")
        print(f"\n--- its prefix cut ---\nclaim={usable[0]['claim']} "
              f"leak={usable[0]['leak']}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=K_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]"],
                            seed=SEED)
    print(f"generating {len(prompts)} x {K_SAMPLES} continuations", flush=True)
    outputs = llm.generate(prompts, params)
    parse_stats = defaultdict(lambda: [0, 0])
    for s, out in zip(usable, outputs):
        continuations = [o.text for o in out.outputs]
        s["cc"], n_parsed = consistency(continuations, s["claim"])
        parse_stats[s["lang"]][0] += n_parsed
        parse_stats[s["lang"]][1] += len(continuations)

    print("\nparse rate per language:")
    for lang in sorted(parse_stats):
        parsed, total = parse_stats[lang]
        print(f"  {lang:<6} {100*parsed/max(total,1):.0f}%")

    results = report(usable)
    results["harvest"] = harvest_auc(pools, usable)
    (OUT_DIR / "stage20_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage20_chains.jsonl").open("w", encoding="utf-8") as handle:
        for s in usable:
            handle.write(json.dumps(
                {k: v for k, v in s.items()
                 if k not in ("text", "question", "prefix")},
                ensure_ascii=False) + "\n")
    print("wrote stage20_results.json and stage20_chains.jsonl")


if __name__ == "__main__":
    main()
