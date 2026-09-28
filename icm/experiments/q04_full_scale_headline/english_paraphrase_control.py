"""Stage 31 — E3: THE MONOLINGUAL PARAPHRASE CONTROL.

THE ATTACK (hostile review, K4/E3): "37 languages" may be doing no work.
If 37 English PARAPHRASES of each problem produce the same consensus gains,
then the mechanism is ensembling over prompt diversity, not cross-lingual
transfer, and every cross-lingual framing in the paper is unsupported.

DESIGN (matched as tightly as possible to the real pipeline):
  * same 150 problems as qwen3-14b-phaseA
  * 37 English "views" per problem = 1 original + 36 model-written
    paraphrases (generated once, T=0.9, deduped, answer-preserving by
    construction: paraphrase the WORDING, never the numbers -- validated by
    checking that the question's numeric multiset is preserved; failures are
    resampled, then dropped)
  * per view, the same generation regimes as the real pools (T=0.2/0.5/0.8),
    ~20 chains, diverse-8 view applied identically
  * the IDENTICAL decision stack: initialize -> 4-lap converge -> metrics,
    with each paraphrase treated as a "language"
  * REAL 37-language arm recomputed here on the same 150 problems so the
    two numbers are produced by one code path

PRE-REGISTERED:
  H0 (review's hypothesis): paraphrase coverage/precision within 1 point of
     the cross-lingual arm -> "cross-lingual" is not doing the work; the
     paper must be reframed as diversity ensembling.
  H1 (ours): cross-lingual materially exceeds paraphrase, because different
     languages fail differently while paraphrases share the English prior.
  Secondary: same-wrong-answer rate (the 25-30x correlated-error lift).
     Prediction: paraphrase views are MORE correlated than languages. This
     is the mechanistic claim; if paraphrases are equally decorrelated, H1
     loses even if headline numbers differ.

Usage: --self-test | --phase paraphrase [--dry-run] | --phase chains
       | --phase decide | via stage31_para.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match, norm
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool, metrics
from ..q01_can_voting_label_data.consensus_loop_37lang import diverse8_view
from .full_scale_37x1500 import World

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
PHASEA_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
FULL37_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
PARA_CACHE = OUT_DIR / "stage31_paraphrases.jsonl"
CHAIN_CACHE = OUT_DIR / "stage31_chains.jsonl"
N_VIEWS = 37                 # 1 original + 36 paraphrases
REGIMES = ((0.2, 8), (0.5, 8), (0.8, 4))     # matches best_regimes_v2
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 400
PARA_MAX_TOKENS = 320
PARA_TEMPERATURE = 0.9
TOP_P = 0.95
ITERATIONS = 4
SEED = 20260824
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numeric_multiset(text):
    """Numbers in a question, normalized: a paraphrase must preserve these
    exactly or it may have changed the problem."""
    out = []
    for tok in NUMBER_RE.findall(text):
        v = norm(tok)
        if v is not None:
            out.append(v)
    return sorted(out)


def paraphrase_ok(original, candidate):
    """Answer-preserving check: same numbers, different wording, sane length."""
    if not candidate or len(candidate) < 20:
        return False
    if numeric_multiset(original) != numeric_multiset(candidate):
        return False
    if candidate.strip() == original.strip():
        return False
    return len(candidate) < 3 * len(original) + 200


def build_para_prompt(question, k):
    """Few-shot-free instruction for a base model: restate the problem."""
    return ("Rewrite the following math word problem in different words. "
            "Keep every number and the question exactly the same; change "
            "only the wording, names, and sentence structure.\n\n"
            f"Original problem:\n{question.strip()}\n\n"
            f"Rewritten version {k}:\n")


def build_chain_prompt(question, demos):
    blocks = []
    for demo_q, demo_c in demos:
        blocks.append("[Q]\n" + demo_q.strip() + "\n[A]\n" + demo_c.strip())
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n")
    return "\n\n".join(blocks)


def english_demos(pools, n=2):
    """2 English chains from landslide pools as format anchors (label-blind,
    same convention as the real pipeline's static demos)."""
    out = []
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["lang"] != "en" or pool["share"] < 0.70:
            continue
        for c in pool["chains"]:
            if c["answer"] == pool["plurality"] and len(c["text"]) <= 700:
                out.append((pool["question"], c["text"]))
                break
        if len(out) >= n:
            break
    return out[:n]


def correlation_stats(view_answers, gold):
    """Same-wrong-answer rate vs null, per the project's correlated-error
    measurement. view_answers: {view_id: answer}."""
    wrong = [(v, a) for v, a in view_answers.items()
             if a is not None and not match(a, gold)]
    pairs = same = 0
    for i in range(len(wrong)):
        for j in range(i + 1, len(wrong)):
            pairs += 1
            same += match(wrong[i][1], wrong[j][1])
    return pairs, same


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- numeric preservation ---")
    q = "Natalia sold 48 clips in April, then half as many in May."
    check("same numbers pass",
          paraphrase_ok(q, "In April, Natalia sold 48 clips; in May she "
                           "sold half that amount."), True)
    check("changed number fails",
          paraphrase_ok(q, "In April, Natalia sold 84 clips; in May she "
                           "sold half that amount."), False)
    check("dropped number fails",
          paraphrase_ok(q, "Natalia sold some clips in April and half as "
                           "many in May."), False)
    check("identical text fails", paraphrase_ok(q, q), False)
    check("too short fails", paraphrase_ok(q, "48 clips"), False)
    check("normalizes 48.0 vs 48",
          numeric_multiset("48 and 48.0"), ["48", "48"])

    print("--- prompts ---")
    p = build_para_prompt("Q text", 3)
    check("asks to keep numbers", "Keep every number" in p, True)
    check("ends open", p.endswith("Rewritten version 3:\n"), True)
    c = build_chain_prompt("Q?", [("DQ", "DC")])
    check("chain prompt ends open", c.endswith("[Q]\nQ?\n[A]\n"), True)

    print("--- correlation stats ---")
    pairs, same = correlation_stats({"v1": "40", "v2": "40", "v3": "5",
                                     "v4": "99"}, "5")
    check("3 wrong -> 3 pairs", pairs, 3)
    check("one matching wrong pair", same, 1)

    print("--- diverse8 reused identically ---")
    chains = [{"answer": str(i % 5), "text": f"c{i}"} for i in range(16)]
    check("view capped at 8", len(diverse8_view("train_v0_1", chains)), 8)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def phase_paraphrase(args):
    pools = load_pools(PHASEA_ROOT)
    problems = {}
    for iid, pool in pools.items():
        if pool["lang"] == "en":
            problems[iid.rsplit("_", 1)[-1]] = pool["question"]
    print(f"english problems: {len(problems)}; "
          f"target views/problem: {N_VIEWS} (1 original + {N_VIEWS-1} paraphrases)")
    jobs = [(prob, k, build_para_prompt(q, k))
            for prob, q in sorted(problems.items())
            for k in range(1, N_VIEWS)]
    print(f"paraphrase jobs: {len(jobs)}")
    if args.dry_run:
        print(f"\n--- sample prompt ---\n{jobs[0][2]}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=1, temperature=PARA_TEMPERATURE, top_p=TOP_P,
                            max_tokens=PARA_MAX_TOKENS,
                            stop=["\n\n", "Original problem:", "Rewritten"],
                            seed=SEED)
    outputs = llm.generate([p for _, _, p in jobs], params)
    kept = defaultdict(list)
    rejected = 0
    for (prob, k, _), out in zip(jobs, outputs):
        text = out.outputs[0].text.strip()
        if paraphrase_ok(problems[prob], text) and not any(
                text == t for t in kept[prob]):
            kept[prob].append(text)
        else:
            rejected += 1
    with PARA_CACHE.open("w", encoding="utf-8") as h:
        for prob in sorted(kept):
            h.write(json.dumps({"problem": prob,
                                "original": problems[prob],
                                "paraphrases": kept[prob]},
                               ensure_ascii=False) + "\n")
    sizes = [len(v) for v in kept.values()]
    print(f"kept {sum(sizes)} paraphrases ({rejected} rejected); "
          f"per problem min {min(sizes)} median "
          f"{sorted(sizes)[len(sizes)//2]} max {max(sizes)}")
    print(f"wrote {PARA_CACHE}")


def phase_chains(args):
    pools = load_pools(PHASEA_ROOT)
    demos = english_demos(pools)
    rows = [json.loads(l) for l in PARA_CACHE.open(encoding="utf-8")]
    jobs = []
    for row in rows:
        views = [row["original"]] + row["paraphrases"][:N_VIEWS - 1]
        for v_idx, question in enumerate(views):
            for temp, n in REGIMES:
                jobs.append((row["problem"], v_idx, question, temp, n))
    print(f"problems {len(rows)}; chain jobs {len(jobs)} "
          f"(views x regimes); demos {len(demos)}")
    if args.dry_run:
        prob, v, q, t, n = jobs[0]
        print(f"\n--- chain prompt (T={t}, n={n}) ---")
        print(build_chain_prompt(q, demos)[:500])
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    by_regime = defaultdict(list)
    for job in jobs:
        by_regime[(job[3], job[4])].append(job)
    with CHAIN_CACHE.open("w", encoding="utf-8") as h:
        for (temp, n), group in sorted(by_regime.items()):
            params = SamplingParams(n=n, temperature=temp, top_p=TOP_P,
                                    max_tokens=MAX_NEW_TOKENS,
                                    stop=["[Q]"], seed=SEED)
            prompts = [build_chain_prompt(q, demos) for _, _, q, _, _ in group]
            print(f"[T={temp} n={n}] generating {len(prompts)} prompts",
                  flush=True)
            outputs = llm.generate(prompts, params)
            for (prob, v_idx, q, _, _), out in zip(group, outputs):
                for o in out.outputs:
                    h.write(json.dumps({"problem": prob, "view": v_idx,
                                        "text": o.text}, ensure_ascii=False)
                            + "\n")
    print(f"wrote {CHAIN_CACHE}")


def run_pipeline(pools, label):
    """The identical decision stack used everywhere else in this project."""
    world = World(pools)
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(ITERATIONS):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_pool(pools[iid],
                                            world.context(state, iid), None, 0.0)
        state = ns
    m = metrics(pools, state, contested)
    print(f"[{label}] {m}  (pools {len(pools)}, contested {len(contested)})")
    return m, state, contested


def phase_decide(args):
    from ..q02_can_the_model_judge_itself.continuation_consistency_judge import extract_answer
    results = {}

    # --- paraphrase arm: build pools in the real schema, then diverse-8
    para_chains = defaultdict(list)
    for line in CHAIN_CACHE.open(encoding="utf-8"):
        r = json.loads(line)
        ans = extract_answer(r["text"])
        if ans is not None and r["text"].strip():
            para_chains[(r["problem"], r["view"])].append(
                {"answer": ans, "text": r["text"]})
    gold = {}
    for iid, pool in load_pools(PHASEA_ROOT).items():
        gold[iid.rsplit("_", 1)[-1]] = pool["gold"]
    para_pools = {}
    for (prob, view), chains in para_chains.items():
        if not chains or prob not in gold:
            continue
        iid = f"train_v{view:02d}_{prob}"
        view_chains = diverse8_view(iid, chains)
        counts = Counter(c["answer"] for c in view_chains)
        para_pools[iid] = {
            "gold": gold[prob], "lang": f"v{view:02d}", "question": "",
            "chains": view_chains,
            "share": counts.most_common(1)[0][1] / len(view_chains),
            "plurality": max(counts, key=lambda a: (counts[a], a))}
    print(f"paraphrase pools built: {len(para_pools)}")
    m_para, state_para, _ = run_pipeline(para_pools, "PARAPHRASE (37 views)")
    results["paraphrase"] = m_para

    # --- cross-lingual arm on the SAME problems, same code path
    real = load_pools(FULL37_ROOT)
    keep = set(gold)
    real = {i: p for i, p in real.items() if i.rsplit("_", 1)[-1] in keep}
    for iid, pool in real.items():
        view = diverse8_view(iid, pool["chains"])
        counts = Counter(c["answer"] for c in view)
        pool["chains"] = view
        pool["share"] = counts.most_common(1)[0][1] / len(view)
        pool["plurality"] = max(counts, key=lambda a: (counts[a], a))
    m_real, state_real, _ = run_pipeline(real, "CROSS-LINGUAL (37 langs)")
    results["cross_lingual"] = m_real

    # --- correlated-error comparison (the mechanistic claim)
    def corr(pools_, state_, tag):
        by_problem = defaultdict(dict)
        for iid, s in state_.items():
            by_problem[iid.rsplit("_", 1)[-1]][pools_[iid]["lang"]] = s["answer"]
        pairs = same = 0
        for prob, views in by_problem.items():
            g = pools_[next(iter(
                i for i in pools_ if i.rsplit("_", 1)[-1] == prob))]["gold"]
            p, s_ = correlation_stats(views, g)
            pairs += p
            same += s_
        rate = same / max(pairs, 1)
        print(f"[{tag}] wrong-pairs {pairs}, same-wrong {same} "
              f"-> same-wrong-answer rate {100*rate:.1f}%")
        return {"wrong_pairs": pairs, "same_wrong": same,
                "same_wrong_rate": round(rate, 4)}
    results["correlation_paraphrase"] = corr(para_pools, state_para, "PARA")
    results["correlation_cross_lingual"] = corr(real, state_real, "XLING")

    gap_cov = m_real["coverage"] - m_para["coverage"]
    gap_prec = m_real["precision"] - m_para["precision"]
    results["verdict"] = {
        "coverage_gap": round(gap_cov, 4), "precision_gap": round(gap_prec, 4),
        "H0_paraphrase_matches": bool(abs(gap_cov) < 0.01
                                      and abs(gap_prec) < 0.01)}
    print(f"\ncoverage gap (xling - para): {gap_cov:+.4f}")
    print(f"precision gap (xling - para): {gap_prec:+.4f}")
    print("H0 (paraphrase matches, cross-lingual does no work): "
          f"{results['verdict']['H0_paraphrase_matches']}")
    (OUT_DIR / "stage31_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage31_results.json")
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("paraphrase", "chains", "decide"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase == "paraphrase":
        phase_paraphrase(args)
    elif args.phase == "chains":
        phase_chains(args)
    elif args.phase == "decide":
        phase_decide(args)
    else:
        parser.error("need --self-test or --phase {paraphrase,chains,decide}")


if __name__ == "__main__":
    main()
