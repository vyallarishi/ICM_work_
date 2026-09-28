"""Stage 24 — SEED ROBUSTNESS of generative MP (stages 22/23).

The recovery result rests on one sampling seed per scale. This sweep reruns
generation at 5 seeds for BOTH scales (prompts fixed; only vLLM sampling
varies), re-decides per seed, and reports mean +- sd of every headline
metric. Prompts, converged state, peers, and containment identical across
seeds by construction.

PRE-REGISTERED: the result is seed-robust if recovery precision stays
>= 95% at every seed and recovery rate varies by < 15 points; harm commits
(wrong answers actually committed, not withdrawals) stay <= a handful.

Usage: --self-test | --dry-run | full via stage24_seeds.sbatch (vLLM venv).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17            # patches gate.TIER to 37
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context, metrics
from ..q02_can_the_model_judge_itself.continuation_consistency_judge import extract_answer
from ..q02_can_the_model_judge_itself.ensemble_judge_in_loop import static_pos_demos
from .regeneration_15lang import (M_SAMPLES, MAX_MODEL_LEN, MAX_NEW_TOKENS,
                           TEMPERATURE, TOP_P, build_regen_prompt,
                           converge_w0, select_regen_peers, weighted_decide)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
SEEDS = (11, 22, 33, 44, 55)
ALPHA = 0.5
BASE_SEED = 20260816
ROOTS = {
    "15lang": "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA",
    "37lang": "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37",
}


def load(root_key):
    root = ROOTS[root_key]
    pools = s17.load_pools(root)
    if root_key == "37lang":
        pools = s17.apply_diverse_views(pools)
    return pools


def build_jobs(pools, offset):
    state, contested, oscillators = converge_w0(pools)
    demos = static_pos_demos(pools)
    rng = random.Random(BASE_SEED + offset)
    jobs = []
    for iid in contested:
        peers = select_regen_peers(pools, state, iid, rng)
        if not peers:
            continue
        jobs.append((iid, build_regen_prompt(pools[iid]["question"], peers,
                                             demos.get(pools[iid]["lang"], []))))
    return state, contested, oscillators, jobs


def decide(pools, state, contested, oscillators, regen):
    new_state = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        others = world_context(pools, new_state, iid)
        new_state[iid]["answer"] = weighted_decide(
            pools[iid], others, regen.get(iid, []), ALPHA,
            iid in oscillators)
    row = metrics(pools, new_state, contested)
    abst = [i for i in contested if state[i]["answer"] is None]
    recovered = [i for i in abst if new_state[i]["answer"] is not None]
    rec_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                 for i in recovered)
    wrong_commits = flips_withdrawn = 0
    for iid in contested:
        if state[iid]["answer"] is None:
            continue
        was_ok = match(state[iid]["answer"], pools[iid]["gold"])
        now = new_state[iid]["answer"]
        if was_ok and now is None:
            flips_withdrawn += 1
        elif was_ok and not match(now, pools[iid]["gold"]):
            wrong_commits += 1
    wrong_commits += len(recovered) - rec_ok
    row.update({"recovered": len(recovered), "recovered_correct": rec_ok,
                "recovery_precision": round(rec_ok / max(len(recovered), 1), 4),
                "wrong_commits_added": wrong_commits,
                "correct_withdrawn": flips_withdrawn,
                "abstained_before": len(abst)})
    return row


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    check("TIER is 37 after s17 import", len(gate.TIER), 37)
    check("5 seeds", len(SEEDS), 5)
    check("seeds distinct", len(set(SEEDS)), 5)
    pool = {"chains": [{"answer": "30"}]}
    check("machinery reachable",
          weighted_decide(pool, ["99"] * 10, ["99"] * 3, 0.5, False), "99")
    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    prepared = {}
    for key, offset in (("15lang", 22), ("37lang", 23)):
        pools = load(key)
        state, contested, oscillators, jobs = build_jobs(pools, offset)
        prepared[key] = (pools, state, contested, oscillators, jobs)
        print(f"[{key}] pools {len(pools)}; contested {len(contested)}; "
              f"jobs {len(jobs)}; baseline "
              f"{metrics(pools, state, contested)}")
    if args.dry_run:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=BASE_SEED)
    results = {}
    for key, (pools, state, contested, oscillators, jobs) in prepared.items():
        rows = []
        for seed in SEEDS:
            params = SamplingParams(n=M_SAMPLES, temperature=TEMPERATURE,
                                    top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
                                    stop=["[Q]", "[S]"], seed=seed)
            outputs = llm.generate([p for _, p in jobs], params)
            regen = {}
            for (iid, _), out in zip(jobs, outputs):
                answers = [extract_answer(o.text) for o in out.outputs]
                regen[iid] = [a for a in answers if a is not None]
            with (OUT_DIR / f"stage24_regen_{key}_s{seed}.jsonl").open(
                    "w", encoding="utf-8") as handle:
                for iid, answers in regen.items():
                    handle.write(json.dumps({"iid": iid, "answers": answers},
                                            ensure_ascii=False) + "\n")
            row = decide(pools, state, contested, oscillators, regen)
            row["seed"] = seed
            rows.append(row)
            print(f"[{key} seed {seed}] {row}", flush=True)
        summary = {}
        for metric in ("coverage", "precision", "recovered",
                       "recovery_precision", "wrong_commits_added",
                       "correct_withdrawn"):
            vals = [r[metric] for r in rows]
            summary[metric] = {"mean": round(st.mean(vals), 4),
                               "sd": round(st.pstdev(vals), 4),
                               "min": min(vals), "max": max(vals)}
        results[key] = {"per_seed": rows, "summary": summary}
        print(f"[{key}] SUMMARY: {json.dumps(summary, indent=2)}", flush=True)
    (OUT_DIR / "stage24_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage24_results.json")


if __name__ == "__main__":
    main()
