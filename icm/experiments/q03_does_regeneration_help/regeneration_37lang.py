"""Stage 23 — GENERATIVE MP AT FULL SCALE: 37 languages, diverse-8 views.

Field test of stage22 (15 langs: 51/79 abstentions recovered at 98%, zero
harm, precision 95.53 -> 95.63, coverage 96.49 -> 98.76). Same mechanism,
same containment (late seeding, minority peer, fractional votes, oscillator
unanimity bar, single round, regen quarantined from demos/harvest), on the
qwen3-14b-full37 pools under stage17's diverse-8 views.

Reference (stage17 w0 converged): contested 85.0 / coverage 98.02 /
precision 95.22; window37.log: 1177 contested.

PRE-REGISTERED READS (order matters):
  * harm_correct_to_wrong ~ 0 and precision >= 95.22, else the idea fails
    at scale regardless of gains.
  * abstention recovery rate and precision comparable to 15-lang (63% at
    98%); the 37-lang jury is stronger, so world majorities are cleaner --
    recovery precision should hold or rise.
  * re-derivation-as-filter replication: of strong-and-wrong world
    majorities over abstained pools, most should FAIL to regenerate
    (15-lang: 3 of 4 blocked; n was tiny -- this run is the real test).

Usage: --self-test | --phase gen [--dry-run] | --phase decide
       via stage23_regen37.sbatch (vLLM venv; decide is CPU).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17          # patches gate.TIER to 37 langs
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import world_context, metrics
from ..q02_can_the_model_judge_itself.ensemble_judge_in_loop import static_pos_demos
from .regeneration_15lang import (M_SAMPLES, MAX_MODEL_LEN, MAX_NEW_TOKENS, SEED,
                           TEMPERATURE, TOP_P, build_regen_prompt,
                           converge_w0, select_regen_peers, weighted_decide)
from ..q02_can_the_model_judge_itself.continuation_consistency_judge import extract_answer

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
REGEN_CACHE = OUT_DIR / "stage23_regen37.jsonl"
ALPHAS = (0.5,)          # alpha was inert at 15 langs (mechanism is
                         # answer-presence); one arm suffices


def load_pools_37(root):
    return s17.apply_diverse_views(s17.load_pools(root))


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
    check("TIER patched to 37 languages", len(gate.TIER), 37)
    check("ceb present", "ceb" in gate.TIER, True)

    print("--- diverse-8 view invariants ---")
    chains = [{"answer": str(i % 5), "text": f"c{i}"} for i in range(16)]
    view = s17.diverse8_view("train_en_1", chains)
    check("view capped at 8", len(view), 8)
    check("view answer-diverse", len({c["answer"] for c in view}) >= 5, True)

    print("--- stage22 machinery imports intact ---")
    pool = {"chains": [{"answer": "30"}, {"answer": "30"}, {"answer": "10"}]}
    check("weighted_decide works here",
          weighted_decide(pool, ["10"] * 8 + ["30"] * 2, ["10"] * 4, 1.0,
                          False), "10")
    check("veto dissolution works here",
          weighted_decide({"chains": [{"answer": "30"}]}, ["99"] * 10,
                          ["99"] * 3, 0.5, False), "99")

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def phase_gen(args):
    pools = load_pools_37(args.root)
    state, contested, oscillators = converge_w0(pools)
    print(f"converged w0 baseline: {metrics(pools, state, contested)}")
    demos = static_pos_demos(pools)
    rng = random.Random(SEED + 23)
    jobs = []
    for iid in contested:
        peers = select_regen_peers(pools, state, iid, rng)
        if not peers:
            continue
        prompt = build_regen_prompt(pools[iid]["question"], peers,
                                    demos.get(pools[iid]["lang"], []))
        jobs.append((iid, prompt))
    abstained = [i for i in contested if state[i]["answer"] is None]
    print(f"pools {len(pools)}; contested {len(contested)} "
          f"(abstained {len(abstained)}, oscillators {len(oscillators)}); "
          f"regen jobs {len(jobs)}; "
          f"demo banks {sum(1 for d in demos.values() if len(d)==2)}/{len(demos)} full")
    if args.dry_run:
        print(f"\n--- sample regen prompt (first 500) ---\n{jobs[0][1][:500]}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=M_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]", "[S]"],
                            seed=SEED)
    outputs = llm.generate([p for _, p in jobs], params)
    with REGEN_CACHE.open("w", encoding="utf-8") as handle:
        for (iid, _), out in zip(jobs, outputs):
            answers = [extract_answer(o.text) for o in out.outputs]
            handle.write(json.dumps(
                {"iid": iid, "answers": [a for a in answers if a is not None]},
                ensure_ascii=False) + "\n")
    print(f"wrote {REGEN_CACHE} ({len(jobs)} pools)")


def phase_decide(args):
    pools = load_pools_37(args.root)
    state, contested, oscillators = converge_w0(pools)
    base = metrics(pools, state, contested)
    print(f"converged w0 baseline: {base}")
    regen = {}
    for line in REGEN_CACHE.open(encoding="utf-8"):
        row = json.loads(line)
        regen[row["iid"]] = row["answers"]
    print(f"regen cache: {len(regen)} pools, "
          f"{sum(len(a) for a in regen.values())} parsed answers")

    results = {"w0_converged": base}
    for alpha in ALPHAS:
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            others = world_context(pools, new_state, iid)
            new_state[iid]["answer"] = weighted_decide(
                pools[iid], others, regen.get(iid, []), alpha,
                iid in oscillators)
        row = metrics(pools, new_state, contested)

        answered_before = {i for i in contested
                           if state[i]["answer"] is not None}
        flips_cw = flips_wc = 0
        for iid in answered_before:
            was = match(state[iid]["answer"], pools[iid]["gold"])
            now_ans = new_state[iid]["answer"]
            now = now_ans is not None and match(now_ans, pools[iid]["gold"])
            if was and not now:
                flips_cw += 1
            elif not was and now:
                flips_wc += 1
        abst = [i for i in contested if state[i]["answer"] is None]
        recovered = [i for i in abst if new_state[i]["answer"] is not None]
        rec_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                     for i in recovered)
        gold_injected = sum(
            1 for i in abst
            if any(match(a, pools[i]["gold"]) for a in regen.get(i, [])))
        # re-derivation-as-filter replication (the 3-of-4 result, real n)
        wrong_world = blocked = 0
        for iid in abst:
            others = world_context(pools, state, iid)
            if not others:
                continue
            top, n = Counter(others).most_common(1)[0]
            if n / len(others) >= 0.70 and not match(top, pools[iid]["gold"]):
                wrong_world += 1
                new_ans = new_state[iid]["answer"]
                if new_ans is None or not match(new_ans, top):
                    blocked += 1
        # per-tier recovery
        tier_rows = {}
        from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
        for tier in ("high", "mid", "low"):
            t_abst = [i for i in abst if gate.TIER[pools[i]["lang"]] == tier]
            t_rec = [i for i in t_abst if new_state[i]["answer"] is not None]
            t_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                       for i in t_rec)
            tier_rows[tier] = {"abstained": len(t_abst),
                               "recovered": len(t_rec), "correct": t_ok}
        row.update({
            "harm_correct_to_wrong": flips_cw,
            "gain_wrong_to_correct": flips_wc,
            "abstained_before": len(abst),
            "abstained_gold_injected": gold_injected,
            "abstained_recovered": len(recovered),
            "abstained_recovered_correct": rec_ok,
            "wrong_world_majorities": wrong_world,
            "wrong_world_blocked": blocked,
            "recovery_by_tier": tier_rows,
        })
        results[f"alpha_{alpha}"] = row
        print(f"[alpha={alpha}] {json.dumps(row, indent=2)}")
    (OUT_DIR / "stage23_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage23_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("gen", "decide"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase == "gen":
        phase_gen(args)
    elif args.phase == "decide":
        phase_decide(args)
    else:
        parser.error("need --self-test or --phase {gen,decide}")


if __name__ == "__main__":
    main()
