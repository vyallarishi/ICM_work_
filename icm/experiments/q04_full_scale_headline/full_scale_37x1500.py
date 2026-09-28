"""Stage 25 — THE FROZEN PIPELINE AT SCALE: 37 languages x 1500 problems.

Runs the validated stack on qwen3-14b-full37-1500 (55,500 pools, ~1M chains):
  converge (pure voting, diverse-8 views) -> one generative-MP round
  (stage22/23 semantics: late seeding, minority peer, alpha=0.5, T=1,
  oscillator unanimity, single round) -> decide -> metrics.

Engineering change, semantics-identical: world_context is O(all pools) per
call in stage9; at 55k pools the converge phase would be quadratic. This
file uses a problem-indexed world context. --self-test PROVES equivalence
by reproducing stage17/23's converged numbers on the 150-problem root
bit-for-bit before the new root is touched.

Artifacts are tagged (stage25_*) -- nothing from the 150-problem runs is
overwritten.

Usage: --self-test (login-safe, ~2 min) | --phase gen [--dry-run]
       | --phase decide | via stage25_scale.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17            # patches gate.TIER to 37
from ..q02_can_the_model_judge_itself import verdict_judge_gate as gate
from ..q02_can_the_model_judge_itself.verdict_judge_gate import match
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool, metrics
from ..q02_can_the_model_judge_itself.continuation_consistency_judge import extract_answer
from ..q02_can_the_model_judge_itself.ensemble_judge_in_loop import static_pos_demos
from ..q03_does_regeneration_help.regeneration_15lang import (M_SAMPLES, MAX_MODEL_LEN, MAX_NEW_TOKENS,
                           TEMPERATURE, TOP_P, build_regen_prompt,
                           select_regen_peers, weighted_decide)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = ("/scratch/rvyalla/multilingual-icm/results/"
                "qwen3-14b-full37-1500")
OLD_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
REGEN_CACHE = OUT_DIR / "stage25_regen.jsonl"
ALPHA = 0.5
ITERATIONS = 4
SEED = 20260816


def load(root):
    return s17.apply_diverse_views(s17.load_pools(root))


class World:
    """Problem-indexed world context; semantics identical to
    stage9.world_context (excludes target pool's language, skips NULLs)."""

    def __init__(self, pools):
        self.by_problem = defaultdict(list)
        for iid, pool in pools.items():
            self.by_problem[iid.rsplit("_", 1)[-1]].append(iid)
        self.pools = pools

    def context(self, state, iid):
        lang = self.pools[iid]["lang"]
        out = []
        for other in self.by_problem[iid.rsplit("_", 1)[-1]]:
            if other == iid or self.pools[other]["lang"] == lang:
                continue
            answer = state[other]["answer"]
            if answer is not None:
                out.append(answer)
        return out


def converge(pools, world):
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    history = {i: [state[i]["answer"]] for i in contested}
    for _ in range(ITERATIONS):
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            new_state[iid]["answer"] = update_pool(
                pools[iid], world.context(state, iid), None, 0.0)
        state = new_state
        for iid in contested:
            history[iid].append(state[iid]["answer"])
    oscillators = {i for i in contested
                   if sum(a != b for a, b in zip(history[i], history[i][1:]))
                   >= 2}
    return state, contested, oscillators


def self_test():
    """Equivalence proof: indexed world context reproduces the recorded
    stage17/stage23 numbers on the 150-problem root exactly."""
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    pools = load(OLD_ROOT)
    world = World(pools)
    state, contested, oscillators = converge(pools, world)
    m = metrics(pools, state, contested)
    check("contested count", len(contested), 1177)
    check("converged metrics reproduce stage17/23 exactly",
          m, {"contested_answered": 1067, "contested_accuracy": 0.85,
              "coverage": 0.9802, "precision": 0.9522})
    check("oscillator count reproduces stage23", len(oscillators), 13)
    abstained = sum(1 for i in contested if state[i]["answer"] is None)
    check("abstained count reproduces stage23", abstained, 110)
    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED (indexed world context is equivalent)")


def phase_gen(args):
    pools = load(args.root)
    world = World(pools)
    state, contested, oscillators = converge(pools, world)
    print(f"converged baseline: {metrics(pools, state, contested)}")
    demos = static_pos_demos(pools)
    rng = random.Random(SEED + 25)
    jobs = []
    for iid in contested:
        peers = select_regen_peers(pools, state, iid, rng)
        if not peers:
            continue
        jobs.append((iid, build_regen_prompt(pools[iid]["question"], peers,
                                             demos.get(pools[iid]["lang"], []))))
    abstained = [i for i in contested if state[i]["answer"] is None]
    print(f"pools {len(pools)}; contested {len(contested)} "
          f"(abstained {len(abstained)}, oscillators {len(oscillators)}); "
          f"regen jobs {len(jobs)}")
    if args.dry_run:
        print(f"\n--- sample prompt (first 300) ---\n{jobs[0][1][:300]}")
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
    pools = load(args.root)
    world = World(pools)
    state, contested, oscillators = converge(pools, world)
    base = metrics(pools, state, contested)
    print(f"converged baseline: {base}")
    regen = {}
    for line in REGEN_CACHE.open(encoding="utf-8"):
        row = json.loads(line)
        regen[row["iid"]] = row["answers"]

    new_state = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        others = world.context(new_state, iid)
        new_state[iid]["answer"] = weighted_decide(
            pools[iid], others, regen.get(iid, []), ALPHA,
            iid in oscillators)
    row = metrics(pools, new_state, contested)

    abst = [i for i in contested if state[i]["answer"] is None]
    recovered = [i for i in abst if new_state[i]["answer"] is not None]
    rec_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                 for i in recovered)
    wrong_commits = withdrawn = 0
    for iid in contested:
        if state[iid]["answer"] is None:
            continue
        was_ok = match(state[iid]["answer"], pools[iid]["gold"])
        now = new_state[iid]["answer"]
        if was_ok and now is None:
            withdrawn += 1
        elif was_ok and not match(now, pools[iid]["gold"]):
            wrong_commits += 1
    tier_rows = {}
    for tier in ("high", "mid", "low"):
        t_abst = [i for i in abst if gate.TIER[pools[i]["lang"]] == tier]
        t_rec = [i for i in t_abst if new_state[i]["answer"] is not None]
        t_ok = sum(match(new_state[i]["answer"], pools[i]["gold"])
                   for i in t_rec)
        tier_rows[tier] = {"abstained": len(t_abst), "recovered": len(t_rec),
                           "correct": t_ok}
    row.update({"abstained_before": len(abst), "recovered": len(recovered),
                "recovered_correct": rec_ok,
                "recovery_precision": round(rec_ok / max(len(recovered), 1), 4),
                "wrong_commits_added": wrong_commits + len(recovered) - rec_ok,
                "correct_withdrawn": withdrawn,
                "recovery_by_tier": tier_rows})
    results = {"baseline": base, "with_regen": row}
    (OUT_DIR / "stage25_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(row, indent=2))
    print("wrote stage25_results.json")


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
