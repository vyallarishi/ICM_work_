"""Stage 26 — S(a): MUTUAL PREDICTABILITY SCORED GENERATIVELY.

THE OBJECTION THIS ANSWERS (advisor): stage22/23's regeneration shows the
model a mixed peer set and asks only "did answer a appear?" -- that is
conditional resampling plus a presence check, and calling it MP oversells
it. Pool-size ablation (rung7) already shows extra UNCONDITIONAL samples
don't substitute (8 -> 18.7 chains buys +0.55 coverage), so conditioning
matters; what is NOT yet measured is whether conditioning is SELECTIVE.

THE MEASUREMENT: for each contested/abstained pool and each candidate
answer a, condition on peers drawn EXCLUSIVELY from cluster a and estimate

    S(a) = P(this language re-derives a | peers that derive a)

by sampling m=8 chains and counting how many land on a. This is ICM's
mutual predictability -- predictability of a label given neighbours
carrying that label -- estimated through generation (the model's strong
channel, 80-90%) instead of verdict logits (measured 0.5-0.65 AUC).

PRE-REGISTERED PREDICTIONS (gold used only in scoring, never in prompts):
  P1 SELECTIVITY: among pools where world-majority is WRONG and the pool
     contains a correct minority answer, S(correct) > S(wrong-majority)
     more often than chance (>60% of such pools). If S is flat (45-55%),
     conditioning is NOT selective, the advisor is right, and the honest
     conclusion is that generation delivers hints rather than verifying.
  P2 CALIBRATION: S(committed answer) is higher for correct commits than
     wrong ones (AUC > 0.70) -- a per-pool confidence signal usable to
     gate SFT harvesting.
  P3 MARGIN RULE: committing argmax_a S(a) only when the top-two margin
     exceeds t improves label accuracy over presence-based recovery at
     matched coverage.
ABLATIONS (same pools, same m):
  ANSWER_ONLY  peers replaced by "another language answers a" (no
               reasoning) -> tests whether the reasoning does the work.
  SHUFFLED     peers derive a but for a DIFFERENT problem -> tests
               whether the model is digit-matching.

Usage: --self-test | --dry-run | full via stage26_sa.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import verdict_judge_gate as gate
from .verdict_judge_gate import auc, load_pools, match
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool
from .continuation_consistency_judge import extract_answer
from .ensemble_judge_in_loop import static_pos_demos
from ..q03_does_regeneration_help.regeneration_15lang import (M_SAMPLES, MAX_MODEL_LEN, MAX_NEW_TOKENS,
                           TEMPERATURE, TOP_P, PEER_MAX_CHARS)
from ..q04_full_scale_headline.full_scale_37x1500 import World

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
M_SAMPLES_SA = 8            # per (pool, candidate) -- S(a) needs resolution
N_PEERS = 3
MAX_CANDIDATES = 2          # top-2 candidates per pool (world top + local top)
ITERATIONS = 4
SEED = 20260819
ARMS = ("full", "answer_only", "shuffled")


def converge(pools, world):
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(ITERATIONS):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_pool(pools[iid], world.context(state, iid),
                                            None, 0.0)
        state = ns
    return state, contested


def candidates_for(pools, state, world, iid):
    """Label-blind candidate answers to score: the world's top answer and
    the pool's own plurality (deduped, max MAX_CANDIDATES)."""
    others = world.context(state, iid)
    out = []
    if others:
        out.append(Counter(others).most_common(1)[0][0])
    local = pools[iid]["plurality"]
    if not any(match(local, a) for a in out):
        out.append(local)
    return out[:MAX_CANDIDATES]


def peers_for_answer(pools, state, iid, answer, rng, problem_override=None):
    """N_PEERS chains from OTHER languages whose committed answer matches
    `answer` (label-blind). problem_override -> SHUFFLED ablation."""
    problem = problem_override or iid.rsplit("_", 1)[-1]
    lang = pools[iid]["lang"]
    hits = []
    for other, s in state.items():
        if other.rsplit("_", 1)[-1] != problem or pools[other]["lang"] == lang:
            continue
        if s["answer"] is None or not match(s["answer"], answer):
            continue
        chains = [c for c in pools[other]["chains"]
                  if match(c["answer"], answer)
                  and len(c["text"]) <= PEER_MAX_CHARS]
        if chains:
            hits.append(min(chains, key=lambda c: len(c["text"]))["text"])
    rng.shuffle(hits)
    return hits[:N_PEERS]


def shuffled_peers(pools, state, iid, problems, rng):
    """SHUFFLED ablation: N_PEERS chains from other languages solving a
    DIFFERENT problem (each internally consistent, but irrelevant here).
    Tests whether the model merely copies digits from context."""
    lang = pools[iid]["lang"]
    for _ in range(12):
        other_problem = rng.choice(problems)
        if other_problem == iid.rsplit("_", 1)[-1]:
            continue
        hits = []
        for other, s in state.items():
            if other.rsplit("_", 1)[-1] != other_problem:
                continue
            if pools[other]["lang"] == lang or s["answer"] is None:
                continue
            chains = [c for c in pools[other]["chains"]
                      if match(c["answer"], s["answer"])
                      and len(c["text"]) <= PEER_MAX_CHARS]
            if chains:
                hits.append(min(chains, key=lambda c: len(c["text"]))["text"])
        if len(hits) >= N_PEERS:
            rng.shuffle(hits)
            return hits[:N_PEERS]
    return []


def build_prompt(question, peers, demos, answer=None, answer_only=False):
    blocks = []
    for demo, label in demos:
        if label:
            blocks.append("[Q]\n" + demo["question"].strip() + "\n[A]\n"
                          + demo["chain_text"].strip()[:PEER_MAX_CHARS])
    if answer_only:
        for _ in range(N_PEERS):
            blocks.append("[S]\nThe answer is " + str(answer) + ".\n#### "
                          + str(answer))
    else:
        for peer in peers:
            blocks.append("[S]\n" + peer.strip()[:PEER_MAX_CHARS])
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n")
    return "\n\n".join(blocks)


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- prompt shapes ---")
    demos = [({"question": "DQ", "chain_text": "DC\n#### 1"}, True)]
    p_full = build_prompt("Q?", ["P1\n#### 5", "P2\n#### 5"], demos)
    check("peers present", p_full.count("[S]"), 2)
    check("ends open", p_full.endswith("[Q]\nQ?\n[A]\n"), True)
    p_ao = build_prompt("Q?", [], demos, answer="5", answer_only=True)
    check("answer_only has no reasoning", "P1" not in p_ao, True)
    check("answer_only states the answer", "The answer is 5" in p_ao, True)
    check("answer_only peer count", p_ao.count("[S]"), N_PEERS)

    print("--- candidate selection is label-blind ---")
    pools = {
        "train_en_1": {"lang": "en", "gold": "5", "question": "q",
                       "share": 0.9, "plurality": "5",
                       "chains": [{"answer": "5", "text": "t"}]},
        "train_de_1": {"lang": "de", "gold": "5", "question": "q",
                       "share": 0.9, "plurality": "5",
                       "chains": [{"answer": "5", "text": "t"}]},
        "train_am_1": {"lang": "am", "gold": "5", "question": "q",
                       "share": 0.75, "plurality": "40",
                       "chains": [{"answer": "40", "text": "t40"},
                                  {"answer": "5", "text": "t5"}]},
    }
    world = World(pools)
    state = {i: {"answer": p["plurality"], "kind": "committed"}
             for i, p in pools.items()}
    cands = candidates_for(pools, state, world, "train_am_1")
    check("two candidates (world 5, local 40)", sorted(cands), ["40", "5"])
    peers5 = peers_for_answer(pools, state, "train_am_1", "5",
                              random.Random(0))
    check("peers for 5 come from other languages", len(peers5), 2)
    peers40 = peers_for_answer(pools, state, "train_am_1", "40",
                               random.Random(0))
    check("no other language derives 40", peers40, [])

    print("--- shuffled peers ---")
    pools2 = dict(pools)
    pools2["train_en_2"] = {"lang": "en", "gold": "9", "question": "q2",
                            "share": 1.0, "plurality": "9",
                            "chains": [{"answer": "9", "text": "nine"}]}
    pools2["train_de_2"] = {"lang": "de", "gold": "9", "question": "q2",
                            "share": 1.0, "plurality": "9",
                            "chains": [{"answer": "9", "text": "neun"}]}
    pools2["train_fr_2"] = {"lang": "fr", "gold": "9", "question": "q2",
                            "share": 1.0, "plurality": "9",
                            "chains": [{"answer": "9", "text": "neuf"}]}
    state2 = {i: {"answer": p["plurality"], "kind": "committed"}
              for i, p in pools2.items()}
    sh = shuffled_peers(pools2, state2, "train_am_1", ["1", "2"],
                        random.Random(3))
    check("shuffled peers come from another problem",
          all(t in ("nine", "neun", "neuf") for t in sh) and len(sh) == 3, True)

    print("--- S(a) arithmetic ---")
    samples = ["x\n#### 5", "y\n#### 5", "z\n#### 40", "w\n#### 5"]
    answers = [extract_answer(s) for s in samples]
    s5 = sum(match(a, "5") for a in answers) / len(answers)
    check("S(5) = 0.75", round(s5, 4), 0.75)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def build_jobs(pools, state, contested, world):
    """(iid, answer, arm) -> prompt, for every candidate of every pool."""
    demos = static_pos_demos(pools)
    rng = random.Random(SEED)
    problems = sorted({i.rsplit("_", 1)[-1] for i in pools})
    jobs = []
    for iid in contested:
        for answer in candidates_for(pools, state, world, iid):
            peers = peers_for_answer(pools, state, iid, answer, rng)
            if len(peers) < 1:
                continue           # cannot condition on this answer
            q = pools[iid]["question"]
            d = demos.get(pools[iid]["lang"], [])
            jobs.append((iid, answer, "full",
                         build_prompt(q, peers, d)))
            jobs.append((iid, answer, "answer_only",
                         build_prompt(q, [], d, answer=answer,
                                      answer_only=True)))
            sh = shuffled_peers(pools, state, iid, problems, rng)
            if sh:
                jobs.append((iid, answer, "shuffled",
                             build_prompt(q, sh, d)))
    return jobs


def analyze(pools, state, contested, scores):
    """scores[(iid, answer, arm)] = S. Evaluate P1/P2/P3."""
    results = {}
    for arm in ARMS:
        pairs = defaultdict(dict)
        for (iid, answer, a), s in scores.items():
            if a == arm:
                pairs[iid][answer] = s
        # P1 selectivity: pools with >=2 candidates, one correct one not
        wins = total = 0
        for iid, per_answer in pairs.items():
            gold = pools[iid]["gold"]
            correct = [a for a in per_answer if match(a, gold)]
            wrong = [a for a in per_answer if not match(a, gold)]
            if not correct or not wrong:
                continue
            total += 1
            if max(per_answer[a] for a in correct) > max(per_answer[a]
                                                         for a in wrong):
                wins += 1
        # P2 calibration: S of the argmax answer, correct vs wrong pools
        pos, neg = [], []
        for iid, per_answer in pairs.items():
            best = max(per_answer, key=lambda a: (per_answer[a], a))
            (pos if match(best, pools[iid]["gold"]) else neg).append(
                per_answer[best])
        block = {
            "n_pools_scored": len(pairs),
            "P1_selectivity_pools": total,
            "P1_correct_wins": wins,
            "P1_win_rate": round(wins / max(total, 1), 4),
            "P2_auc": round(auc(pos, neg), 4) if pos and neg else None,
            "P2_n_correct": len(pos), "P2_n_wrong": len(neg),
        }
        # P3 margin rule: accuracy/coverage frontier over top-2 margin
        rows = []
        for iid, per_answer in pairs.items():
            ranked = sorted(per_answer.items(), key=lambda kv: -kv[1])
            if not ranked:
                continue
            margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
            rows.append((margin, match(ranked[0][0], pools[iid]["gold"])))
        rows.sort(reverse=True)
        frontier = []
        for t in (0.0, 0.125, 0.25, 0.375, 0.5):
            kept = [ok for m, ok in rows if m >= t]
            frontier.append({"margin": t, "kept": len(kept),
                             "coverage": round(len(kept) / max(len(rows), 1), 4),
                             "accuracy": round(sum(kept) / max(len(kept), 1), 4)})
        block["P3_frontier"] = frontier
        results[arm] = block
        print(f"\n=== ARM {arm} ===")
        print(json.dumps(block, indent=2))
    return results


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
    world = World(pools)
    state, contested = converge(pools, world)
    jobs = build_jobs(pools, state, contested, world)
    by_arm = Counter(a for _, _, a, _ in jobs)
    print(f"pools {len(pools)}; contested {len(contested)}; "
          f"jobs {len(jobs)} {dict(by_arm)}; samples/job {M_SAMPLES_SA}")
    if args.dry_run:
        iid, answer, arm, prompt = jobs[0]
        print(f"\n--- {arm} prompt for {iid} candidate {answer} (600 chars) ---")
        print(prompt[:600])
        ao = next(j for j in jobs if j[2] == "answer_only")
        print(f"\n--- answer_only prompt (400 chars) ---\n{ao[3][:400]}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=M_SAMPLES_SA, temperature=TEMPERATURE,
                            top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
                            stop=["[Q]", "[S]"], seed=SEED)
    outputs = llm.generate([p for _, _, _, p in jobs], params)
    scores = {}
    raw = []
    for (iid, answer, arm, _), out in zip(jobs, outputs):
        got = [extract_answer(o.text) for o in out.outputs]
        got = [g for g in got if g is not None]
        s = (sum(match(g, answer) for g in got) / len(got)) if got else 0.0
        scores[(iid, answer, arm)] = s
        raw.append({"iid": iid, "answer": answer, "arm": arm, "S": s,
                    "n_parsed": len(got)})
    with (OUT_DIR / "stage26_scores.jsonl").open("w", encoding="utf-8") as h:
        for r in raw:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    results = analyze(pools, state, contested, scores)
    (OUT_DIR / "stage26_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage26_results.json and stage26_scores.jsonl")


if __name__ == "__main__":
    main()
