"""Stage 27 — TRUNCATED-PEER SELECTIVITY: can the model complete a correct
derivation better than a broken one?

WHY (stage26, negative): conditioning on 3 same-answer peers gave
P1 selectivity 52.2% (chance) and P2 AUC 0.514; peers from an UNRELATED
problem scored the same (0.520). Full peers hand over an endpoint
('#### N'), so the model copies rather than derives, and nothing about
correctness is tested.

THIS EXPERIMENT: strip each peer's FINAL step and answer line, leaving
steps 1..n-1. The model must finish the derivation itself. Hypothesis: a
correct partial derivation can be completed to its endpoint (the
arithmetic cooperates); a broken one cannot be completed to ITS endpoint
as reliably. If so, re-derivation rate becomes evidence about correctness
-- selectivity -- which is what stage26 failed to find.

ARMS (all peers same-cluster, 3 peers, m=8 samples):
  full        peers intact (stage26 replication anchor)
  truncated   peers' last reasoning line + '#### N' removed
  trunc_wrong truncated peers drawn from a WRONG cluster on pools whose
              consensus answer is CORRECT (the adversarial arm: if the
              model completes broken derivations just as readily, the
              mechanism is hint-following, full stop)

LEAK CONTROL: peer answers leak through prose (e.g. '<<400/10=40>>40').
Each prompt records whether the candidate answer's value still appears in
the truncated peer text; all metrics are reported overall AND leak-free.

PRE-REGISTERED (gold only in scoring, never in prompts):
  P1 selectivity: on pools with both a correct and a wrong candidate,
     S(correct) > S(wrong) in >60% of pools (stage26 full: 52.2%).
  P2 calibration: AUC of S(argmax) separating correct from wrong commits
     > 0.70 (stage26 full: 0.514).
  P3 adversarial: S(wrong cluster | truncated) must be materially LOWER
     than S(correct cluster | truncated); a gap < 0.10 means the model
     completes broken derivations as readily as sound ones -> hint pipe
     confirmed, three independent formulations agreeing.
Failure on P1/P2/P3 closes the verification hypothesis.

Usage: --self-test | --dry-run | full via stage27_trunc.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .verdict_judge_gate import auc, load_pools, match, norm
from ..q01_can_voting_label_data.consensus_loop_15lang import initialize, update_pool
from .continuation_consistency_judge import extract_answer
from .ensemble_judge_in_loop import static_pos_demos
from ..q03_does_regeneration_help.regeneration_15lang import (MAX_MODEL_LEN, MAX_NEW_TOKENS, PEER_MAX_CHARS,
                           TEMPERATURE, TOP_P)
from ..q04_full_scale_headline.full_scale_37x1500 import World
from .generative_mutual_predictability import candidates_for, peers_for_answer

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
M_SAMPLES = 8
N_PEERS = 3
ITERATIONS = 4
SEED = 20260823
ARMS = ("full", "truncated", "trunc_wrong")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def truncate_peer(text):
    """Drop '#### N' lines and the final reasoning line: the model must
    complete the derivation instead of copying its endpoint."""
    lines = [l for l in text.split("\n")
             if l.strip() and not l.lstrip().startswith("####")]
    if len(lines) < 2:
        return None
    return "\n".join(lines[:-1])


def leaks(text, answer):
    """Does the candidate answer's value still appear in the peer text?"""
    if answer is None:
        return False
    return any(match(norm(n), answer) for n in NUMBER_RE.findall(text))


def build_prompt(question, peers, demos):
    blocks = []
    for demo, label in demos:
        if label:
            blocks.append("[Q]\n" + demo["question"].strip() + "\n[A]\n"
                          + demo["chain_text"].strip()[:PEER_MAX_CHARS])
    for peer in peers:
        blocks.append("[S]\n" + peer.strip()[:PEER_MAX_CHARS])
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n")
    return "\n\n".join(blocks)


def converge(pools, world):
    state = initialize(pools)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(ITERATIONS):
        ns = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            ns[iid]["answer"] = update_pool(pools[iid],
                                            world.context(state, iid), None, 0.0)
        state = ns
    return state, contested


def peers_supporting(pools, iid, answer, rng, n=N_PEERS):
    """Chains from OTHER languages (same problem) whose CHAIN answer matches
    `answer` -- looser than peers_for_answer, which requires that language's
    committed answer to match. Needed for the adversarial arm, where wrong
    answers rarely win any language's commit."""
    problem = iid.rsplit("_", 1)[-1]
    lang = pools[iid]["lang"]
    hits = []
    for other, pool in pools.items():
        if other.rsplit("_", 1)[-1] != problem or pool["lang"] == lang:
            continue
        chains = [c for c in pool["chains"] if match(c["answer"], answer)
                  and len(c["text"]) <= PEER_MAX_CHARS]
        if chains:
            hits.append(min(chains, key=lambda c: len(c["text"]))["text"])
    rng.shuffle(hits)
    return hits[:n]


def wrong_candidate(pools, state, world, iid):
    """A candidate answer this pool generated that is NOT the consensus
    answer (label-blind: 'not what the world/loop settled on')."""
    settled = state[iid]["answer"]
    counts = Counter(c["answer"] for c in pools[iid]["chains"])
    alts = [a for a in counts if settled is None or not match(a, settled)]
    return max(alts, key=lambda a: (counts[a], a)) if alts else None


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- truncate_peer ---")
    chain = "step one\nstep two\nstep three\n#### 42"
    check("drops answer line and last step", truncate_peer(chain),
          "step one\nstep two")
    check("two body lines keep one",
          truncate_peer("step one\nstep two\n#### 7"), "step one")
    check("one body line -> None (nothing left to complete)",
          truncate_peer("only step\n#### 7"), None)
    check("single line -> None", truncate_peer("just this"), None)
    check("no answer marker still truncates",
          truncate_peer("a\nb\nc"), "a\nb")

    print("--- leak detection ---")
    check("leak via inline calc", leaks("400 / 10 = <<400/10=40>>40 tubes", "40"),
          True)
    check("no leak", leaks("400 / 10 = ...", "40"), False)
    check("intermediate value not the answer", leaks("6 * 2 = 12", "36"), False)

    print("--- prompt ---")
    demos = [({"question": "DQ", "chain_text": "DC\n#### 1"}, True)]
    p = build_prompt("Q?", ["P1 step", "P2 step"], demos)
    check("peers present", p.count("[S]"), 2)
    check("ends open", p.endswith("[Q]\nQ?\n[A]\n"), True)
    check("no answer markers from peers", "#### " in p.split("[S]")[1], False)

    print("--- wrong_candidate is label-blind ---")
    pools = {"train_am_1": {"lang": "am", "gold": "5", "question": "q",
                            "share": 0.75, "plurality": "40",
                            "chains": [{"answer": "40", "text": "a\nb\n#### 40"},
                                       {"answer": "40", "text": "c\nd\n#### 40"},
                                       {"answer": "5", "text": "e\nf\n#### 5"}]}}
    world = World(pools)
    state = {"train_am_1": {"answer": "5", "kind": "contested"}}
    check("picks the non-settled answer",
          wrong_candidate(pools, state, world, "train_am_1"), "40")

    print("--- S(a) arithmetic ---")
    got = [extract_answer(s) for s in ["x\n#### 5", "y\n#### 5", "z\n#### 9"]]
    check("S = 2/3", round(sum(match(g, "5") for g in got) / len(got), 3), 0.667)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def build_jobs(pools, state, contested, world):
    demos = static_pos_demos(pools)
    rng = random.Random(SEED)
    jobs = []
    for iid in contested:
        q = pools[iid]["question"]
        d = demos.get(pools[iid]["lang"], [])
        for answer in candidates_for(pools, state, world, iid):
            peers = peers_for_answer(pools, state, iid, answer, rng)
            if not peers:
                continue
            jobs.append((iid, answer, "full", build_prompt(q, peers, d),
                         any(leaks(p, answer) for p in peers)))
            trunc = [t for t in (truncate_peer(p) for p in peers) if t]
            if trunc:
                jobs.append((iid, answer, "truncated",
                             build_prompt(q, trunc, d),
                             any(leaks(t, answer) for t in trunc)))
        # adversarial arm: truncated peers supporting a NON-settled answer
        alt = wrong_candidate(pools, state, world, iid)
        if alt is not None:
            alt_peers = peers_supporting(pools, iid, alt, rng)
            trunc_alt = [t for t in (truncate_peer(p) for p in alt_peers) if t]
            if trunc_alt:
                jobs.append((iid, alt, "trunc_wrong",
                             build_prompt(q, trunc_alt, d),
                             any(leaks(t, alt) for t in trunc_alt)))
    return jobs


def analyze(pools, state, scores):
    """scores[(iid, answer, arm)] = (S, leaked)."""
    results = {}
    for arm in ARMS:
        per_pool = defaultdict(dict)
        leak_flag = {}
        for (iid, answer, a), (s, leaked) in scores.items():
            if a == arm:
                per_pool[iid][answer] = s
                leak_flag[(iid, answer)] = leaked
        block = {"n_pools": len(per_pool)}
        for subset, keep in (("all", lambda i, a: True),
                             ("leak_free", lambda i, a: not leak_flag[(i, a)])):
            wins = total = 0
            for iid, per_answer in per_pool.items():
                gold = pools[iid]["gold"]
                right = [a for a in per_answer if match(a, gold) and keep(iid, a)]
                wrong = [a for a in per_answer
                         if not match(a, gold) and keep(iid, a)]
                if not right or not wrong:
                    continue
                total += 1
                if max(per_answer[a] for a in right) > max(per_answer[a]
                                                           for a in wrong):
                    wins += 1
            pos, neg = [], []
            for iid, per_answer in per_pool.items():
                cands = [a for a in per_answer if keep(iid, a)]
                if not cands:
                    continue
                best = max(cands, key=lambda a: (per_answer[a], a))
                (pos if match(best, pools[iid]["gold"]) else neg).append(
                    per_answer[best])
            mean_s = ([s for (i, a), s in
                       (((i, a), per_pool[i][a]) for i in per_pool
                        for a in per_pool[i]) if keep(i, a)])
            block[subset] = {
                "P1_pools": total, "P1_wins": wins,
                "P1_win_rate": round(wins / max(total, 1), 4),
                "P2_auc": round(auc(pos, neg), 4) if pos and neg else None,
                "P2_n_correct": len(pos), "P2_n_wrong": len(neg),
                "mean_S": round(sum(mean_s) / max(len(mean_s), 1), 4),
            }
        results[arm] = block
        print(f"\n=== ARM {arm} ===")
        print(json.dumps(block, indent=2))
    # P3: adversarial gap
    t = results.get("truncated", {}).get("all", {}).get("mean_S")
    w = results.get("trunc_wrong", {}).get("all", {}).get("mean_S")
    if t is not None and w is not None:
        results["P3_gap"] = {"mean_S_truncated_settled": t,
                             "mean_S_truncated_wrong": w,
                             "gap": round(t - w, 4),
                             "verdict": ("SELECTIVE" if t - w >= 0.10
                                         else "NOT SELECTIVE (hint pipe)")}
        print(f"\nP3: S(settled|trunc)={t}  S(wrong|trunc)={w}  "
              f"gap={t-w:+.4f} -> {results['P3_gap']['verdict']}")
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
    by_arm = Counter(a for _, _, a, _, _ in jobs)
    leaked = sum(1 for j in jobs if j[4] and j[2] == "truncated")
    n_trunc = by_arm.get("truncated", 0)
    print(f"pools {len(pools)}; contested {len(contested)}; jobs {len(jobs)} "
          f"{dict(by_arm)}; truncated-arm leak rate "
          f"{100*leaked/max(n_trunc,1):.1f}%; samples/job {M_SAMPLES}")
    if args.dry_run:
        for arm in ARMS:
            j = next((x for x in jobs if x[2] == arm), None)
            if j:
                print(f"\n--- {arm} prompt for {j[0]} cand {j[1]} "
                      f"(leaked={j[4]}, last 400 chars) ---")
                print(j[3][-400:])
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=M_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]", "[S]"],
                            seed=SEED)
    outputs = llm.generate([p for _, _, _, p, _ in jobs], params)
    scores, raw = {}, []
    for (iid, answer, arm, _, leaked), out in zip(jobs, outputs):
        got = [extract_answer(o.text) for o in out.outputs]
        got = [g for g in got if g is not None]
        s = (sum(match(g, answer) for g in got) / len(got)) if got else 0.0
        scores[(iid, answer, arm)] = (s, leaked)
        raw.append({"iid": iid, "answer": answer, "arm": arm, "S": s,
                    "leaked": leaked, "n_parsed": len(got)})
    with (OUT_DIR / "stage27_scores.jsonl").open("w", encoding="utf-8") as h:
        for r in raw:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    results = analyze(pools, state, scores)
    (OUT_DIR / "stage27_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage27_results.json and stage27_scores.jsonl")


if __name__ == "__main__":
    main()
