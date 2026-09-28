"""Stage 28 — POWERED SELECTIVITY TEST: truncated-peer re-derivation at
37 languages x 1500 problems, with harder truncation.

STATUS ENTERING THIS RUN (stage27, 15 langs, 460 contested pools):
  P3 gap  S(settled|trunc) 0.445 vs S(wrong|trunc) 0.314 = +0.131 SELECTIVE
  P1      57.1% all / 62.5% leak-free -- but leak-free n = 16 pools (weak)
  P2      AUC 0.594 (bar 0.70, missed)
  full-peer control: P1 39.1%, mean_S 0.647 (endpoint copying)
This run powers P1 up ~25x (11,907 contested pools) and attacks the 26.6%
leak rate that forced the leak-free subset to be tiny.

HARDER TRUNCATION (arm `trunc2`): drop the last TWO reasoning lines and
strip <<a*b=c>> calculator annotations, the dominant leak vector. Arm
`truncated` keeps stage27's rule as the replication anchor.

ARMS: full | truncated | trunc2 | trunc_wrong | trunc2_wrong
PRE-REGISTERED (unchanged bars; gold only in scoring):
  P1 > 0.60 leak-free at n >= 200 pools   (stage27: 0.625 at n=16)
  P2 AUC > 0.70                            (stage27: 0.594)
  P3 gap >= 0.10 for both truncation rules (stage27: +0.131)
  P4 (new) trunc2 leak rate < 15% and its P3 gap >= truncated's -- i.e.
     removing the leak vector does not destroy the signal, which would
     indicate the stage27 gap was leak-driven.
FAILURE of P1 at power closes the verification hypothesis for good.

Usage: --self-test | --dry-run | full via stage28_trunc_scale.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..q01_can_voting_label_data import consensus_loop_37lang as s17           # patches gate.TIER to 37 langs
from .verdict_judge_gate import auc, match
from .continuation_consistency_judge import extract_answer
from .ensemble_judge_in_loop import static_pos_demos
from ..q03_does_regeneration_help.regeneration_15lang import (MAX_MODEL_LEN, MAX_NEW_TOKENS, PEER_MAX_CHARS,
                           TEMPERATURE, TOP_P)
from ..q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from .generative_mutual_predictability import candidates_for, peers_for_answer
from .truncated_peer_rederivation import (build_prompt, leaks, peers_supporting,
                               truncate_peer, wrong_candidate)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = ("/scratch/rvyalla/multilingual-icm/results/"
                "qwen3-14b-full37-1500")
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
M_SAMPLES = 8
N_PEERS = 3
SEED = 20260823
CALC_RE = re.compile(r"<<[^>]*>>")
ARMS = ("full", "truncated", "trunc2", "trunc_wrong", "trunc2_wrong")
TRUNC_RULES = {"truncated": 1, "trunc2": 2}


def truncate_peer2(text):
    """Harder truncation: strip <<a*b=c>> annotations (main leak vector),
    then drop the last TWO reasoning lines."""
    stripped = CALC_RE.sub("", text)
    lines = [l for l in stripped.split("\n")
             if l.strip() and not l.lstrip().startswith("####")]
    if len(lines) < 3:
        return None
    return "\n".join(lines[:-2])


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- truncate_peer2 ---")
    chain = ("a = 6 * 2 = <<6*2=12>>12\nb = 12 * 3 = <<12*3=36>>36\n"
             "so the answer is 36\n#### 36")
    check("drops last two lines and calc annotations",
          truncate_peer2(chain), "a = 6 * 2 = 12")
    check("too short -> None", truncate_peer2("a\nb\n#### 3"), None)
    check("no calc annotations survive",
          "<<" in (truncate_peer2("x=<<1+1=2>>2\ny\nz\nw\n#### 9") or ""), False)

    print("--- truncation reduces leakage vs stage27 rule ---")
    peer = ("Natalia sold 48 / 2 = <<48/2=24>>24 clips in May.\n"
            "In total she sold 48 + 24 = <<48+24=72>>72 clips.\n"
            "So the answer is 72.\n#### 72")
    t1, t2 = truncate_peer(peer), truncate_peer2(peer)
    check("stage27 rule still leaks 72 here", leaks(t1, "72"), True)
    check("stage28 rule removes the leak", leaks(t2, "72"), False)

    print("--- 37-language wiring ---")
    from . import verdict_judge_gate as gate
    check("TIER patched to 37", len(gate.TIER), 37)

    print("--- prompt ---")
    demos = [({"question": "DQ", "chain_text": "DC\n#### 1"}, True)]
    p = build_prompt("Q?", ["P1 step"], demos)
    check("ends open", p.endswith("[Q]\nQ?\n[A]\n"), True)

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
            for arm, rule in (("truncated", truncate_peer),
                              ("trunc2", truncate_peer2)):
                cut = [t for t in (rule(p) for p in peers) if t]
                if cut:
                    jobs.append((iid, answer, arm, build_prompt(q, cut, d),
                                 any(leaks(t, answer) for t in cut)))
        alt = wrong_candidate(pools, state, world, iid)
        if alt is not None:
            alt_peers = peers_supporting(pools, iid, alt, rng, n=N_PEERS)
            for arm, rule in (("trunc_wrong", truncate_peer),
                              ("trunc2_wrong", truncate_peer2)):
                cut = [t for t in (rule(p) for p in alt_peers) if t]
                if cut:
                    jobs.append((iid, alt, arm, build_prompt(q, cut, d),
                                 any(leaks(t, alt) for t in cut)))
    return jobs


def analyze(pools, scores):
    results = {}
    for arm in ARMS:
        per_pool = defaultdict(dict)
        leak_flag = {}
        for (iid, answer, a), (s, leaked) in scores.items():
            if a == arm:
                per_pool[iid][answer] = s
                leak_flag[(iid, answer)] = leaked
        block = {"n_pools": len(per_pool),
                 "leak_rate": round(
                     sum(leak_flag.values()) / max(len(leak_flag), 1), 4)}
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
                if max(per_answer[a] for a in right) > max(
                        per_answer[a] for a in wrong):
                    wins += 1
            pos, neg, allvals = [], [], []
            for iid, per_answer in per_pool.items():
                cands = [a for a in per_answer if keep(iid, a)]
                if not cands:
                    continue
                allvals.extend(per_answer[a] for a in cands)
                best = max(cands, key=lambda a: (per_answer[a], a))
                (pos if match(best, pools[iid]["gold"]) else neg).append(
                    per_answer[best])
            block[subset] = {
                "P1_pools": total, "P1_wins": wins,
                "P1_win_rate": round(wins / max(total, 1), 4),
                "P2_auc": round(auc(pos, neg), 4) if pos and neg else None,
                "P2_n_correct": len(pos), "P2_n_wrong": len(neg),
                "mean_S": round(sum(allvals) / max(len(allvals), 1), 4),
            }
        results[arm] = block
        print(f"\n=== ARM {arm} (leak rate {block['leak_rate']:.1%}) ===")
        print(json.dumps(block, indent=2))
    for settled, wrong, tag in (("truncated", "trunc_wrong", "P3_rule1"),
                                ("trunc2", "trunc2_wrong", "P3_rule2")):
        for subset in ("all", "leak_free"):
            t = results.get(settled, {}).get(subset, {}).get("mean_S")
            w = results.get(wrong, {}).get(subset, {}).get("mean_S")
            if t is None or w is None:
                continue
            results[f"{tag}_{subset}"] = {
                "S_settled": t, "S_wrong": w, "gap": round(t - w, 4),
                "verdict": ("SELECTIVE" if t - w >= 0.10
                            else "NOT SELECTIVE (hint pipe)")}
            print(f"{tag} [{subset}]: {t} vs {w} -> gap {t-w:+.4f} "
                  f"{results[f'{tag}_{subset}']['verdict']}")
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

    pools = load(args.root)
    world = World(pools)
    state, contested, _ = converge(pools, world)
    jobs = build_jobs(pools, state, contested, world)
    by_arm = Counter(a for _, _, a, _, _ in jobs)
    print(f"pools {len(pools)}; contested {len(contested)}; jobs {len(jobs)}")
    for arm in ARMS:
        n = by_arm.get(arm, 0)
        lk = sum(1 for j in jobs if j[2] == arm and j[4])
        print(f"  {arm:<14} {n:>6} jobs, leak rate {100*lk/max(n,1):.1f}%")
    if args.dry_run:
        for arm in ("truncated", "trunc2"):
            j = next((x for x in jobs if x[2] == arm and not x[4]), None)
            if j:
                print(f"\n--- {arm} (leak-free) last 300 chars ---\n{j[3][-300:]}")
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
    with (OUT_DIR / "stage28_scores.jsonl").open("w", encoding="utf-8") as h:
        for r in raw:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    results = analyze(pools, scores)
    (OUT_DIR / "stage28_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage28_results.json and stage28_scores.jsonl")


if __name__ == "__main__":
    main()
