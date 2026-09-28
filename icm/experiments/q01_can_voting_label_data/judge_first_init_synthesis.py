"""Stage 15 — THE SYNTHESIS RUN (user-specified):
diverse-8 views, 15 languages, MP-FIRST initialization, consensus-dominant
early weights, trust annealing toward the judge across rounds.

Design:
  views     diverse-8 (answer coverage + frequency texture, ~8 chains/pool)
  init      landslide pools (vote share >= 0.7 in-view) freeze at their vote
            answer (voting strong at the start, per spec). CONTESTED pools
            initialize from the JUDGE'S proposal: J0 scores their chains,
            initial answer = argmax over answers of mean debiased margin.
  loop      6 laps; per lap: demos re-drawn from current labels -> J3
            re-scores contested chains -> update:
              score(a) = w_vote(k) * world_share(a)
                       + (w_judge(k)/|jury|) * [judge pick == a]
                       + 0.001 * local_count(a)
            with veto (strong world answer absent from pool -> NULL).
  schedule  w_vote  = 1.0, 0.95, 0.90, 0.85, 0.80, 0.80   (voting eases off)
            w_judge = 0.5, 1.0,  1.5,  2.0,  2.0,  2.0    (trust anneals up,
            capped at 2 — stage10 measured w=4 to dip)
Readouts: judge-proposal accuracy before any enforcement (the MP-first
quality), per-lap trajectory, final system numbers. Vote-first comparison =
stage13 (same views, vote init) offline.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..q02_can_the_model_judge_itself.verdict_judge_gate import build_prompt, load_pools, match
from .consensus_loop_15lang import draw_demos, initialize, world_context
from .judge_weight_annealed_diverse8 import diverse4_view as diverse8_view  # size-8 inside

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANDSLIDE = 0.70
WORLD_STRENGTH = 0.70
W_VOTE = [1.0, 0.95, 0.90, 0.85, 0.80, 0.80]
W_JUDGE = [0.5, 1.0, 1.5, 2.0, 2.0, 2.0]
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32
SEED = 20260818


def apply_views(pools):
    for iid, pool in pools.items():
        view = diverse8_view(iid, [c for c in pool["chains"]
                                   if len(c["text"]) <= CHAIN_MAX_CHARS])
        if not view:
            view = pool["chains"][:1]
        pool["chains"] = view
        counts = Counter(c["answer"] for c in view)
        pool["share"] = counts.most_common(1)[0][1] / len(view)
        pool["plurality"] = max(counts, key=lambda a: (counts[a], a))
    return pools


def scheduled_update(pool, others, judge_pick, w_vote, w_judge):
    counts = Counter(c["answer"] for c in pool["chains"])
    if not counts:
        return None
    if others:
        top_world, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) >= WORLD_STRENGTH and top_world not in counts:
            return None
    def score(a):
        ws = (sum(match(o, a) for o in others) / len(others)) if others else 0.0
        bonus = (w_judge / max(len(others), 1)) if (
            judge_pick is not None and match(a, judge_pick)) else 0.0
        return w_vote * ws + bonus + 0.001 * counts[a]
    return max(counts, key=lambda a: (score(a), a))


def proposal_from_margins(pool, margin_of_chain):
    """MP-first initial answer: argmax over answers of MEAN debiased margin."""
    by_answer = defaultdict(list)
    for k, chain in enumerate(pool["chains"]):
        margin = margin_of_chain(k)
        if margin is not None:
            by_answer[chain["answer"]].append(margin)
    if not by_answer:
        return pool["plurality"]
    return max(by_answer,
               key=lambda a: (sum(by_answer[a]) / len(by_answer[a]), a))


def metrics(pools, state, contested_ids):
    answered = [i for i in contested_ids if state[i]["answer"] is not None]
    hits = sum(match(state[i]["answer"], pools[i]["gold"]) for i in answered)
    committed = [i for i, s in state.items() if s["answer"] is not None]
    total_hits = sum(match(state[i]["answer"], pools[i]["gold"])
                     for i in committed)
    return {"contested_answered": len(answered),
            "contested_accuracy": round(hits / max(len(answered), 1), 4),
            "coverage": round(len(committed) / len(state), 4),
            "precision": round(total_hits / max(len(committed), 1), 4)}


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- scheduled_update ---")
    pool = {"chains": [{"answer": "10"}, {"answer": "10"},
                       {"answer": "20"}, {"answer": "30"}]}
    tied = ["10"] * 7 + ["20"] * 7
    check("early lap: light judge breaks ties only",
          scheduled_update(pool, tied, "20", 1.0, 0.5), "20")
    check("early lap: judge cannot override world",
          scheduled_update(pool, ["10"] * 9 + ["20"] * 5, "20", 1.0, 0.5), "10")
    late_gap = ["10"] * 8 + ["20"] * 6   # gap 2/14 ~ 0.143
    check("late lap: annealed judge (2.0) closes small gaps",
          scheduled_update(pool, late_gap, "20", 0.8, 2.0), "20")
    check("late lap: veto still absolute",
          scheduled_update(pool, ["99"] * 10, "99", 0.8, 2.0), None)
    check("vote-easing alone never flips without judge",
          scheduled_update(pool, ["10"] * 9 + ["20"] * 5, None, 0.8, 2.0), "10")

    print("--- proposal_from_margins ---")
    pool2 = {"plurality": "20",
             "chains": [{"answer": "20"}, {"answer": "20"},
                        {"answer": "10"}, {"answer": "10"}]}
    margins = {0: -0.5, 1: -0.7, 2: 0.9, 3: 0.4}
    check("judge proposal overrides plurality",
          proposal_from_margins(pool2, lambda k: margins.get(k)), "10")
    check("no margins -> falls back to plurality",
          proposal_from_margins(pool2, lambda k: None), "20")
    check("mean not max decides",
          proposal_from_margins(pool2, lambda k: {0: 2.0, 1: -1.9,
                                                  2: 0.3, 3: 0.2}[k]), "10")

    print("--- schedule sanity ---")
    check("six laps of weights", (len(W_VOTE), len(W_JUDGE)), (6, 6))
    check("judge anneals up, capped at 2",
          (W_JUDGE[0] < W_JUDGE[-1], max(W_JUDGE)), (True, 2.0))
    check("vote eases down", W_VOTE[0] > W_VOTE[-1], True)

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

    pools = apply_views(load_pools(args.root))
    vote_state = initialize(pools)
    contested_ids = [i for i, s in vote_state.items()
                     if s["kind"] == "contested"]
    chains_per_lap = sum(len(pools[i]["chains"]) for i in contested_ids)
    print(f"pools {len(pools)}; contested {len(contested_ids)}; "
          f"chains/lap {chains_per_lap}")
    print(f"vote-static baseline: {metrics(pools, vote_state, contested_ids)}")
    if args.dry_run:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    marker_ids = {m: tokenizer.encode(m, add_special_tokens=False)[0]
                  for m in (" Yes", " No")}
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    def score(chain_jobs, demos_by_lang):
        margins = {}
        for role_swap in (False, True):
            positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
            entries = []
            for iid, k in chain_jobs:
                pool = pools[iid]
                chain = pool["chains"][k]
                prompt = build_prompt(pool["question"], chain["text"],
                                      positive, negative,
                                      demos_by_lang.get(pool["lang"], []))
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > MAX_PROMPT_TOKENS:
                    continue
                entries.append((n_tokens, iid, k, prompt))
            entries.sort(key=lambda e: e[0])
            for start in range(0, len(entries), BATCH_SIZE):
                batch = entries[start:start + BATCH_SIZE]
                enc = tokenizer([p for *_, p in batch], return_tensors="pt",
                                padding=True,
                                pad_to_multiple_of=256).to(model.device)
                with torch.inference_mode():
                    logits = model(**enc, use_cache=False, **last_logit).logits
                    lp = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                for row, (_, iid, k, _) in enumerate(batch):
                    m = float(lp[row, marker_ids[positive]]
                              - lp[row, marker_ids[negative]])
                    margins.setdefault((iid, k), {})[
                        "swap" if role_swap else "norm"] = m
        out = {}
        for key, both in margins.items():
            if "norm" in both and "swap" in both:
                out[key] = (both["norm"] + both["swap"]) / 2
        return out

    # ---- MP-FIRST INITIALIZATION (J0: no demos) -------------------------
    jobs = [(iid, k) for iid in contested_ids
            for k in range(len(pools[iid]["chains"]))]
    print(f"[init/J0] scoring {len(jobs)} chains", flush=True)
    init_margins = score(jobs, {})
    state = {i: dict(s) for i, s in vote_state.items()}
    divergent = 0
    for iid in contested_ids:
        proposal = proposal_from_margins(
            pools[iid], lambda k, _iid=iid: init_margins.get((_iid, k)))
        if not match(proposal, vote_state[iid]["answer"]):
            divergent += 1
        state[iid]["answer"] = proposal
    row = metrics(pools, state, contested_ids)
    print(f"[init] MP-first proposals: {row} "
          f"(diverge from vote on {divergent} pools)", flush=True)
    trajectory = [dict(row, lap=0, divergent_from_vote=divergent)]

    # ---- THE ANNEALED LOOP ---------------------------------------------
    for lap in range(1, 7):
        w_vote, w_judge = W_VOTE[lap - 1], W_JUDGE[lap - 1]
        rng = random.Random((SEED, lap).__repr__())
        demos = draw_demos(pools, state, rng)
        lap_margins = score(jobs, demos)
        picks = {}
        for iid in contested_ids:
            best, best_m = None, None
            for k, chain in enumerate(pools[iid]["chains"]):
                m = lap_margins.get((iid, k))
                if m is not None and (best_m is None or m > best_m):
                    best, best_m = chain["answer"], m
            picks[iid] = best
        changes = 0
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested_ids:
            others = world_context(pools, state, iid)
            new = scheduled_update(pools[iid], others, picks.get(iid),
                                   w_vote, w_judge)
            if new != state[iid]["answer"]:
                changes += 1
            new_state[iid]["answer"] = new
        state = new_state
        row = dict(metrics(pools, state, contested_ids), lap=lap,
                   changes=changes, w_vote=w_vote, w_judge=w_judge)
        trajectory.append(row)
        print(f"[lap {lap}] {row}", flush=True)

    (OUT_DIR / "stage15_results.json").write_text(
        json.dumps(trajectory, indent=2))
    print("wrote stage15_results.json")


if __name__ == "__main__":
    main()
