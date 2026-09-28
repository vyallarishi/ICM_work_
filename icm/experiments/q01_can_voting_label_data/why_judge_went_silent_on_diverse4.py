"""Stage 11 — why did the judge go silent on diverse-4 views? Autopsy + fix.

Three suspects, one run:
  S1 representation starvation: one arbitrary chain per answer robs the judge
     of redundancy. Test: score window pools under three representations —
     diverse-4 (1 chain/answer), 2+2 (top-two answers x two chains, same
     budget), full pool — with max- and mean-margin aggregation.
  S2 vanished niche: iterated consensus may leave no decision-relevant ties.
     Test: replay the judge-free loop (CPU) on both view types and count
     window pools (top-2 world-share gap small enough for a judge to flip).
  S3 small numbers: report per-window-pool counts so deltas carry their n.

The fix candidates ARE the arms: whichever representation/aggregation restores
top-2 disambiguation accuracy becomes the repaired judge; if only full-pool
does, the fix is adaptive scope (diverse views everywhere, full-pool judging
on window pools only).

Usage: --self-test | --dry-run | full via sbatch.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..q02_can_the_model_judge_itself.verdict_judge_gate import build_prompt, load_pools, match
from .consensus_loop_15lang import (
    initialize,
    update_pool,
    world_context,
    draw_demos,
)
from .consensus_loop_diverse4_views import diverse4_view

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32
WINDOW_GAP = 0.15      # a judge at w<=4 (bonus 4/14=0.29) reaches gaps ~<=0.29;
                       # 0.15 covers light-to-mid schedules; reported either way
REPLAY_LAPS = 4
SEED = 20260817


def make_views(pools):
    """diverse-4, 2+2, and full views per pool (label-blind constructions)."""
    views = {}
    for iid, pool in pools.items():
        chains = [c for c in pool["chains"] if len(c["text"]) <= CHAIN_MAX_CHARS]
        if not chains:
            continue
        diverse = diverse4_view(iid, chains)
        rng = random.Random((1729, iid, "twotwo").__repr__())
        shuffled = list(chains)
        rng.shuffle(shuffled)
        groups = defaultdict(list)
        for c in shuffled:
            groups[c["answer"]].append(c)
        ordered = sorted(groups.values(), key=lambda g: -len(g))
        twotwo = []
        for group in ordered[:2]:
            twotwo.extend(group[:2])
        for group in ordered[2:]:
            if len(twotwo) >= 4:
                break
            twotwo.append(group[0])
        views[iid] = {"diverse": diverse, "twotwo": twotwo[:4], "full": chains}
    return views


def replay_w0(pools, view_key, views):
    """Judge-free iterated consensus on the given views. Returns final state."""
    shadow = {}
    for iid, pool in pools.items():
        if iid not in views:
            continue
        view = views[iid][view_key]
        counts = Counter(c["answer"] for c in view)
        shadow[iid] = dict(pool)
        shadow[iid]["chains"] = view
        shadow[iid]["share"] = counts.most_common(1)[0][1] / len(view)
        shadow[iid]["plurality"] = max(counts, key=lambda a: (counts[a], a))
    state = initialize(shadow)
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for _ in range(REPLAY_LAPS):
        new_state = {i: dict(s) for i, s in state.items()}
        for iid in contested:
            others = world_context(shadow, state, iid)
            new_state[iid]["answer"] = update_pool(shadow[iid], others, None, 0.0)
        state = new_state
    return shadow, state, contested


def window_pools(shadow, state, contested):
    """Pools where a judge could flip the decision: top-2 in-view answers'
    world-shares within WINDOW_GAP, both answers present in the view."""
    window = []
    for iid in contested:
        if state[iid]["answer"] is None:
            continue
        counts = Counter(c["answer"] for c in shadow[iid]["chains"])
        if len(counts) < 2:
            continue
        others = world_context(shadow, state, iid)
        if not others:
            continue
        shares = sorted(((sum(match(o, a) for o in others) / len(others), a)
                         for a in counts), reverse=True)
        gap = shares[0][0] - shares[1][0]
        if gap <= WINDOW_GAP:
            window.append({"iid": iid, "top2": (shares[0][1], shares[1][1]),
                           "gap": round(gap, 3)})
    return window


def top2_metrics(window, shadow, pools, margin_of):
    """For window pools whose top-2 contains gold: does the judge rank the
    gold answer above the rival, per representation x aggregation?"""
    out = {}
    for rep in ("diverse", "twotwo", "full"):
        for agg in ("max", "mean"):
            n = hits = 0
            for w in window:
                gold = pools[w["iid"]]["gold"]
                a_top, a_second = w["top2"]
                if not (match(a_top, gold) or match(a_second, gold)):
                    continue
                scores = {}
                for answer in (a_top, a_second):
                    ms = [margin_of(w["iid"], rep, k)
                          for k, c in enumerate(shadow[w["iid"]]["reps"][rep])
                          if match(c["answer"], answer)
                          and margin_of(w["iid"], rep, k) is not None]
                    if not ms:
                        scores[answer] = None
                        continue
                    scores[answer] = max(ms) if agg == "max" else sum(ms) / len(ms)
                if scores.get(a_top) is None or scores.get(a_second) is None:
                    continue
                n += 1
                gold_answer = a_top if match(a_top, gold) else a_second
                rival = a_second if match(a_top, gold) else a_top
                hits += scores[gold_answer] > scores[rival]
            out[f"{rep}_{agg}"] = {"n": n, "acc": round(hits / max(n, 1), 4)}
    return out


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- make_views ---")
    pool = {"chains": [{"answer": "10", "text": f"t{i}"} for i in range(9)]
            + [{"answer": "20", "text": f"u{i}"} for i in range(5)]
            + [{"answer": "30", "text": "v0"}, {"answer": "40", "text": "w0"}]}
    views = make_views({"train_en_1": dict(pool, gold="10", lang="en",
                                           question="q", share=0.5,
                                           plurality="10")})
    v = views["train_en_1"]
    check("diverse covers 4 answers",
          len({c["answer"] for c in v["diverse"]}), 4)
    two_answers = Counter(c["answer"] for c in v["twotwo"])
    check("2+2 covers exactly top-2", sorted(two_answers), ["10", "20"])
    check("2+2 has two chains each", sorted(two_answers.values()), [2, 2])
    check("full keeps all", len(v["full"]), 16)

    print("--- window detection ---")
    shadow = {
        "train_en_1": {"lang": "en", "gold": "10", "question": "q",
                       "chains": [{"answer": "10", "text": "a"},
                                  {"answer": "20", "text": "b"}],
                       "share": 0.5, "plurality": "10"},
        "train_de_1": {"lang": "de", "gold": "10", "question": "q",
                       "chains": [{"answer": "10", "text": "c"}] * 4,
                       "share": 1.0, "plurality": "10"},
        "train_hi_1": {"lang": "hi", "gold": "10", "question": "q",
                       "chains": [{"answer": "20", "text": "d"}] * 4,
                       "share": 1.0, "plurality": "20"},
    }
    state = {"train_en_1": {"answer": "10", "kind": "contested"},
             "train_de_1": {"answer": "10", "kind": "committed"},
             "train_hi_1": {"answer": "20", "kind": "committed"}}
    window = window_pools(shadow, state, ["train_en_1"])
    check("near-tie pool is in window", len(window), 1)
    check("top2 recorded", sorted(window[0]["top2"]), ["10", "20"])
    state2 = dict(state)
    shadow2 = dict(shadow)
    shadow2["train_hi_1"] = dict(shadow["train_hi_1"],
                                 chains=[{"answer": "10", "text": "d"}] * 4,
                                 plurality="10")
    state2["train_hi_1"] = {"answer": "10", "kind": "committed"}
    window2 = window_pools(shadow2, state2, ["train_en_1"])
    check("decisive world -> no window", len(window2), 0)

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
    views = make_views(pools)

    print("=== S2: decision windows after judge-free convergence ===")
    windows = {}
    shadows = {}
    for key in ("diverse", "full"):
        shadow, state, contested = replay_w0(pools, key, views)
        window = window_pools(shadow, state, contested)
        windows[key], shadows[key] = window, shadow
        gold_in = sum(1 for w in window
                      if any(match(a, pools[w["iid"]]["gold"]) for a in w["top2"]))
        print(f"{key:>8}: contested {len(contested)}, window {len(window)}, "
              f"window-with-gold-in-top2 {gold_in}")

    # attach representations for the scored (diverse-replay) window
    window = windows["diverse"]
    shadow = shadows["diverse"]
    for w in window:
        shadow[w["iid"]]["reps"] = views[w["iid"]]
    total = sum(len(views[w["iid"]][r]) for w in window
                for r in ("diverse", "twotwo", "full"))
    print(f"chains to score across representations: {total} x 2 passes")

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

    # demos from the converged judge-free state (realistic in-loop context)
    _, conv_state, _ = replay_w0(pools, "diverse", views)
    rng = random.Random(SEED)
    demos = draw_demos({i: shadows["diverse"][i] for i in shadows["diverse"]},
                       conv_state, rng)

    margins = {}
    for role_swap in (False, True):
        positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
        entries = []
        seen = set()
        for w in window:
            pool = pools[w["iid"]]
            for rep in ("diverse", "twotwo", "full"):
                for k, chain in enumerate(views[w["iid"]][rep]):
                    key = (w["iid"], id(chain))
                    if key in seen:
                        continue
                    seen.add(key)
                    prompt = build_prompt(pool["question"], chain["text"],
                                          positive, negative,
                                          demos.get(pool["lang"], []))
                    n_tokens = len(tokenizer.encode(prompt))
                    if n_tokens > MAX_PROMPT_TOKENS:
                        continue
                    entries.append((n_tokens, key, prompt))
        entries.sort(key=lambda e: e[0])
        print(f"[pass {'swap' if role_swap else 'norm'}] {len(entries)} chains",
              flush=True)
        for start in range(0, len(entries), BATCH_SIZE):
            batch = entries[start:start + BATCH_SIZE]
            enc = tokenizer([p for _, _, p in batch], return_tensors="pt",
                            padding=True, pad_to_multiple_of=256).to(model.device)
            with torch.inference_mode():
                logits = model(**enc, use_cache=False, **last_logit).logits
                log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            for row, (_, key, _) in enumerate(batch):
                margin = float(log_probs[row, marker_ids[positive]]
                               - log_probs[row, marker_ids[negative]])
                margins.setdefault(key, {})[
                    "swap" if role_swap else "norm"] = margin

    def margin_of(iid, rep, k):
        chain = views[iid][rep][k]
        both = margins.get((iid, id(chain)), {})
        if "norm" in both and "swap" in both:
            return (both["norm"] + both["swap"]) / 2
        return None

    print()
    print("=== S1: top-2 disambiguation accuracy by representation ===")
    results = top2_metrics(window, shadow, pools, margin_of)
    for name, block in results.items():
        print(f"{name:<16} n={block['n']:>4}  acc={100*block['acc']:.1f}%")
    results["windows"] = {k: len(v) for k, v in windows.items()}
    (OUT_DIR / "stage11_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage11_results.json")


if __name__ == "__main__":
    main()
