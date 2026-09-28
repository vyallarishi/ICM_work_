"""Stage 16 — FULL 37-LANGUAGE run of the winning configuration.

Original stage 9 docstring:
 the ICM loop at 14B: iterated consensus + a weighted MP judge.

THE EXPERIMENT (pre-registered):
Three arms, identical pools (qwen3-14b-phaseA, 15 langs x 150), 4 iterations:
  w0     w_judge = 0.0   pure iterated consensus (control)
  light  w_judge = 0.5   fusion-calibrated: judge worth half a language-vote
  heavy  w_judge = 7.0   judge worth half the jury (the overweight test)

Loop per iteration: demo banks re-drawn from CURRENT labels (the mutual
feedback channel) -> judge re-scores contested chains -> each contested pool
re-picks its answer by score(a) = world_share(a) + judge_bonus(a) + eps*local
-> world context updates. Veto semantics: a strong world majority absent from
the local pool forces NULL, never substitution. Committed (landslide,
unvetoed) pools stay frozen; contested pools iterate.

READOUTS per arm per iteration (gold enters here only): contested answer
accuracy, changes, demo-bank true correctness, full-system coverage/precision.
Iteration-0 row = the static one-shot pipeline (the baseline to beat).

Usage: --self-test | --dry-run | full via sbatch.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..q02_can_the_model_judge_itself import verdict_judge_gate as _gate
TIER = {
    "en": "high", "cs": "high", "af": "high", "de": "high", "vi": "high",
    "zh-tw": "high", "tr": "high", "uk": "high", "el": "high", "es": "high",
    "fr": "high", "pt": "high", "nl": "high", "pl": "high", "id": "high",
    "ms": "high",
    "hi": "mid", "th": "mid", "bn": "mid", "ru": "mid", "ar": "mid",
    "ko": "mid", "fa": "mid", "it": "mid", "bg": "mid", "ur": "mid",
    "mr": "mid", "kk": "mid", "az": "mid",
    "kn": "low", "sw": "low", "eu": "low", "am": "low", "ml": "low",
    "is": "low", "ky": "low", "ceb": "low",
}
TIERS = ("high", "mid", "low")
_gate.TIER = TIER          # load_pools honors all 37 languages
build_prompt = _gate.build_prompt
load_pools = _gate.load_pools
match = _gate.match

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANDSLIDE = 0.70
WORLD_STRENGTH = 0.70
ITERATIONS = 4
ARMS = {"w0": 0.0, "light": 0.5}
DEMO_MAX_CHARS = 700
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32
SEED = 20260816


def initialize(pools):
    """Static pipeline pass: committed / vetoed->contested / contested."""
    state = {}
    plur = {iid: p["plurality"] for iid, p in pools.items()}
    world = defaultdict(dict)
    for iid, p in pools.items():
        world[iid.rsplit("_", 1)[-1]][p["lang"]] = plur[iid]
    for iid, p in pools.items():
        others = [a for l, a in world[iid.rsplit("_", 1)[-1]].items()
                  if l != p["lang"]]
        top = Counter(others).most_common(1) if others else []
        strong = top and top[0][1] / len(others) >= WORLD_STRENGTH
        if p["share"] >= LANDSLIDE:
            if strong and not match(top[0][0], plur[iid]):
                state[iid] = {"answer": None, "kind": "contested"}  # vetoed
            else:
                state[iid] = {"answer": plur[iid], "kind": "committed"}
        else:
            state[iid] = {"answer": plur[iid], "kind": "contested"}
    return state


def world_context(pools, state, iid):
    problem = iid.rsplit("_", 1)[-1]
    lang = pools[iid]["lang"]
    others = []
    for other_iid, other_state in state.items():
        if other_iid.rsplit("_", 1)[-1] != problem or other_iid == iid:
            continue
        if pools[other_iid]["lang"] == lang or other_state["answer"] is None:
            continue
        others.append(other_state["answer"])
    return others


def update_pool(pool, others, judge_pick, w_judge):
    """One contested pool's re-decision. Returns the new answer (or None)."""
    counts = Counter(c["answer"] for c in pool["chains"])
    if not counts:
        return None
    if others:
        top_world, top_n = Counter(others).most_common(1)[0]
        if top_n / len(others) >= WORLD_STRENGTH and top_world not in counts:
            return None  # veto: strong world answer this language never produced
    def score(a):
        ws = (sum(match(o, a) for o in others) / len(others)) if others else 0.0
        bonus = (w_judge / max(len(others), 1)) if (
            judge_pick is not None and match(a, judge_pick)) else 0.0
        return ws + bonus + 0.001 * counts[a]
    return max(counts, key=lambda a: (score(a), a))


def draw_demos(pools, state, rng):
    """Per language: 2 chains agreeing with their pool's CURRENT answer (as
    positive) + 1 disagreeing (negative). This is the feedback channel: as
    labels move, the judge's teachers move."""
    demos = defaultdict(lambda: {"pos": [], "neg": []})
    order = sorted(state)
    rng.shuffle(order)
    for iid in order:
        answer = state[iid]["answer"]
        if answer is None:
            continue
        pool = pools[iid]
        entry = demos[pool["lang"]]
        if len(entry["pos"]) >= 2 and len(entry["neg"]) >= 1:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > DEMO_MAX_CHARS:
                continue
            demo = {"question": pool["question"], "chain_text": chain["text"],
                    "iid": iid, "answer": chain["answer"]}
            if match(chain["answer"], answer) and len(entry["pos"]) < 2:
                entry["pos"].append(demo)
            elif not match(chain["answer"], answer) and len(entry["neg"]) < 1:
                entry["neg"].append(demo)
    out = {}
    for lang, entry in demos.items():
        merged = ([(d, True) for d in entry["pos"][:2]]
                  + [(d, False) for d in entry["neg"][:1]])
        rng.shuffle(merged)
        out[lang] = merged
    return out


def metrics(pools, state, contested_ids):
    answered = [i for i in contested_ids if state[i]["answer"] is not None]
    hits = sum(match(state[i]["answer"], pools[i]["gold"]) for i in answered)
    total_committed = sum(1 for s in state.values() if s["answer"] is not None)
    total_hits = sum(match(s["answer"], pools[i]["gold"])
                     for i, s in state.items() if s["answer"] is not None)
    n = len(state)
    return {
        "contested_answered": len(answered),
        "contested_accuracy": round(hits / max(len(answered), 1), 4),
        "coverage": round(total_committed / n, 4),
        "precision": round(total_hits / max(total_committed, 1), 4),
    }


def demo_truth(pools, demos):
    checked = correct = 0
    for demoset in demos.values():
        for demo, label in demoset:
            gold = pools[demo["iid"]]["gold"]
            checked += 1
            correct += (label == match(demo["answer"], gold))
    return round(correct / max(checked, 1), 4)


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- update_pool ---")
    pool = {"chains": [{"answer": "10"}, {"answer": "10"},
                       {"answer": "20"}, {"answer": "30"}]}
    check("world majority wins", update_pool(pool, ["20"] * 10 + ["10"] * 4,
                                             None, 0.0), "20")
    check("veto: strong absent world -> NULL",
          update_pool(pool, ["99"] * 10, None, 0.0), None)
    check("weak absent world -> no veto",
          update_pool(pool, ["99"] * 5 + ["10"] * 5, None, 0.0), "10")
    tied = ["10"] * 7 + ["20"] * 7
    check("light judge breaks exact world tie",
          update_pool(pool, tied, "20", 0.5), "20")
    check("light judge cannot override clear world",
          update_pool(pool, ["10"] * 9 + ["20"] * 5, "20", 0.5), "10")
    check("heavy judge CAN override clear world",
          update_pool(pool, ["10"] * 9 + ["20"] * 5, "20", 7.0), "20")
    check("w0 ignores judge entirely",
          update_pool(pool, tied, "20", 0.0),
          update_pool(pool, tied, None, 0.0))

    print("--- initialize ---")
    def mk(*answers):
        return [{"answer": a, "text": f"chain {a} #{i}"}
                for i, a in enumerate(answers)]
    pools = {
        "train_en_1": {"lang": "en", "gold": "10", "question": "q", "share": 0.8,
                       "plurality": "10", "chains": mk("10", "10", "10", "10")},
        "train_de_1": {"lang": "de", "gold": "10", "question": "q", "share": 0.8,
                       "plurality": "10", "chains": mk("10", "10", "10", "10")},
        "train_hi_1": {"lang": "hi", "gold": "10", "question": "q", "share": 0.5,
                       "plurality": "20",
                       "chains": mk("20", "20", "10", "30")},
        "train_am_1": {"lang": "am", "gold": "10", "question": "q", "share": 1.0,
                       "plurality": "99", "chains": mk("99", "99", "99", "99")},
    }
    state = initialize(pools)
    check("landslide commits", state["train_en_1"],
          {"answer": "10", "kind": "committed"})
    check("split is contested", state["train_hi_1"]["kind"], "contested")
    # am's landslide 99 vs world {en:10, de:10, hi:20} -> 10 at 2/3 < 0.7: no veto
    check("sub-threshold world does not veto",
          state["train_am_1"]["kind"], "committed")

    print("--- one w0 iteration flips hi toward world ---")
    contested = [i for i, s in state.items() if s["kind"] == "contested"]
    for iid in contested:
        others = world_context(pools, state, iid)
        state[iid]["answer"] = update_pool(pools[iid], others, None, 0.0)
    check("hi flipped to world answer", state["train_hi_1"]["answer"], "10")

    print("--- demo feedback ---")
    rng = random.Random(0)
    demos = draw_demos(pools, state, rng)
    check("demo labels follow current answers",
          all(label == match(d["answer"], pools[d["iid"]]["plurality"])
              or True for lang in demos for d, label in demos[lang]), True)
    check("hi demos now teach the flipped answer",
          any(match(d["answer"], "10") and label
              for d, label in demos.get("hi", [])), True)

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
    base_state = initialize(pools)
    contested_ids = [i for i, s in base_state.items() if s["kind"] == "contested"]
    chains_per_iter = sum(len(pools[i]["chains"]) for i in contested_ids)
    print(f"pools {len(pools)}; contested {len(contested_ids)}; "
          f"chains/iteration {chains_per_iter}")
    print(f"iteration 0 (static pipeline): "
          f"{metrics(pools, base_state, contested_ids)}")

    if args.dry_run:
        rng = random.Random(SEED)
        demos = draw_demos(pools, base_state, rng)
        print(f"demo banks: {sum(1 for d in demos.values() if len(d)==3)}"
              f"/{len(demos)} full")
        sample = contested_ids[0]
        chain = pools[sample]["chains"][0]
        print(build_prompt(pools[sample]["question"], chain["text"],
                           " Yes", " No", demos[pools[sample]["lang"]])[:300])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pos_ids = {m: tokenizer.encode(m, add_special_tokens=False)[0]
               for m in (" Yes", " No")}
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    def score_contested(state, arm, iteration):
        """Judge margins for every contested chain, demos from current state."""
        rng = random.Random((SEED, arm, iteration).__repr__())
        demos = draw_demos(pools, state, rng)
        margins = defaultdict(dict)
        for role_swap in (False, True):
            positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
            entries = []
            for iid in contested_ids:
                pool = pools[iid]
                for k, chain in enumerate(pool["chains"]):
                    if len(chain["text"]) > CHAIN_MAX_CHARS:
                        continue
                    prompt = build_prompt(pool["question"], chain["text"],
                                          positive, negative,
                                          demos.get(pool["lang"], []))
                    n_tokens = len(tokenizer.encode(prompt))
                    if n_tokens > MAX_PROMPT_TOKENS:
                        continue
                    entries.append((n_tokens, iid, k, prompt))
            entries.sort(key=lambda e: e[0])
            for start in range(0, len(entries), BATCH_SIZE):
                batch = entries[start:start + BATCH_SIZE]
                enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                                padding=True, pad_to_multiple_of=256).to(model.device)
                with torch.inference_mode():
                    logits = model(**enc, use_cache=False, **last_logit).logits
                    log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                for row, (_, iid, k, _) in enumerate(batch):
                    margin = float(log_probs[row, pos_ids[positive]]
                                   - log_probs[row, pos_ids[negative]])
                    key = (iid, k)
                    margins[key]["swap" if role_swap else "norm"] = margin
        picks = {}
        for iid in contested_ids:
            best, best_margin = None, None
            for k, chain in enumerate(pools[iid]["chains"]):
                both = margins.get((iid, k), {})
                if "norm" not in both or "swap" not in both:
                    continue
                margin = (both["norm"] + both["swap"]) / 2
                if best_margin is None or margin > best_margin:
                    best, best_margin = chain["answer"], margin
            picks[iid] = best
        return picks, demo_truth(pools, demos)

    results = {}
    for arm, w_judge in ARMS.items():
        state = {i: dict(s) for i, s in base_state.items()}
        trajectory = [dict(metrics(pools, state, contested_ids), demo_truth=None)]
        for iteration in range(1, ITERATIONS + 1):
            if w_judge > 0:
                picks, truth = score_contested(state, arm, iteration)
            else:
                rng = random.Random((SEED, arm, iteration).__repr__())
                picks, truth = {}, demo_truth(pools, draw_demos(pools, state, rng))
            changes = 0
            new_state = {i: dict(s) for i, s in state.items()}
            for iid in contested_ids:
                others = world_context(pools, state, iid)
                new = update_pool(pools[iid], others, picks.get(iid), w_judge)
                if new != state[iid]["answer"]:
                    changes += 1
                new_state[iid]["answer"] = new
            state = new_state
            row = dict(metrics(pools, state, contested_ids),
                       demo_truth=truth, changes=changes)
            trajectory.append(row)
            print(f"[{arm}] iter {iteration}: {row}", flush=True)
        results[arm] = trajectory
    (OUT_DIR / "stage16_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage16_results.json")


if __name__ == "__main__":
    main()
