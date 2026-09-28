"""Stage 21 — THE PROMOTION TEST: ensemble content judge (J3 + continuation)
in the full iterated-consensus loop. Does the upgraded judge graduate from
whisper (+2.6 single-seed) to core component?

Channels (chain-level, per-language z-scored, combined additively):
  J3  verdict margin, demos re-drawn from CURRENT labels each lap (stage9)
  cc  continuation consistency (stage20): fraction of k=4 regenerations of
      the chain's 60%-prefix landing on its own answer. Lap-independent
      (format demos are the static label-blind landslide bank) -> computed
      ONCE in --phase cc (vLLM venv), cached, reused every lap.
Gate-sample reference: J3 0.646, cc 0.643, ensemble 0.674 contested AUC.

ARMS (identical pools, 4 laps, SEED as stage9):
  w0    no judge                      (stage9 replication: 86.6 contested)
  j3    J3 pick, w=0.5                (stage9 replication: 89.2)
  ens   ensemble pick (zJ3+zcc), w=0.5
  ens2  ensemble pick, w=2.0          (does a better judge earn more weight?
                                       heavy J3 collapsed at w=7)

PRE-REGISTERED READS (15 languages):
  * ens contested > 89.2 -> the ensemble adds in-loop value over the whisper
  * "core component" bar: ens (or ens2) - w0 >= ~4 points contested, or
    ens2 sustains gains where overweighted J3 declined
  * ens <= 89.2 -> content ensemble is absorbed like its parts; the judge's
    core role remains harvest cleaning (world-filter two-stage), not labeling

Usage:
  --phase cc   [--dry-run]   vLLM continuation scoring (multilingual-icm venv)
  --phase loop [--dry-run]   the 4-arm loop (sm-icm venv, reads cc cache)
  --self-test
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from .verdict_judge_gate import build_prompt, load_pools, match
from ..q01_can_voting_label_data.consensus_loop_15lang import (draw_demos, demo_truth, initialize, metrics,
                         update_pool, world_context)
from .continuation_consistency_judge import (build_cont_prompt, consistency,
                                  truncate_chain)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
CC_CACHE = OUT_DIR / "stage21_cc.jsonl"
LANDSLIDE = 0.70
ITERATIONS = 4
ARMS = {"w0": (0.0, None), "j3": (0.5, "j3"),
        "ens": (0.5, "ens"), "ens2": (2.0, "ens")}
K_SAMPLES = 4
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 400
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 3000
MAX_MODEL_LEN = 4096
BATCH_SIZE = 32
SEED = 20260816          # stage9's seed: identical init, demos, trajectories


def zscores_by_lang(values_by_key, lang_of):
    """(iid,k)->value -> (iid,k)->z within language; None -> 0.0."""
    per_lang = defaultdict(list)
    for key, value in values_by_key.items():
        if value is not None:
            per_lang[lang_of(key)].append(value)
    mu = {l: st.mean(v) for l, v in per_lang.items()}
    sd = {l: max(st.pstdev(v), 1e-9) for l, v in per_lang.items()}
    out = {}
    for key, value in values_by_key.items():
        lang = lang_of(key)
        out[key] = (0.0 if value is None or lang not in mu
                    else (value - mu[lang]) / sd[lang])
    return out


def ensemble_picks(pools, contested_ids, j3_margins, cc_scores):
    """Per contested pool: answer of the argmax-(zJ3+zcc) chain."""
    lang_of = lambda key: pools[key[0]]["lang"]
    z_j3 = zscores_by_lang(j3_margins, lang_of)
    z_cc = zscores_by_lang(cc_scores, lang_of)
    picks = {}
    for iid in contested_ids:
        best, best_score = None, None
        for k, chain in enumerate(pools[iid]["chains"]):
            key = (iid, k)
            if j3_margins.get(key) is None and cc_scores.get(key) is None:
                continue
            score = z_j3.get(key, 0.0) + z_cc.get(key, 0.0)
            if best_score is None or score > best_score:
                best, best_score = chain["answer"], score
        picks[iid] = best
    return picks


def j3_picks(pools, contested_ids, j3_margins):
    """stage9 semantics: answer of the argmax-margin chain."""
    picks = {}
    for iid in contested_ids:
        best, best_margin = None, None
        for k, chain in enumerate(pools[iid]["chains"]):
            margin = j3_margins.get((iid, k))
            if margin is None:
                continue
            if best_margin is None or margin > best_margin:
                best, best_margin = chain["answer"], margin
        picks[iid] = best
    return picks


def static_pos_demos(pools):
    """Label-blind continuation format demos: per language, 2 short chains
    agreeing with a landslide pool's plurality (stage20 convention)."""
    demos = defaultdict(list)
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["share"] < LANDSLIDE or len(demos[pool["lang"]]) >= 2:
            continue
        for chain in pool["chains"]:
            if (chain["answer"] == pool["plurality"]
                    and len(chain["text"]) <= 700
                    and len(demos[pool["lang"]]) < 2):
                demos[pool["lang"]].append(
                    ({"question": pool["question"],
                      "chain_text": chain["text"]}, True))
    return demos


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- zscores_by_lang ---")
    vals = {("a", 0): 1.0, ("a", 1): 3.0, ("b", 0): None}
    z = zscores_by_lang(vals, lambda k: "en")
    check("centered", round(z[("a", 0)] + z[("a", 1)], 6), 0.0)
    check("None -> 0", z[("b", 0)], 0.0)

    print("--- ensemble_picks ---")
    pools = {"train_en_1": {"lang": "en", "chains": [
        {"answer": "10", "text": "t0"}, {"answer": "20", "text": "t1"}]}}
    j3 = {("train_en_1", 0): 2.0, ("train_en_1", 1): 0.0}
    cc = {("train_en_1", 0): 0.0, ("train_en_1", 1): 1.0}
    check("j3-only pick", j3_picks(pools, ["train_en_1"], j3)["train_en_1"],
          "10")
    check("cc flips a j3-margin tie in combined z",
          ensemble_picks(pools, ["train_en_1"],
                         {("train_en_1", 0): 1.0, ("train_en_1", 1): 1.0},
                         cc)["train_en_1"], "20")
    check("agreeing channels reinforce",
          ensemble_picks(pools, ["train_en_1"], j3,
                         {("train_en_1", 0): 1.0, ("train_en_1", 1): 0.0}
                         )["train_en_1"], "10")
    check("missing both channels -> other chain wins",
          ensemble_picks(pools, ["train_en_1"],
                         {("train_en_1", 1): 5.0}, {})["train_en_1"], "20")

    print("--- arms table sanity ---")
    check("w0 has no picker", ARMS["w0"], (0.0, None))
    check("ens2 overweights ensemble", ARMS["ens2"], (2.0, "ens"))

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def phase_cc(args):
    pools = load_pools(args.root)
    base_state = initialize(pools)
    contested_ids = [i for i, s in base_state.items()
                     if s["kind"] == "contested"]
    demos = static_pos_demos(pools)
    jobs = []
    for iid in contested_ids:
        pool = pools[iid]
        for k, chain in enumerate(pool["chains"]):
            prefix = truncate_chain(chain["text"])
            if prefix is None:
                continue
            prompt = build_cont_prompt(pool["question"], prefix,
                                       demos.get(pool["lang"], []))
            jobs.append((iid, k, chain["answer"], prompt))
    print(f"pools {len(pools)}; contested {len(contested_ids)}; "
          f"continuation jobs {jobs and len(jobs)}")
    if args.dry_run:
        print(f"sample prompt (first 400):\n{jobs[0][3][:400]}")
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_dir, dtype="bfloat16",
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              seed=SEED)
    params = SamplingParams(n=K_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                            max_tokens=MAX_NEW_TOKENS, stop=["[Q]"],
                            seed=SEED)
    outputs = llm.generate([j[3] for j in jobs], params)
    with CC_CACHE.open("w", encoding="utf-8") as handle:
        for (iid, k, claim, _), out in zip(jobs, outputs):
            score, n_parsed = consistency([o.text for o in out.outputs], claim)
            handle.write(json.dumps({"iid": iid, "k": k, "cc": score,
                                     "n_parsed": n_parsed}) + "\n")
    print(f"wrote {CC_CACHE} ({len(jobs)} chains)")


def phase_loop(args):
    pools = load_pools(args.root)
    base_state = initialize(pools)
    contested_ids = [i for i, s in base_state.items()
                     if s["kind"] == "contested"]
    cc_scores = {}
    if CC_CACHE.exists():
        for line in CC_CACHE.open(encoding="utf-8"):
            row = json.loads(line)
            cc_scores[(row["iid"], row["k"])] = row["cc"]
    covered = sum(1 for iid in contested_ids
                  for k in range(len(pools[iid]["chains"]))
                  if (iid, k) in cc_scores)
    total = sum(len(pools[iid]["chains"]) for iid in contested_ids)
    print(f"pools {len(pools)}; contested {len(contested_ids)}; "
          f"cc cache covers {covered}/{total} contested chains")
    print(f"iteration 0: {metrics(pools, base_state, contested_ids)}")
    if not args.dry_run and covered == 0:
        raise SystemExit("cc cache missing/empty -- run --phase cc first")
    if args.dry_run:
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

    def score_margins(state, arm, iteration):
        """J3 margins for contested chains, demos from current state
        (stage9's score_contested, returning raw margins)."""
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
                    margins[(iid, k)]["swap" if role_swap else "norm"] = margin
        merged = {}
        for key, both in margins.items():
            if "norm" in both and "swap" in both:
                merged[key] = (both["norm"] + both["swap"]) / 2
        return merged, demo_truth(pools, demos)

    results = {}
    for arm, (w_judge, picker) in ARMS.items():
        state = {i: dict(s) for i, s in base_state.items()}
        trajectory = [dict(metrics(pools, state, contested_ids),
                           demo_truth=None)]
        for iteration in range(1, ITERATIONS + 1):
            if picker is not None:
                margins, truth = score_margins(state, arm, iteration)
                if picker == "ens":
                    picks = ensemble_picks(pools, contested_ids, margins,
                                           cc_scores)
                else:
                    picks = j3_picks(pools, contested_ids, margins)
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
    (OUT_DIR / "stage21_results.json").write_text(json.dumps(results, indent=2))
    print("wrote stage21_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--phase", choices=("cc", "loop"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.phase == "cc":
        phase_cc(args)
    elif args.phase == "loop":
        phase_loop(args)
    else:
        parser.error("need --self-test or --phase {cc,loop}")


if __name__ == "__main__":
    main()
