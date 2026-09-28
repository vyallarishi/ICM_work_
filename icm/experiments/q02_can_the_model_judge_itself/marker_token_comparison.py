"""Stage 41 — DOES THE MARKER TOKEN CHANGE THE VERDICT JUDGE'S SIGNAL?

MOTIVATION: marker choice has demonstrably mattered before.
  * 8B forensics: the score was dominated by a preference for the letter "A"
    (marker-bias component 0.86 vs content 0.17 over 400k cached verdicts).
  * E3 condition C: switching the markers to Yes/No moved overall AUC
    0.584 -> 0.626 with everything else held fixed.
  * stage32's step-level judge used " +"/" -" and reached 0.835/0.891, but
    that formulation changed the QUESTION as well as the tokens, so the
    marker's own contribution was never isolated.

THIS RUN isolates it: stage8's verdict judge, identical sample, identical
demos, identical debiasing -- only the marker pair varies.

ARMS (each scored in both role orders and averaged, as in stage8):
  yesno    " Yes" / " No"      stage8's shipped markers (replication anchor)
  plusmin  " +"   / " -"       stage32's markers, the thing being tested
  truefalse" True"/ " False"   the original SM-ICM markers, for completeness
  ab       " A"   / " B"       the 8B failure case, as a negative control:
                               if this scores near the others, marker choice
                               is not what rescued anything

PRE-REGISTERED:
  * The comparison that matters is plusmin vs yesno on CONTESTED pools.
    stage8 reference: J3 contested 0.636, ALL 0.732.
  * A gain >= 0.03 contested would make marker choice a real lever and
    justify re-running earlier judges with better markers.
  * A gain < 0.01, or a negative, means stage32's strength came from the
    step-level question and the length correction, not from the tokens.
  * ab is the control: near-parity with yesno would mean this model is not
    marker-sensitive at 14B the way the 8B model was.
Every marker must be single-token or the run aborts.

Usage: --self-test | --dry-run | full via stage41_markers.sbatch
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from .verdict_judge_gate import (BATCH_SIZE, CONDITIONS, GATE_AUC,
                               MAX_PROMPT_TOKENS, SEED, auc, build_demo_bank,
                               build_prompt, gate_sample, load_pools)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
MARKERS = {
    "yesno": (" Yes", " No"),
    "plusmin": (" +", " -"),
    "truefalse": (" True", " False"),
    "ab": (" A", " B"),
}
SLICES = (("ALL", lambda s: True),
          ("contested", lambda s: s["contested"]),
          ("landslide", lambda s: not s["contested"]),
          ("high", lambda s: s["tier"] == "high"),
          ("mid", lambda s: s["tier"] == "mid"),
          ("low", lambda s: s["tier"] == "low"))


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- marker table ---")
    check("four arms", sorted(MARKERS), ["ab", "plusmin", "truefalse", "yesno"])
    check("yesno matches stage8", MARKERS["yesno"], (" Yes", " No"))
    check("plusmin matches stage32", MARKERS["plusmin"], (" +", " -"))
    check("all markers lead with a space",
          all(p.startswith(" ") and n.startswith(" ")
              for p, n in MARKERS.values()), True)
    check("positive and negative differ",
          all(p != n for p, n in MARKERS.values()), True)

    print("--- prompt built with each marker pair ---")
    demos = [({"question": "DQ", "chain_text": "DC"}, True),
             ({"question": "DQ2", "chain_text": "DC2"}, False)]
    for name, (pos, neg) in MARKERS.items():
        prompt = build_prompt("Q?", "C", pos, neg, demos)
        check(f"{name}: header names the positive marker",
              f"Mark [V] with{pos}" in prompt, True)
        check(f"{name}: positive demo labelled", f"[V]{pos}" in prompt, True)
        check(f"{name}: negative demo labelled", f"[V]{neg}" in prompt, True)
        check(f"{name}: query slot left open", prompt.endswith("[V]"), True)

    print("--- role swap flips which token means correct ---")
    pos, neg = MARKERS["plusmin"]
    normal = build_prompt("Q?", "C", pos, neg, demos)
    swapped = build_prompt("Q?", "C", neg, pos, demos)
    check("swap changes the header", normal != swapped, True)
    check("swapped header names the swapped positive",
          f"Mark [V] with{neg}" in swapped, True)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def report(sample):
    results = {}
    print(f"\n{'arm':<11}" + "".join(f"{name:>12}" for name, _ in SLICES))
    for arm in MARKERS:
        scored = [s for s in sample if s.get(arm) is not None]
        if not scored:
            continue
        block = {"n_scored": len(scored)}
        line = f"{arm:<11}"
        for label, pred in SLICES:
            g = [s[arm] for s in scored if pred(s) and s["is_gold"]]
            r = [s[arm] for s in scored if pred(s) and not s["is_gold"]]
            a = auc(g, r) if g and r else None
            block[label] = {"auc": round(a, 4) if a is not None else None,
                            "n_gold": len(g), "n_rest": len(r)}
            line += f"{a if a is not None else float('nan'):>12.4f}"
        print(line)
        results[arm] = block

    ref = results.get("yesno", {})
    print(f"\n{'arm':<11}{'contested':>12}{'vs yesno':>11}   verdict")
    for arm in MARKERS:
        if arm not in results:
            continue
        a = results[arm]["contested"]["auc"]
        base = ref.get("contested", {}).get("auc")
        delta = None if (a is None or base is None) else round(a - base, 4)
        note = ""
        if arm != "yesno" and delta is not None:
            note = ("REAL LEVER (>=0.03)" if delta >= 0.03
                    else "no meaningful gain")
        print(f"{arm:<11}{a if a is not None else float('nan'):>12.4f}"
              f"{delta if delta is not None else 0:>11.4f}   {note}")
        results[arm]["delta_vs_yesno_contested"] = delta
    best = max((results[a]["contested"]["auc"] for a in results
                if results[a]["contested"]["auc"] is not None), default=None)
    print(f"\nbest contested AUC {best} vs gate {GATE_AUC}: "
          f"{'PASSES' if best and best >= GATE_AUC else 'still below gate'}")
    results["gate"] = {"best_contested": best, "gate": GATE_AUC,
                       "passes": bool(best and best >= GATE_AUC)}
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
    rng = random.Random(SEED)
    demos = build_demo_bank(pools, rng)          # same rng order as stage8 ->
    demo_instances = {d["iid"] for ds in demos.values() for d, _ in ds}
    sample = [s for s in gate_sample(pools, rng)  # identical gate sample
              if s["instance_id"] not in demo_instances]
    print(f"pools {len(pools)}; gate sample {len(sample)} chains "
          f"({sum(s['is_gold'] for s in sample)} gold); "
          f"arms {sorted(MARKERS)}")
    print("reference (stage8 J3): contested 0.636, ALL 0.732")

    if args.dry_run:
        s = sample[0]
        for name, (pos, neg) in MARKERS.items():
            prompt = build_prompt(s["question"], s["text"], pos, neg,
                                  demos.get(s["lang"], []))
            print(f"\n--- {name} header + query tail ---")
            print(prompt[:110].replace("\n", " | "))
            print("   ...", prompt[-60:].replace("\n", " | "))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def marker_id(marker):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"marker {marker!r} is not single-token: {ids}")
        return ids[0]

    ids = {}
    for name, (pos, neg) in MARKERS.items():
        ids[name] = (marker_id(pos), marker_id(neg))
        print(f"  {name}: {pos!r}={ids[name][0]}  {neg!r}={ids[name][1]}")

    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    for arm, (pos_marker, neg_marker) in MARKERS.items():
        for role_swap in (False, True):
            positive, negative = ((neg_marker, pos_marker) if role_swap
                                  else (pos_marker, neg_marker))
            pos_id, neg_id = marker_id(positive), marker_id(negative)
            entries = []
            for index, s in enumerate(sample):
                prompt = build_prompt(s["question"], s["text"], positive,
                                      negative, demos.get(s["lang"], []))
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > MAX_PROMPT_TOKENS:
                    continue
                entries.append((n_tokens, index, prompt))
            entries.sort(key=lambda e: e[0])
            key = f"{arm}{'_swap' if role_swap else '_norm'}"
            print(f"[{key}] scoring {len(entries)} chains", flush=True)
            for start in range(0, len(entries), BATCH_SIZE):
                batch = entries[start:start + BATCH_SIZE]
                enc = tokenizer([p for _, _, p in batch], return_tensors="pt",
                                padding=True, pad_to_multiple_of=256).to(model.device)
                with torch.inference_mode():
                    logits = model(**enc, use_cache=False, **last_logit).logits
                    log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                for row, (_, index, _) in enumerate(batch):
                    sample[index][key] = float(log_probs[row, pos_id]
                                               - log_probs[row, neg_id])
                if start % (BATCH_SIZE * 60) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        for s in sample:
            n, w = s.get(f"{arm}_norm"), s.get(f"{arm}_swap")
            s[arm] = (n + w) / 2 if n is not None and w is not None else None

    results = report(sample)
    (OUT_DIR / "stage41_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage41_chains.jsonl").open("w", encoding="utf-8") as h:
        for s in sample:
            h.write(json.dumps({k: v for k, v in s.items()
                                if k not in ("text", "question")},
                               ensure_ascii=False) + "\n")
    print("\nwrote stage41_results.json and stage41_chains.jsonl")


if __name__ == "__main__":
    main()
