"""Stage 8 — THE GATE: can Qwen3-14B-Base judge its own chains above chance?

This is Phase A's deciding measurement, the fork between two hypotheses:
  capability hypothesis (user/advisor): the 8B judge failed because the model
      was too weak; a stronger same-family base model judges usefully.
  identity hypothesis: self-judgment duplicates the vote regardless of scale.

Design mirrors E3 exactly for comparability with the 8B numbers:
  * per pool, up to 2 gold + 2 non-gold chains (gold used ONLY to build this
    diagnostic sample and in evaluation — never in prompts or demos)
  * judge = the same 14B that generated the chains
  * J3: instruction + 3 pseudo-labeled in-language demos (from local-landslide
    pools, label-blind) + Yes/No markers; J0: no-demo control
  * both marker orders, debiased margin = mean of role-consistent margins

PRE-REGISTERED GATE: chain-level AUC on CONTESTED pools (top-answer share
< 0.7) >= 0.70 -> Phase B (full ICM energy at 14B) is justified.
8B reference points: 0.51-0.58 contested, 0.584-0.658 overall.

Usage: --self-test | --dry-run [--root SMOKE_ROOT] | full via sbatch.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
TIER = {
    "en": "high", "de": "high", "es": "high", "fr": "high",
    "ru": "mid", "hi": "mid", "ar": "mid", "ko": "mid", "fa": "mid",
    "am": "low", "sw": "low", "kn": "low", "eu": "low", "ml": "low", "is": "low",
}
TIERS = ("high", "mid", "low")
LANDSLIDE = 0.70
CHAIN_MAX_CHARS = 1800
DEMO_MAX_CHARS = 700
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32
GATE_AUC = 0.70
SEED = 20260815
CONDITIONS = ("J0", "J3")


def norm(x):
    try:
        d = Decimal(str(x).replace(",", "").replace(" ", ""))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not d.is_finite():
        return None
    d = d.normalize()
    if d == d.to_integral_value():
        try:
            d = d.quantize(Decimal(1))
        except InvalidOperation:
            return None  # absurd magnitude (degenerate digit-run): unusable
    return f"{d:f}"


def match(a, b):
    if a is None or b is None:
        return False
    x, y = Decimal(a), Decimal(b)
    if x == y:
        return True
    s = max(abs(x), abs(y))
    return s > 0 and abs(x - y) / s <= Decimal("0.005")


def auc(pos, neg):
    if not pos or not neg:
        return None
    ranked = sorted([(v, 0) for v in neg] + [(v, 1) for v in pos],
                    key=lambda t: t[0])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop][0] == ranked[index][0]:
            stop += 1
        rank_sum += (index + stop + 1) / 2 * sum(f for _, f in ranked[index:stop])
        index = stop
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_pools(root):
    """instance -> {gold, question, lang, chains:[{answer,text}], share}."""
    pools = {}
    for path in sorted(glob.glob(f"{root}/final/pooled/generated/*.jsonl")):
        lang = Path(path).stem
        if lang not in TIER:
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold = norm(row.get("gold_number"))
                chains = []
                for c in row.get("candidates", []):
                    answer = norm(c.get("pred_number"))
                    text = c.get("text") or ""
                    if answer is None or not text.strip():
                        continue
                    if c.get("finish_reason") == "length":
                        continue
                    chains.append({"answer": answer, "text": text})
                if not chains or gold is None:
                    continue
                counts = Counter(c["answer"] for c in chains)
                top_n = counts.most_common(1)[0][1]
                pools[row["id"]] = {
                    "gold": gold, "lang": lang, "question": row["question"],
                    "chains": chains, "share": top_n / len(chains),
                    "plurality": max(counts, key=lambda a: (counts[a], a)),
                }
    return pools


def build_demo_bank(pools, rng):
    """3 pseudo-labeled demos per language from LOCAL-LANDSLIDE pools only
    (label-blind: pseudo-label = agrees with the landslide answer)."""
    bank = defaultdict(lambda: {"pos": [], "neg": []})
    for instance_id in sorted(pools):
        pool = pools[instance_id]
        if pool["share"] < LANDSLIDE:
            continue
        entry = bank[pool["lang"]]
        if len(entry["pos"]) >= 2 and len(entry["neg"]) >= 1:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > DEMO_MAX_CHARS:
                continue
            demo = {"question": pool["question"], "chain_text": chain["text"],
                    "iid": instance_id, "answer": chain["answer"]}
            if chain["answer"] == pool["plurality"] and len(entry["pos"]) < 2:
                entry["pos"].append(demo)
            elif chain["answer"] != pool["plurality"] and len(entry["neg"]) < 1:
                entry["neg"].append(demo)
    demos = {}
    for lang, entry in bank.items():
        merged = ([(d, True) for d in entry["pos"][:2]]
                  + [(d, False) for d in entry["neg"][:1]])
        rng.shuffle(merged)
        demos[lang] = merged
    return demos


def gate_sample(pools, rng):
    """Up to 2 gold + 2 non-gold chains per pool (diagnostic sampling, E3-style).
    Excludes demo pools implicitly? Demos come from landslides; sample includes
    all pools but demo INSTANCES are excluded per-language at scoring time."""
    sample = []
    for instance_id in sorted(pools):
        pool = pools[instance_id]
        gold_chains = [c for c in pool["chains"]
                       if match(c["answer"], pool["gold"])
                       and len(c["text"]) <= CHAIN_MAX_CHARS]
        rest = [c for c in pool["chains"]
                if not match(c["answer"], pool["gold"])
                and len(c["text"]) <= CHAIN_MAX_CHARS]
        rng.shuffle(gold_chains)
        rng.shuffle(rest)
        for chain, is_gold in ([(c, True) for c in gold_chains[:2]]
                               + [(c, False) for c in rest[:2]]):
            sample.append({"instance_id": instance_id, "lang": pool["lang"],
                           "tier": TIER[pool["lang"]], "question": pool["question"],
                           "text": chain["text"], "is_gold": is_gold,
                           "contested": pool["share"] < LANDSLIDE})
    return sample


def build_prompt(question, chain_text, positive, negative, demos=()):
    header = ("<ICM>\nMark [V] with" + positive
              + " if the reasoning is correct and" + negative
              + " if it is incorrect.\n\n")
    blocks = []
    for demo, label in demos:
        blocks.append("[Q]\n" + demo["question"].strip() + "\n[R]\n"
                      + demo["chain_text"].strip()[:DEMO_MAX_CHARS] + "\n[V]"
                      + (positive if label else negative))
    blocks.append("[Q]\n" + question.strip() + "\n[R]\n"
                  + chain_text.strip()[:CHAIN_MAX_CHARS] + "\n[V]")
    return header + "\n\n".join(blocks)


def report(sample):
    results = {}
    for name in CONDITIONS:
        scored = [s for s in sample if s.get(name) is not None]
        print("=" * 74)
        print(f"CONDITION {name}  ({len(scored)} chains scored)")
        print("=" * 74)
        block = {}
        print(f"{'slice':<12}{'n_gold':>8}{'n_rest':>8}{'AUC':>8}")
        for label, pred in (("ALL", lambda s: True),
                            ("contested", lambda s: s["contested"]),
                            ("landslide", lambda s: not s["contested"]),
                            ("high", lambda s: s["tier"] == "high"),
                            ("mid", lambda s: s["tier"] == "mid"),
                            ("low", lambda s: s["tier"] == "low")):
            gold_m = [s[name] for s in scored if pred(s) and s["is_gold"]]
            rest_m = [s[name] for s in scored if pred(s) and not s["is_gold"]]
            score = auc(gold_m, rest_m)
            block[label] = {"auc": round(score, 3) if score is not None else None,
                            "n_gold": len(gold_m), "n_rest": len(rest_m)}
            print(f"{label:<12}{len(gold_m):>8}{len(rest_m):>8}"
                  f"{score if score is not None else float('nan'):>8.3f}")
        results[name] = block
        contested = block["contested"]["auc"]
        verdict = ("GATE PASSED" if contested is not None and contested >= GATE_AUC
                   else "GATE NOT PASSED")
        print(f"verdict ({name}): contested AUC {contested} vs {GATE_AUC} -> {verdict}")
        results[name]["verdict"] = verdict
    return results


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- norm/match/auc ---")
    check("norm scientific trap", norm("2190.0"), "2190")
    check("auc separable", auc([2.0, 1.5], [0.1, 0.2]), 1.0)
    check("auc chance", auc([1.0, 1.0], [1.0, 1.0]), 0.5)

    print("--- load_pools on the SMOKE data (real schema) ---")
    pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-smoke")
    check("smoke languages loaded", sorted({p['lang'] for p in pools.values()}),
          ["am", "en"])
    check("300 pools", len(pools), 300)
    some = next(iter(pools.values()))
    check("chains have answers", all("answer" in c for c in some["chains"]), True)
    check("no truncated chains kept",
          all(len(c["text"]) > 0 for c in some["chains"]), True)

    print("--- demo bank (label-blind) ---")
    rng = random.Random(0)
    demos = build_demo_bank(pools, rng)
    check("both languages have banks", sorted(demos), ["am", "en"])
    for lang in ("en", "am"):
        labels = sorted(label for _, label in demos[lang])
        check(f"{lang} bank has 2 pos + 1 neg", labels, [False, True, True])
    # label-blindness: demos derive from plurality, never from gold
    for lang, demoset in demos.items():
        for demo, label in demoset:
            pool = pools[demo["iid"]]
            check(f"{lang} demo label follows plurality not gold",
                  label, demo["answer"] == pool["plurality"])
            break

    print("--- gate sample ---")
    sample = gate_sample(pools, random.Random(1))
    check("sample nonempty", len(sample) > 500, True)
    golds = sum(s["is_gold"] for s in sample)
    check("both classes present", 0 < golds < len(sample), True)
    contested = sum(s["contested"] for s in sample)
    check("contested slice exists", contested > 0, True)

    print("--- prompt ---")
    prompt = build_prompt("Q?", "C", " Yes", " No",
                          [({"question": "DQ", "chain_text": "DC"}, True)])
    check("demo labeled", "[V] Yes" in prompt, True)
    check("query open", prompt.endswith("[V]"), True)

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
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load_pools(args.root)
    rng = random.Random(SEED)
    demos = build_demo_bank(pools, rng)
    demo_instances = {demo["iid"] for demoset in demos.values()
                      for demo, _ in demoset}
    sample = [s for s in gate_sample(pools, rng)
              if s["instance_id"] not in demo_instances]
    print(f"pools {len(pools)}; gate sample {len(sample)} chains "
          f"({sum(s['is_gold'] for s in sample)} gold / "
          f"{sum(not s['is_gold'] for s in sample)} non-gold); "
          f"contested {sum(s['contested'] for s in sample)}")
    print(f"languages with full 3-demo banks: "
          f"{sum(1 for d in demos.values() if len(d) == 3)}/{len(TIER)}")

    if args.dry_run:
        s = sample[0]
        print("\n--- sample J3 prompt (first 400 chars) ---")
        print(build_prompt(s["question"], s["text"], " Yes", " No",
                           demos.get(s["lang"], []))[:400])
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
            raise ValueError(f"marker {marker!r} not single-token")
        return ids[0]

    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    for name in CONDITIONS:
        for role_swap in (False, True):
            positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
            pos_id, neg_id = marker_id(positive), marker_id(negative)
            entries = []
            for index, s in enumerate(sample):
                demo_set = demos.get(s["lang"], []) if name == "J3" else ()
                prompt = build_prompt(s["question"], s["text"],
                                      positive, negative, demo_set)
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > MAX_PROMPT_TOKENS:
                    continue
                entries.append((n_tokens, index, prompt))
            entries.sort(key=lambda e: e[0])
            key = f"{name}{'_swap' if role_swap else ''}"
            print(f"[{key}] scoring {len(entries)} chains", flush=True)
            for start in range(0, len(entries), BATCH_SIZE):
                batch = entries[start:start + BATCH_SIZE]
                enc = tokenizer([p for _, _, p in batch], return_tensors="pt",
                                padding=True, pad_to_multiple_of=256).to(model.device)
                with torch.inference_mode():
                    logits = model(**enc, use_cache=False, **last_logit).logits
                    log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                for row, (_, index, _) in enumerate(batch):
                    margin = float(log_probs[row, pos_id] - log_probs[row, neg_id])
                    sample[index][key] = margin
                if start % (BATCH_SIZE * 40) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        for s in sample:
            normal, swapped = s.get(name), s.get(f"{name}_swap")
            s[name] = ((normal + swapped) / 2
                       if normal is not None and swapped is not None else None)

    results = report(sample)
    (OUT_DIR / f"stage8_results{args.tag}.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / f"stage8_chains{args.tag}.jsonl").open("w", encoding="utf-8") as handle:
        for s in sample:
            handle.write(json.dumps({k: v for k, v in s.items()
                                     if k not in ("text", "question")},
                                    ensure_ascii=False) + "\n")
    print("wrote stage8_results.json and stage8_chains.jsonl")


if __name__ == "__main__":
    main()
