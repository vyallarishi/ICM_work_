"""Stage 18 — TWO REFORMULATED JUDGES on the stage-8 gate sample (14B, phaseA).

Baseline to beat: stage8 J3 verdict judge, contested AUC 0.636 (gate 0.70).

ARM 1  LR2 (likelihood-ratio / proper mutual predictability):
    score(chain) = logP(chain | Q, 2 demos agreeing w/ their landslide)
                 - logP(chain | Q, 2 demos disagreeing w/ their landslide)
    Read over ALL chain tokens instead of one Yes/No token. The same chain
    appears on both sides, so length and surface form cancel in the
    difference. Demos are label-blind (landslide pseudo-labels, stage-8
    convention), balanced 2-vs-2 per language.

ARM 2  C3 (canonicalized verdict judge):
    stage8's J3 exactly, but chain AND demo texts pass through a
    canonicalizer stripping LaTeX/markdown/step-numbering.
    PRE-REGISTERED (updated by format_confound.py, Aug 19): format features
    explain only R^2=0.025 of margin variance, so C3 is predicted to change
    contested AUC by <0.02. This arm is the direct test of that prediction.

Both arms also reported with within-language centering (label-blind), which
format_confound.py showed lifts J3 contested 0.636 -> 0.653.

READOUT: AUC gold-vs-rest on ALL/contested/landslide/tier slices, raw and
lang-centered, vs the cached J3 reference. Gate stays 0.70 contested.

Usage: --self-test | --dry-run | full via sbatch (stage18_lr.sbatch).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from .verdict_judge_gate import (CHAIN_MAX_CHARS, DEMO_MAX_CHARS, LANDSLIDE,
                               SEED, TIER, auc, build_demo_bank, build_prompt,
                               gate_sample, load_pools)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
MAX_PROMPT_TOKENS = 3000
BATCH_VERDICT = 32     # verdict passes keep only the last logit (cheap)
BATCH_LR = 8           # LR passes need full-sequence logits (memory-heavy)
LOGIT_CHUNK = 512      # sequence-chunked fp32 log-softmax for LR


# ---------------------------------------------------------------- canonical
_LATEX_SUBS = (
    (re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}"), r"(\1/\2)"),
    (re.compile(r"\\(?:text|textbf|mathrm|boxed)\{([^{}]*)\}"), r"\1"),
    (re.compile(r"\\times"), "*"),
    (re.compile(r"\\cdot"), "*"),
    (re.compile(r"\\(?:left|right|,|;|!)"), ""),
    (re.compile(r"\\\[|\\\]|\\\(|\\\)|\$\$|\$"), ""),
)
_BOLD = re.compile(r"\*\*([^*]*)\*\*")
_HEADER = re.compile(r"^\s*#{1,3}(?!#)\s*", re.M)  # NOT ####: GSM8K answer marker
_STEPNUM = re.compile(r"^\s*\d+[\.\)]\s+", re.M)
_BLANKS = re.compile(r"\n{3,}")


def canonicalize(text):
    """Strip presentation (LaTeX, bold, headers, step numbers); keep content.
    '*' as multiplication and <<...>> calculator annotations are preserved."""
    out = _BOLD.sub(r"\1", text)
    out = _HEADER.sub("", out)
    out = _STEPNUM.sub("", out)
    for pattern, repl in _LATEX_SUBS:
        out = pattern.sub(repl, out)
    out = _BLANKS.sub("\n\n", out)
    return "\n".join(line.rstrip() for line in out.split("\n")).strip()


# ---------------------------------------------------------------- LR pieces
def build_lr_bank(pools, rng):
    """Label-blind, BALANCED demo bank: per language, 2 chains agreeing with
    their landslide pool's plurality (pos) and 2 disagreeing (neg), truncated
    to equal counts. Mirrors stage8's bank but symmetric."""
    bank = defaultdict(lambda: {"pos": [], "neg": []})
    for instance_id in sorted(pools):
        pool = pools[instance_id]
        if pool["share"] < LANDSLIDE:
            continue
        entry = bank[pool["lang"]]
        if len(entry["pos"]) >= 2 and len(entry["neg"]) >= 2:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > DEMO_MAX_CHARS:
                continue
            demo = {"question": pool["question"], "chain_text": chain["text"],
                    "iid": instance_id}
            if chain["answer"] == pool["plurality"] and len(entry["pos"]) < 2:
                entry["pos"].append(demo)
            elif chain["answer"] != pool["plurality"] and len(entry["neg"]) < 2:
                entry["neg"].append(demo)
    out = {}
    for lang, entry in bank.items():
        k = min(len(entry["pos"]), len(entry["neg"]), 2)
        out[lang] = {"pos": entry["pos"][:k], "neg": entry["neg"][:k]}
    return out


def lr_context(question, demos):
    """Shared-suffix context: demo blocks then the query block, ending '[A]\\n'
    identically on both sides so the chain's first-token BPE boundary is the
    same in both contexts (any boundary effect cancels in the difference)."""
    blocks = ["<ICM>\nSolved problems.\n"]
    for demo in demos:
        blocks.append("[Q]\n" + demo["question"].strip() + "\n[A]\n"
                      + demo["chain_text"].strip()[:DEMO_MAX_CHARS])
    blocks.append("[Q]\n" + question.strip() + "\n[A]\n")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------- reporting
SLICES = (("ALL", lambda s: True),
          ("contested", lambda s: s["contested"]),
          ("landslide", lambda s: not s["contested"]),
          ("high", lambda s: s["tier"] == "high"),
          ("mid", lambda s: s["tier"] == "mid"),
          ("low", lambda s: s["tier"] == "low"))


def report(sample, arms):
    results = {}
    for arm in arms:
        scored = [s for s in sample if s.get(arm) is not None]
        if not scored:
            continue
        mu = defaultdict(list)
        for s in scored:
            mu[s["lang"]].append(s[arm])
        mu = {l: st.mean(v) for l, v in mu.items()}
        block = {}
        print("=" * 66)
        print(f"ARM {arm}  ({len(scored)} chains scored)")
        print(f"{'slice':<12}{'n_gold':>8}{'n_rest':>8}{'AUC raw':>10}{'AUC centered':>14}")
        for label, pred in SLICES:
            g = [s for s in scored if pred(s) and s["is_gold"]]
            r = [s for s in scored if pred(s) and not s["is_gold"]]
            raw = auc([s[arm] for s in g], [s[arm] for s in r])
            cen = auc([s[arm] - mu[s["lang"]] for s in g],
                      [s[arm] - mu[s["lang"]] for s in r])
            block[label] = {"auc": round(raw, 3) if raw is not None else None,
                            "auc_centered": round(cen, 3) if cen is not None else None,
                            "n_gold": len(g), "n_rest": len(r)}
            print(f"{label:<12}{len(g):>8}{len(r):>8}"
                  f"{raw if raw is not None else float('nan'):>10.3f}"
                  f"{cen if cen is not None else float('nan'):>14.3f}")
        results[arm] = block
    return results


# ---------------------------------------------------------------- self-test
def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- canonicalize ---")
    check("bold stripped", canonicalize("**Step:** 3 + 4"), "Step: 3 + 4")
    check("header stripped", canonicalize("### Solution\n2+2"), "Solution\n2+2")
    check("step numbers stripped", canonicalize("1. First\n2) Second"),
          "First\nSecond")
    check("frac rewritten", canonicalize(r"\[ \frac{36}{3} = 12 \]"),
          "(36/3) = 12")
    check("times rewritten", canonicalize(r"4 \times 12 = 48"), "4 * 12 = 48")
    check("plain GSM8K text untouched",
          canonicalize("He had 180 - 70 = <<180-70=110>>110 left.\n#### 110"),
          "He had 180 - 70 = <<180-70=110>>110 left.\n#### 110")
    check("multiplication star kept", canonicalize("3 * 4 = 12"), "3 * 4 = 12")
    check("non-latin preserved",
          canonicalize("**উত্তর:** ৩ + ৪ = ৭"), "উত্তর: ৩ + ৪ = ৭")

    print("--- lr_context suffix identity ---")
    d1 = [{"question": "Q1", "chain_text": "C1"}]
    d2 = [{"question": "Q2", "chain_text": "C2 longer text"}]
    a, b = lr_context("Qx", d1), lr_context("Qx", d2)
    check("both end with query block", a.endswith("[Q]\nQx\n[A]\n")
          and b.endswith("[Q]\nQx\n[A]\n"), True)
    check("contexts differ in demos only", a != b, True)

    print("--- lr bank on SMOKE data (real schema) ---")
    pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-smoke")
    bank = build_lr_bank(pools, random.Random(0))
    check("smoke languages banked", sorted(bank), ["am", "en"])
    for lang in bank:
        check(f"{lang} bank balanced",
              len(bank[lang]["pos"]) == len(bank[lang]["neg"]), True)
        check(f"{lang} bank nonempty", len(bank[lang]["pos"]) > 0, True)

    print("--- report smoke ---")
    fake = [{"lang": "en", "tier": "high", "contested": c, "is_gold": g,
             "LR2": (2.0 if g else 0.5) + 0.01 * i}
            for i, (g, c) in enumerate([(True, True), (False, True)] * 20)]
    res = report(fake, ["LR2"])
    check("separable fake AUC", res["LR2"]["ALL"]["auc"] > 0.9, True)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


# ---------------------------------------------------------------- main
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
    verdict_demos = build_demo_bank(pools, rng)   # same order as stage8: bank
    sample = [s for s in gate_sample(pools, rng)  # then sample, same rng
              if s["instance_id"] not in
              {d["iid"] for ds in verdict_demos.values() for d, _ in ds}]
    lr_bank = build_lr_bank(pools, random.Random(SEED + 1))
    lr_excl = {d["iid"] for entry in lr_bank.values()
               for side in ("pos", "neg") for d in entry[side]}
    print(f"pools {len(pools)}; gate sample {len(sample)} "
          f"({sum(s['is_gold'] for s in sample)} gold); "
          f"LR banks balanced at "
          f"{sorted((l, len(e['pos'])) for l, e in lr_bank.items())}")

    if args.dry_run:
        s = sample[0]
        print("\n--- canonicalized chain (first 300) ---")
        print(canonicalize(s["text"])[:300])
        print("\n--- LR pos-context (first 400) ---")
        print(lr_context(s["question"], lr_bank[s["lang"]]["pos"])[:400])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    # ---------------- ARM 2: C3 canonicalized verdict judge -------------
    tokenizer.padding_side = "left"
    canon_demos = {lang: [({**d, "chain_text": canonicalize(d["chain_text"])},
                           label) for d, label in ds]
                   for lang, ds in verdict_demos.items()}
    for role_swap in (False, True):
        positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
        pos_id = tokenizer.encode(positive, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(negative, add_special_tokens=False)[0]
        entries = []
        for index, s in enumerate(sample):
            prompt = build_prompt(s["question"], canonicalize(s["text"]),
                                  positive, negative,
                                  canon_demos.get(s["lang"], []))
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens > MAX_PROMPT_TOKENS:
                continue
            entries.append((n_tokens, index, prompt))
        entries.sort(key=lambda e: e[0])
        key = "C3_swap" if role_swap else "C3_norm"
        print(f"[{key}] scoring {len(entries)} chains", flush=True)
        for start in range(0, len(entries), BATCH_VERDICT):
            batch = entries[start:start + BATCH_VERDICT]
            enc = tokenizer([p for _, _, p in batch], return_tensors="pt",
                            padding=True, pad_to_multiple_of=256).to(model.device)
            with torch.inference_mode():
                logits = model(**enc, use_cache=False, **last_logit).logits
                log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
            for row, (_, index, _) in enumerate(batch):
                sample[index][key] = float(log_probs[row, pos_id]
                                           - log_probs[row, neg_id])
            if start % (BATCH_VERDICT * 40) == 0:
                print(f"  {start}/{len(entries)}", flush=True)
                torch.cuda.empty_cache()
    for s in sample:
        n, w = s.get("C3_norm"), s.get("C3_swap")
        s["C3"] = (n + w) / 2 if n is not None and w is not None else None

    # ---------------- ARM 1: LR2 likelihood-ratio judge -----------------
    tokenizer.padding_side = "right"

    def chain_logprob_batch(rows):
        """rows: (index, context, chain). Returns summed logP of chain tokens."""
        full = [c + ch for _, c, ch in rows]
        ctx_lens = [len(tokenizer.encode(c)) for _, c, _ in rows]
        enc = tokenizer(full, return_tensors="pt", padding=True).to(model.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits
            out = []
            for row in range(ids.shape[0]):
                length = int(mask[row].sum())
                lo, hi = ctx_lens[row], length          # predict tokens [lo, hi)
                total = 0.0
                for cs in range(lo, hi, LOGIT_CHUNK):
                    ce = min(cs + LOGIT_CHUNK, hi)
                    lg = logits[row, cs - 1:ce - 1, :].float()
                    lse = torch.logsumexp(lg, dim=-1)
                    tgt = ids[row, cs:ce]
                    total += float((lg.gather(-1, tgt[:, None])[:, 0] - lse).sum())
                out.append(total)
        return out

    for side in ("pos", "neg"):
        entries = []
        for index, s in enumerate(sample):
            bank = lr_bank.get(s["lang"])
            if not bank or not bank["pos"] or s["instance_id"] in lr_excl:
                continue
            context = lr_context(s["question"], bank[side])
            chain = s["text"].strip()[:CHAIN_MAX_CHARS] + "\n"
            n_tokens = (len(tokenizer.encode(context))
                        + len(tokenizer.encode(chain)))
            if n_tokens > MAX_PROMPT_TOKENS:
                continue
            entries.append((n_tokens, index, context, chain))
        entries.sort(key=lambda e: e[0])
        key = f"LR_{side}"
        print(f"[{key}] scoring {len(entries)} chains", flush=True)
        for start in range(0, len(entries), BATCH_LR):
            batch = entries[start:start + BATCH_LR]
            sums = chain_logprob_batch([(i, c, ch) for _, i, c, ch in batch])
            for (_, index, _, _), value in zip(batch, sums):
                sample[index][key] = value
            if start % (BATCH_LR * 100) == 0:
                print(f"  {start}/{len(entries)}", flush=True)
                torch.cuda.empty_cache()
    for s in sample:
        p, n = s.get("LR_pos"), s.get("LR_neg")
        s["LR2"] = p - n if p is not None and n is not None else None

    results = report(sample, ["C3", "LR2"])
    (OUT_DIR / "stage18_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage18_chains.jsonl").open("w", encoding="utf-8") as handle:
        for s in sample:
            handle.write(json.dumps(
                {k: v for k, v in s.items() if k not in ("text", "question")},
                ensure_ascii=False) + "\n")
    print("wrote stage18_results.json and stage18_chains.jsonl")


if __name__ == "__main__":
    main()
