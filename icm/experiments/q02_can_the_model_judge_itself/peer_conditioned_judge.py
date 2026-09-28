"""Stage 19 — PEER-CONDITIONED verdict judge: MP evidence from the SAME
problem in OTHER languages, on the stage-8 gate sample (14B, phaseA).

WHY (from stage18's autopsy, lr_autopsy.log):
  * verdict-token readout works (J3 0.636 contested) and full-chain
    likelihood reading fails (LR2 0.536) -> KEEP the [V] Yes/No readout.
  * other-problem demos steer the model by only ~+-2 nats (they teach
    format, not evidence) -> REPLACE demos with same-problem peers.
  * step-level same-problem signal is large but was never given to the
    judge (rung3: values-Jaccard 0.86 both-right vs 0.27 one-wrong).

ARMS (both: role-swap debiased, language-centered also reported):
  P3full   context = question + up to 3 peer chains for the SAME problem
           from OTHER languages (label-blind: each peer = a chain agreeing
           with its own language's plurality, landslide languages
           preferred), then the target chain, then [V] Yes/No.
  P3blind  same, but each peer's final '#### N' answer line is stripped,
           so agreement must be read from the reasoning steps.
           CAVEAT: the answer may still leak in peer prose; blinding is
           partial and stated as such.

PRE-REGISTERED READS:
  * P3blind contested AUC > 0.65 -> step-level MP signal beyond the vote.
  * P3full >> P3blind -> the judge mostly string-matches answers
    (re-derived voting; adds nothing the vote lacks).
  * gate for standalone use unchanged: contested AUC >= 0.70.
Reference points: J3 0.636 (centered 0.653), C3 0.631, LR2 0.536.

Usage: --self-test | --dry-run | full via sbatch (stage19_peer.sbatch).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from .verdict_judge_gate import (LANDSLIDE, SEED, auc, build_demo_bank,
                               gate_sample, load_pools)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
N_PEERS = 3
PEER_MAX_CHARS = 600
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 4096
BATCH_SIZE = 16
ANSWER_LINE = re.compile(r"^\s*#{4}.*$", re.M)


def strip_answer_line(text):
    """Remove GSM8K '#### N' final-answer lines (P3blind peers)."""
    return ANSWER_LINE.sub("", text).rstrip()


def build_peer_bank(pools, rng):
    """problem -> lang -> one label-blind representative chain (agrees with
    that pool's own plurality, shortest under PEER_MAX_CHARS)."""
    bank = defaultdict(dict)
    for iid in sorted(pools):
        pool = pools[iid]
        problem = iid.rsplit("_", 1)[-1]
        candidates = [c for c in pool["chains"]
                      if c["answer"] == pool["plurality"]
                      and len(c["text"]) <= PEER_MAX_CHARS]
        if candidates:
            pick = min(candidates, key=lambda c: len(c["text"]))
            bank[problem][pool["lang"]] = {
                "text": pick["text"], "landslide": pool["share"] >= LANDSLIDE}
    return bank


def select_peers(peer_bank, problem, target_lang, rng):
    """Up to N_PEERS other-language chains for this problem; landslide
    languages preferred (label-blind), order shuffled deterministically."""
    entry = peer_bank.get(problem, {})
    langs = [l for l in entry if l != target_lang]
    rng.shuffle(langs)
    langs.sort(key=lambda l: not entry[l]["landslide"])   # landslides first
    return [entry[l]["text"] for l in langs[:N_PEERS]]


def build_peer_prompt(question, peers, chain_text, positive, negative,
                      blind=False):
    header = ("<ICM>\nSolutions to the same problem in several languages "
              "are shown, then a final attempt.\nMark [V] with" + positive
              + " if the final attempt is correct and" + negative
              + " if it is incorrect.\n\n[Q]\n" + question.strip())
    blocks = [header]
    for peer in peers:
        text = strip_answer_line(peer) if blind else peer
        blocks.append("[S]\n" + text.strip()[:PEER_MAX_CHARS])
    blocks.append("[R]\n" + chain_text.strip()[:CHAIN_MAX_CHARS] + "\n[V]")
    return "\n\n".join(blocks)


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


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- strip_answer_line ---")
    check("answer line removed", strip_answer_line("2+2=4\n#### 4"), "2+2=4")
    check("mid-text #### removed too",
          strip_answer_line("a\n#### 7\nb"), "a\n\nb")
    check("no marker untouched", strip_answer_line("2+2=4"), "2+2=4")

    print("--- peer bank + selection on SMOKE data ---")
    pools = load_pools("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-smoke")
    rng = random.Random(0)
    bank = build_peer_bank(pools, rng)
    check("bank has problems", len(bank) > 100, True)
    some_problem = next(p for p in bank if len(bank[p]) == 2)
    peers_en = select_peers(bank, some_problem, "en", random.Random(1))
    check("target language excluded", len(peers_en), 1)  # smoke = en+am only
    # label-blindness: peer text agrees with plurality by construction
    for problem, entry in list(bank.items())[:5]:
        for lang, rec in entry.items():
            pool = pools[f"train_{lang}_{problem}"]
            found = any(c["text"] == rec["text"]
                        and c["answer"] == pool["plurality"]
                        for c in pool["chains"])
            check(f"peer {lang}/{problem} is a plurality chain", found, True)
        break

    print("--- prompt ---")
    prompt = build_peer_prompt("Q?", ["P1\n#### 5", "P2\n#### 5"], "C",
                               " Yes", " No", blind=False)
    check("peers present", prompt.count("[S]"), 2)
    check("query open", prompt.endswith("[V]"), True)
    check("full arm keeps answers", "#### 5" in prompt, True)
    blind = build_peer_prompt("Q?", ["P1\n#### 5"], "C", " Yes", " No",
                              blind=True)
    check("blind arm strips answers", "####" not in blind, True)
    check("blind keeps reasoning", "P1" in blind, True)

    print("--- report smoke ---")
    fake = [{"lang": "en", "tier": "high", "contested": True, "is_gold": g,
             "P3full": (2.0 if g else 0.5) + 0.01 * i}
            for i, g in enumerate([True, False] * 20)]
    res = report(fake, ["P3full"])
    check("separable fake AUC", res["P3full"]["ALL"]["auc"] > 0.9, True)

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
    rng = random.Random(SEED)
    demos = build_demo_bank(pools, rng)      # same rng order as stage8 ->
    sample = [s for s in gate_sample(pools, rng)   # identical gate sample
              if s["instance_id"] not in
              {d["iid"] for ds in demos.values() for d, _ in ds}]
    peer_bank = build_peer_bank(pools, random.Random(SEED + 2))
    peer_rng = random.Random(SEED + 3)
    peers_of = {}
    for s in sample:
        iid = s["instance_id"]
        if iid not in peers_of:
            peers_of[iid] = select_peers(
                peer_bank, iid.rsplit("_", 1)[-1], s["lang"],
                random.Random((SEED, iid).__repr__()))
    n_full = sum(1 for p in peers_of.values() if len(p) == N_PEERS)
    print(f"pools {len(pools)}; gate sample {len(sample)}; "
          f"instances with {N_PEERS} peers: {n_full}/{len(peers_of)}")

    if args.dry_run:
        s = sample[0]
        prompt = build_peer_prompt(s["question"], peers_of[s["instance_id"]],
                                   s["text"], " Yes", " No")
        print(f"\n--- P3full prompt (first 600 of {len(prompt)} chars) ---")
        print(prompt[:600])
        print("\n--- P3blind peer example ---")
        print(strip_answer_line(peers_of[s["instance_id"]][0])[:200])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    for arm, blind in (("P3full", False), ("P3blind", True)):
        for role_swap in (False, True):
            positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
            pos_id = tokenizer.encode(positive, add_special_tokens=False)[0]
            neg_id = tokenizer.encode(negative, add_special_tokens=False)[0]
            entries = []
            for index, s in enumerate(sample):
                peers = peers_of[s["instance_id"]]
                if not peers:
                    continue
                prompt = build_peer_prompt(s["question"], peers, s["text"],
                                           positive, negative, blind=blind)
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
                if start % (BATCH_SIZE * 40) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        for s in sample:
            n, w = s.get(f"{arm}_norm"), s.get(f"{arm}_swap")
            s[arm] = (n + w) / 2 if n is not None and w is not None else None

    results = report(sample, ["P3full", "P3blind"])
    (OUT_DIR / "stage19_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage19_chains.jsonl").open("w", encoding="utf-8") as handle:
        for s in sample:
            handle.write(json.dumps(
                {k: v for k, v in s.items() if k not in ("text", "question")},
                ensure_ascii=False) + "\n")
    print("wrote stage19_results.json and stage19_chains.jsonl")


if __name__ == "__main__":
    main()
