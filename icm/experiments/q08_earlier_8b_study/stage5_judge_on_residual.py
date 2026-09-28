"""Stage 5 probe — the ICM judge on the pipeline's residual (GPU, ~1h).

The final pipeline (vote -> world-veto -> constrained tie-break) commits ~88%
of problem-language pairs at ~95% precision and abstains on the rest: pools
where the world is split too, plus vetoed landslides. This probe scores EVERY
chain in those residual pools with the best-measured judge configurations and
answers the pre-registered question:

    Does any judge-margin slice of the residual reach >=70% accuracy,
    i.e. is there a promotable fourth tier, or does the ladder end at
    best-guess?

Judge configs (from E3, the only ones that measured above chance):
    D_ab    instruction + " A"/" B" markers, no demonstrations   (E3: 0.658)
    D_yn    instruction + " Yes"/" No" markers, no demonstrations (untested
            combination of E3's two winning ingredients)
Each chain is scored twice per config (marker roles swapped); the debiased
margin is the mean of the two role-consistent margins.

Residual selection reuses rung7's pipeline_decisions verbatim, so "residual"
here is exactly what the deployed pipeline abstains on at full pool size.

Usage: --self-test (no GPU) | --dry-run (no GPU) | full run via sbatch.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .rung7_small_pools_and_routing import (
    TIER,
    TIERS,
    load,
    match,
    pipeline_decisions,
)

MODEL_DIR = "/lustre10/scratch/rvyalla/sm-icm/assets/models/Qwen3-8B-Base"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 2500
BATCH_SIZE = 32
PROMOTION_TARGET = 0.70   # pre-registered: a slice must reach this to be promoted

CONDITIONS = {
    "D_ab": (" A", " B"),
    "D_yn": (" Yes", " No"),
}


def build_prompt(question: str, chain: str, positive: str, negative: str) -> str:
    return ("<ICM>\nMark [V] with" + positive
            + " if the reasoning is correct and" + negative
            + " if it is incorrect.\n\n[Q]\n" + question.strip()
            + "\n[R]\n" + chain.strip()[:CHAIN_MAX_CHARS] + "\n[V]")


def residual_items(pools, gold):
    """Instances the deployed pipeline abstains on, with their chains."""
    full = {instance_id: [answer for _, answer in chains]
            for instance_id, chains in pools.items()}
    decisions = pipeline_decisions(full)
    residual = []
    for instance_id, decision in decisions.items():
        if decision["answer"] is not None or gold.get(instance_id) is None:
            continue
        residual.append({
            "instance_id": instance_id,
            "lang": "_".join(instance_id.split("_")[1:-1]),
            "tier": TIER["_".join(instance_id.split("_")[1:-1])],
            "route": decision["route"],
            "gold": gold[instance_id],
        })
    return residual


def decide(chains: list[dict], key: str) -> dict:
    """Pick the answer of the highest-margin chain under condition `key`."""
    scored = [c for c in chains if c.get(key) is not None]
    if not scored:
        return {"answer": None, "margin": None}
    best = max(scored, key=lambda c: c[key])
    runner = max((c[key] for c in scored
                  if not match(c["answer"], best["answer"])), default=None)
    gap = best[key] - runner if runner is not None else best[key]
    return {"answer": best["answer"], "margin": round(best[key], 4),
            "gap": round(gap, 4)}


def report(items, chains_by_instance, results_key="results"):
    results = {}
    for condition in CONDITIONS:
        print("=" * 78)
        print(f"CONDITION {condition}")
        print("=" * 78)
        block = {}
        for item in items:
            item[condition] = decide(chains_by_instance[item["instance_id"]], condition)
        print(f"{'tier':<6}{'n':>7}{'judge':>9}{'forced-vote':>13}{'random':>9}")
        for tier in TIERS + ("ALL",):
            rows = [i for i in items if tier in ("ALL", i["tier"])]
            if not rows:
                continue
            judged = [i for i in rows if i[condition]["answer"] is not None]
            hits = sum(match(i[condition]["answer"], i["gold"]) for i in judged)
            forced = 0
            rand = []
            for i in rows:
                chains = chains_by_instance[i["instance_id"]]
                counts = Counter(c["answer"] for c in chains)
                if counts:
                    top = max(counts, key=lambda a: (counts[a], a))
                    forced += match(top, i["gold"])
                    rand.append(sum(match(c["answer"], i["gold"]) for c in chains)
                                / len(chains))
            block[tier] = {
                "n": len(rows), "judged": len(judged),
                "judge_acc": round(hits / max(len(judged), 1), 4),
                "forced_vote_acc": round(forced / max(len(rows), 1), 4),
                "random_acc": round(st.mean(rand), 4) if rand else None,
            }
            print(f"{tier:<6}{len(rows):>7}"
                  f"{100*hits/max(len(judged),1):>8.1f}%"
                  f"{100*forced/max(len(rows),1):>12.1f}%"
                  f"{100*st.mean(rand):>8.1f}%" if rand else "")

        # The pre-registered question: the promotion frontier by margin.
        judged = sorted((i for i in items if i[condition]["answer"] is not None),
                        key=lambda i: -(i[condition]["margin"] or 0))
        frontier = []
        print(f"\npromotion frontier (target {PROMOTION_TARGET:.0%}):")
        print(f"{'keep top':>10}{'n':>7}{'accuracy':>11}")
        promotable = None
        for keep in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
            top = judged[:max(1, int(keep * len(judged)))]
            acc = sum(match(i[condition]["answer"], i["gold"]) for i in top) / len(top)
            frontier.append({"keep": keep, "n": len(top), "accuracy": round(acc, 4)})
            print(f"{keep:>9.0%}{len(top):>7}{100*acc:>10.1f}%")
            if acc >= PROMOTION_TARGET and promotable is None:
                promotable = {"keep": keep, "n": len(top), "accuracy": round(acc, 4)}
        block["frontier"] = frontier
        block["promotable_slice"] = promotable
        verdict = (f"PROMOTABLE: top {promotable['keep']:.0%} "
                   f"({promotable['n']} items) at {100*promotable['accuracy']:.1f}%"
                   if promotable else
                   f"NO promotable slice at the {PROMOTION_TARGET:.0%} target")
        print(f"verdict: {verdict}\n")
        block["verdict"] = verdict
        results[condition] = block
    return results


def self_test() -> None:
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- build_prompt ---")
    prompt = build_prompt("Q?", "6*365=2190\n#### 2190", " A", " B")
    check("has instruction", "Mark [V] with A" in prompt, True)
    check("ends at verdict slot", prompt.endswith("[V]"), True)
    check("swap changes instruction",
          "Mark [V] with B" in build_prompt("Q?", "x", " B", " A"), True)
    check("chain truncated", len(build_prompt("Q?", "x" * 5000, " A", " B")) < 2600, True)

    print("--- decide ---")
    chains = [
        {"answer": "2190", "D_ab": 1.5},
        {"answer": "1825", "D_ab": 0.3},
        {"answer": "1825", "D_ab": -0.2},
    ]
    picked = decide(chains, "D_ab")
    check("max margin wins", picked["answer"], "2190")
    check("gap vs best other answer", picked["gap"], 1.2)
    check("no scores -> None", decide([{"answer": "5"}], "D_ab")["answer"], None)
    tie_pool = [{"answer": "7", "D_ab": 0.9}]
    check("single chain works", decide(tie_pool, "D_ab")["answer"], "7")

    print("--- residual selection consistency ---")
    fake_pools = {
        "train_en_1": [("c1", "10")] * 8 + [("c2", "20")] * 7,   # split locally
        "train_de_1": [("c3", "10")] * 15,                        # landslide "10"
        "train_hi_1": [("c4", "20")] * 15,                        # landslide "20"
        "train_ru_1": [("c5", "30")] * 15,                        # landslide "30"
    }
    # en's jury = {de:10, hi:20, ru:30} -> no 70% world majority -> abstain.
    fake_gold = {k: "10" for k in fake_pools}
    residual = residual_items(fake_pools, fake_gold)
    check("world-split pool is residual",
          "train_en_1" in {r["instance_id"] for r in residual}, True)
    # de's landslide is uncontested by a 70% world majority -> committed.
    check("committed landslide is NOT residual",
          "train_de_1" in {r["instance_id"] for r in residual}, False)
    # a jury with a unanimous 70%+ majority present in the local pool commits:
    fake_pools2 = dict(fake_pools)
    fake_pools2["train_hi_1"] = [("c4", "10")] * 15
    fake_pools2["train_ru_1"] = [("c5", "10")] * 15
    residual2 = residual_items(fake_pools2, fake_gold)
    check("strong world majority resolves the split",
          "train_en_1" in {r["instance_id"] for r in residual2}, False)

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    import glob
    import pyarrow.parquet as pq

    pools, gold = load()
    items = residual_items(pools, gold)
    print(f"residual pools: {len(items)} "
          f"({Counter(i['route'] for i in items)})")
    print(f"by tier: {Counter(i['tier'] for i in items)}")

    wanted = {i["instance_id"] for i in items}
    chains_by_instance = defaultdict(list)
    RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(
                path, columns=["instance_id", "canonical_answer", "parse_status",
                               "question", "chain_text"]).to_pylist():
            if row["instance_id"] not in wanted or row["parse_status"] != "parsed":
                continue
            from .rung7_small_pools_and_routing import norm
            answer = norm(row["canonical_answer"])
            if answer is None:
                continue
            chains_by_instance[row["instance_id"]].append(
                {"answer": answer, "question": row["question"],
                 "chain_text": row["chain_text"]})
    total_chains = sum(len(v) for v in chains_by_instance.values())
    print(f"chains to score: {total_chains} "
          f"(x {2 * len(CONDITIONS)} passes = {total_chains * 2 * len(CONDITIONS)})")

    if args.dry_run:
        sample = chains_by_instance[items[0]["instance_id"]][0]
        print("\n--- sample prompt (first 500 chars) ---")
        print(build_prompt(sample["question"], sample["chain_text"], " A", " B")[:500])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def marker_id(marker):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"marker {marker!r} not a single token: {ids}")
        return ids[0]

    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    last_logit = {"logits_to_keep": 1} if "logits_to_keep" in forward_params else {}

    flat = []
    for instance_id, chains in chains_by_instance.items():
        for index, chain in enumerate(chains):
            flat.append((instance_id, index, chain))

    for name, (positive, negative) in CONDITIONS.items():
        for role_swap in (False, True):
            pos_marker, neg_marker = (negative, positive) if role_swap else (positive, negative)
            pos_id, neg_id = marker_id(pos_marker), marker_id(neg_marker)
            entries = []
            for instance_id, index, chain in flat:
                prompt = build_prompt(chain["question"], chain["chain_text"],
                                      pos_marker, neg_marker)
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > MAX_PROMPT_TOKENS:
                    continue
                entries.append((n_tokens, instance_id, index, prompt))
            entries.sort(key=lambda e: e[0])
            key = f"{name}{'_swap' if role_swap else ''}"
            print(f"[{key}] scoring {len(entries)} chains", flush=True)
            for start in range(0, len(entries), BATCH_SIZE):
                batch = entries[start:start + BATCH_SIZE]
                enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                                padding=True, pad_to_multiple_of=256).to(model.device)
                with torch.inference_mode():
                    logits = model(**enc, use_cache=False, **last_logit).logits
                    log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)
                for row, (_, instance_id, index, _) in enumerate(batch):
                    margin = float(log_probs[row, pos_id] - log_probs[row, neg_id])
                    chains_by_instance[instance_id][index][key] = margin
                if start % (BATCH_SIZE * 40) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        # debias: content margin = mean of the two role-consistent margins
        for chains in chains_by_instance.values():
            for chain in chains:
                normal, swapped = chain.get(name), chain.get(f"{name}_swap")
                chain[name] = ((normal + swapped) / 2
                               if normal is not None and swapped is not None else None)

    results = report(items, chains_by_instance)
    (OUT_DIR / "stage5_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage5_items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps({
                **{k: item[k] for k in ("instance_id", "tier", "route", "gold")},
                **{c: item.get(c) for c in CONDITIONS},
            }, ensure_ascii=False) + "\n")
    print(f"wrote stage5_results.json and stage5_items.jsonl")


if __name__ == "__main__":
    main()
