"""Stage 6 — the USER-SPEC configuration, run exactly as specified, head-to-head
against the current config. Pre-registered.

THE SPEC (differs from the deployed pipeline in all four contested choices):
  1. n = 4 chains per problem-language pool (seeded subsample, trial 0 —
     identical sampling to rung7's n=4 trial 0).
  2. GLOBAL-vote-first ordering: commit the >=70% world-majority answer if the
     language generated it; only then fall back to the local landslide;
     otherwise the pool is residual.
  3. The judge is MUTUAL PREDICTABILITY proper: 3 labeled in-language examples
     (pseudo-labels from this run's own committed pools — label-blind, demo
     problems disjoint from the residual by construction) + instruction +
     Yes/No verdict, role-swap debiased. A 0-demo control runs alongside.
  4. The residual pick is the FUSION rule (world share + judge agreement +
     epsilon local votes) that scored 43.2% on the full-pool residual.

PRE-REGISTERED QUESTIONS
  Q1  does 3-demo mutual predictability beat the 0-demo verifier? (E3 only
      measured 0 and 8 demos; 3 is the untested middle)
  Q2  does fusion beat world-leader-alone on this residual?
  Q3  end-to-end: user-spec vs current config at n=4 — coverage/precision,
      overall and low tier.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .rung7_small_pools_and_routing import SEED, TIER, TIERS, load, match

MODEL_DIR = "/lustre10/scratch/rvyalla/sm-icm/assets/models/Qwen3-8B-Base"
RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
N_CHAINS = 4
TRIAL = 0
WORLD_STRENGTH = 0.70
LANDSLIDE = 0.70
DEMO_MAX_CHARS = 700
QUERY_MAX_CHARS = 1500
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32

CONDITIONS = ("J0", "J3")   # 0-demo control, 3-demo mutual predictability


def subsample(pools) -> dict[str, list[int]]:
    """Indices of the sampled chains per pool (trial 0, rung7-identical seed)."""
    rng = random.Random((SEED, N_CHAINS, TRIAL).__repr__())
    picks = {}
    for instance_id, chains in pools.items():
        answers = [a for _, a in chains]
        if len(answers) > N_CHAINS:
            indices = rng.sample(range(len(answers)), N_CHAINS)
            # rng.sample on the VALUES was used in rung7; to stay faithful we
            # sample values the same way and then map back to first-matching
            # indices, preserving the answer multiset exactly.
            sampled = rng2_free = None
        picks[instance_id] = answers
    return picks


def subsample_answers(pools):
    """Sampled answer lists, byte-identical to rung7 n=4 trial 0."""
    rng = random.Random((SEED, N_CHAINS, TRIAL).__repr__())
    out = {}
    for instance_id, chains in pools.items():
        answers = [a for _, a in chains]
        out[instance_id] = (rng.sample(answers, N_CHAINS)
                           if len(answers) > N_CHAINS else answers)
    return out


def global_first_decisions(sub: dict[str, list[str]]):
    plur = defaultdict(dict)
    for instance_id, answers in sub.items():
        if not answers:
            continue
        counts = Counter(answers)
        plur[instance_id.rsplit("_", 1)[-1]]["_".join(instance_id.split("_")[1:-1])] = \
            max(counts, key=lambda a: (counts[a], a))
    decisions = {}
    for instance_id, answers in sub.items():
        if not answers:
            decisions[instance_id] = {"answer": None, "route": "empty"}
            continue
        counts = Counter(answers)
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        others = [p for l, p in plur[problem].items() if l != lang]
        world = Counter(others).most_common(1) if others else []
        if world and world[0][1] / len(others) >= WORLD_STRENGTH and world[0][0] in counts:
            decisions[instance_id] = {"answer": world[0][0], "route": "global"}
        elif counts.most_common(1)[0][1] / len(answers) >= LANDSLIDE:
            decisions[instance_id] = {"answer": max(counts, key=lambda a: (counts[a], a)),
                                      "route": "local"}
        else:
            decisions[instance_id] = {"answer": None, "route": "residual"}
    return decisions, plur


def build_demo_bank(sub, decisions, chain_lookup, rng):
    """Per language: 3 pseudo-labeled demos (2 correct, 1 incorrect) drawn from
    COMMITTED pools of this run — label-blind, disjoint from the residual."""
    bank = defaultdict(lambda: {"pos": [], "neg": []})
    for instance_id, decision in decisions.items():
        if decision["answer"] is None:
            continue
        lang = "_".join(instance_id.split("_")[1:-1])
        entry = bank[lang]
        if len(entry["pos"]) >= 2 and len(entry["neg"]) >= 1:
            continue
        for chain in chain_lookup.get(instance_id, []):
            if chain["answer"] not in sub[instance_id]:
                continue
            if len(chain["chain_text"]) > DEMO_MAX_CHARS:
                continue
            if match(chain["answer"], decision["answer"]) and len(entry["pos"]) < 2:
                entry["pos"].append(chain)
            elif not match(chain["answer"], decision["answer"]) and len(entry["neg"]) < 1:
                entry["neg"].append(chain)
    demos = {}
    for lang, entry in bank.items():
        merged = ([(c, True) for c in entry["pos"][:2]]
                  + [(c, False) for c in entry["neg"][:1]])
        rng.shuffle(merged)
        demos[lang] = merged
    return demos


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
                  + chain_text.strip()[:QUERY_MAX_CHARS] + "\n[V]")
    return header + "\n\n".join(blocks)


def fusion_pick(local_counts, world_share, judge_answer, n_others):
    def score(answer):
        bonus = (0.5 / max(n_others, 1)) if (judge_answer is not None
                                             and match(answer, judge_answer)) else 0.0
        return world_share.get(answer, 0.0) + bonus + 0.001 * local_counts[answer]
    return max(local_counts, key=lambda a: (score(a), a))


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- global_first_decisions ---")
    sub = {
        "train_en_1": ["10", "10", "20", "30"],       # split locally
        "train_de_1": ["10", "10", "10", "10"],
        "train_hi_1": ["10", "10", "10", "10"],
        "train_ru_1": ["10", "10", "10", "10"],
        "train_am_1": ["40", "40", "40", "20"],       # local landslide, world says 10 (absent)
    }
    decisions, _ = global_first_decisions(sub)
    check("world majority commits the split", decisions["train_en_1"]["answer"], "10")
    check("route is global", decisions["train_en_1"]["route"], "global")
    check("world answer absent -> local landslide", decisions["train_am_1"]["answer"], "40")
    check("route is local", decisions["train_am_1"]["route"], "local")
    sub2 = dict(sub)
    sub2["train_am_1"] = ["40", "40", "20", "30"]      # split AND world answer absent
    decisions2, _ = global_first_decisions(sub2)
    check("no world match + split -> residual", decisions2["train_am_1"]["answer"], None)
    check("residual route", decisions2["train_am_1"]["route"], "residual")

    print("--- build_prompt (3-demo mutual predictability) ---")
    demos = [({"question": "DQ1", "chain_text": "DC1"}, True),
             ({"question": "DQ2", "chain_text": "DC2"}, False),
             ({"question": "DQ3", "chain_text": "DC3"}, True)]
    prompt = build_prompt("Q?", "C", " Yes", " No", demos)
    check("three demo blocks", prompt.count("[Q]"), 4)
    check("positive demo labeled", "[V] Yes" in prompt, True)
    check("negative demo labeled", "[V] No" in prompt, True)
    check("query slot open", prompt.endswith("[V]"), True)
    swap = build_prompt("Q?", "C", " No", " Yes", demos)
    check("swap flips demo labels", "DC1" in swap and "[V] No" in swap.split("DC1")[1][:30], True)

    print("--- demo bank label-blindness ---")
    chain_lookup = {
        "train_en_2": [{"answer": "5", "chain_text": "c", "question": "q"},
                       {"answer": "7", "chain_text": "c2", "question": "q"}],
    }
    sub3 = {"train_en_2": ["5", "5", "5", "7"]}
    decisions3 = {"train_en_2": {"answer": "5", "route": "local"}}
    bank = build_demo_bank(sub3, decisions3, chain_lookup, random.Random(0))
    labels = sorted(label for _, label in bank["en"])
    check("pseudo-labels follow the committed answer", labels, [False, True])

    print("--- fusion_pick ---")
    pick = fusion_pick(Counter({"10": 1, "20": 3}), {"10": 0.4, "20": 0.1}, "20", 36)
    check("world share dominates local count", pick, "10")
    pick2 = fusion_pick(Counter({"10": 2, "20": 2}), {"10": 0.2, "20": 0.2}, "20", 36)
    check("judge breaks exact world ties", pick2, "20")

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
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    import glob
    import pyarrow.parquet as pq

    pools, gold = load()
    sub = subsample_answers(pools)
    decisions, plur = global_first_decisions(sub)

    # committed-tier report (Q3, deterministic part)
    def tier_stats(tier=None):
        committed = correct = total = 0
        for instance_id, decision in decisions.items():
            g = gold.get(instance_id)
            if g is None:
                continue
            if tier and TIER["_".join(instance_id.split("_")[1:-1])] != tier:
                continue
            total += 1
            if decision["answer"] is not None:
                committed += 1
                correct += match(decision["answer"], g)
        return total, committed, correct
    print("=" * 78)
    print(f"USER-SPEC committed tier (n={N_CHAINS}, global-first, trial {TRIAL})")
    print("=" * 78)
    results = {"committed": {}}
    for tier in TIERS + (None,):
        total, committed, correct = tier_stats(tier)
        label = tier or "ALL"
        results["committed"][label] = {
            "n": total, "coverage": round(committed / total, 4),
            "precision": round(correct / max(committed, 1), 4)}
        print(f"{label:<6} n={total:<6} coverage {100*committed/total:5.1f}%  "
              f"precision {100*correct/max(committed,1):5.1f}%")

    residual = [i for i, d in decisions.items()
                if d["route"] == "residual" and gold.get(i)]
    print(f"\nresidual pools: {len(residual)}")

    # chains for residual pools: the EXACT sampled chains
    wanted = set(residual) | {i for i, d in decisions.items() if d["answer"] is not None}
    chain_lookup = defaultdict(list)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(
                path, columns=["instance_id", "canonical_answer", "parse_status",
                               "question", "chain_text"]).to_pylist():
            if row["instance_id"] not in wanted or row["parse_status"] != "parsed":
                continue
            from .rung7_small_pools_and_routing import norm
            answer = norm(row["canonical_answer"])
            if answer is not None:
                chain_lookup[row["instance_id"]].append(
                    {"answer": answer, "question": row["question"],
                     "chain_text": row["chain_text"]})

    # map sampled answers back to concrete chains (first match per answer copy)
    residual_chains = {}
    for instance_id in residual:
        available = list(chain_lookup[instance_id])
        picked = []
        for answer in sub[instance_id]:
            index = next((k for k, c in enumerate(available)
                          if c["answer"] == answer), None)
            if index is not None:
                picked.append(available.pop(index))
        residual_chains[instance_id] = picked
    total_chains = sum(len(v) for v in residual_chains.values())
    rng = random.Random(SEED + 1)
    demos = build_demo_bank(sub, decisions, chain_lookup, rng)
    have3 = sum(1 for d in demos.values() if len(d) == 3)
    print(f"chains to score: {total_chains} x 8 passes")
    print(f"languages with full 3-demo banks: {have3}/37")

    if args.dry_run:
        sample_id = residual[0]
        lang = "_".join(sample_id.split("_")[1:-1])
        chain = residual_chains[sample_id][0]
        print("\n--- sample J3 prompt (first 700 chars) ---")
        print(build_prompt(chain["question"], chain["chain_text"], " Yes", " No",
                           demos.get(lang, []))[:700])
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
            raise ValueError(f"marker {marker!r} not single-token: {ids}")
        return ids[0]

    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    flat = [(instance_id, k, chain)
            for instance_id, chains in residual_chains.items()
            for k, chain in enumerate(chains)]
    for name in CONDITIONS:
        for role_swap in (False, True):
            positive, negative = (" No", " Yes") if role_swap else (" Yes", " No")
            pos_id, neg_id = marker_id(positive), marker_id(negative)
            entries = []
            for instance_id, k, chain in flat:
                lang = "_".join(instance_id.split("_")[1:-1])
                demo_set = demos.get(lang, []) if name == "J3" else ()
                prompt = build_prompt(chain["question"], chain["chain_text"],
                                      positive, negative, demo_set)
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens > MAX_PROMPT_TOKENS:
                    continue
                entries.append((n_tokens, instance_id, k, prompt))
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
                for row, (_, instance_id, k, _) in enumerate(batch):
                    margin = float(log_probs[row, pos_id] - log_probs[row, neg_id])
                    residual_chains[instance_id][k][key] = margin
                if start % (BATCH_SIZE * 40) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        for chains in residual_chains.values():
            for chain in chains:
                normal, swapped = chain.get(name), chain.get(f"{name}_swap")
                chain[name] = ((normal + swapped) / 2
                               if normal is not None and swapped is not None else None)

    # ---------------- residual decisions & report -----------------------
    print()
    print("=" * 78)
    print("RESIDUAL TIER — pre-registered comparisons")
    print("=" * 78)
    rows = []
    for instance_id in residual:
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        local_counts = Counter(sub[instance_id])
        others = [p for l, p in plur[problem].items() if l != lang]
        world_share = {a: sum(match(p, a) for p in others) / max(len(others), 1)
                       for a in local_counts}
        judges = {}
        for name in CONDITIONS:
            scored = [c for c in residual_chains[instance_id] if c.get(name) is not None]
            judges[name] = (max(scored, key=lambda c: c[name])["answer"]
                            if scored else None)
        rows.append({
            "instance_id": instance_id, "tier": TIER[lang], "gold": gold[instance_id],
            "forced_local": max(local_counts, key=lambda a: (local_counts[a], a)),
            "world_leader": max(local_counts, key=lambda a: (world_share[a], a)),
            **{f"judge_{n}": judges[n] for n in CONDITIONS},
            **{f"fusion_{n}": fusion_pick(local_counts, world_share, judges[n],
                                          len(others)) for n in CONDITIONS},
        })
    strategies = (["forced_local", "world_leader"]
                  + [f"judge_{n}" for n in CONDITIONS]
                  + [f"fusion_{n}" for n in CONDITIONS])
    results["residual"] = {"n": len(rows)}
    print(f"{'strategy':<16}{'ALL':>8}{'high':>8}{'mid':>8}{'low':>8}")
    for strategy in strategies:
        cells = {}
        for tier in ("ALL",) + TIERS:
            subset = [r for r in rows if tier in ("ALL", r["tier"])]
            judged = [r for r in subset if r[strategy] is not None]
            hits = sum(match(r[strategy], r["gold"]) for r in judged)
            cells[tier] = hits / max(len(judged), 1)
        results["residual"][strategy] = {t: round(v, 4) for t, v in cells.items()}
        print(f"{strategy:<16}" + "".join(f"{100*cells[t]:>7.1f}%"
                                          for t in ("ALL",) + TIERS))
    (OUT_DIR / "stage6_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage6_items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nwrote stage6_results.json and stage6_items.jsonl")


if __name__ == "__main__":
    main()
