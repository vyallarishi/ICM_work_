"""Stage 7 — the diverse-4 configuration, completed with its MP stage.

Config under test (user-proposed):
  view      diverse-4 per pool (repo's label-blind select_diverse logic:
            one chain per answer group, largest groups first, seeded shuffle,
            frequency-weighted fillers when <4 distinct answers)
  commit    global-first + veto on those views (measured: 73.6% cov @ 96.3%)
  residual  -> judge J3 (3 pseudo-labeled demos) and J0 control, fusion pick

PRE-REGISTERED (before this run):
  judge alone ~45-52% overall on the residual, low tier ~30%;
  fusion a few points above judge-alone;
  end-to-end blend ~84-86% (vs random-4 system 88-91%).

Usage: --self-test | --dry-run | full (sbatch).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .rung7_small_pools_and_routing import SEED, TIER, TIERS, load, match, norm
from .stage6_user_spec_config import (
    build_demo_bank,
    build_prompt,
    fusion_pick,
)

MODEL_DIR = "/lustre10/scratch/rvyalla/sm-icm/assets/models/Qwen3-8B-Base"
RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
WORLD_STRENGTH = 0.70
LANDSLIDE = 0.70
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 32
CONDITIONS = ("J0", "J3")


def diverse4_chains(instance_id: str, chains: list[dict]) -> list[dict]:
    """Repo select_diverse logic at chain level (label-blind, deterministic)."""
    if len(chains) <= 4:
        return list(chains)
    rng = random.Random((1729, instance_id, "diverse").__repr__())
    shuffled = list(chains)
    rng.shuffle(shuffled)
    groups: dict[str, list[dict]] = defaultdict(list)
    for chain in shuffled:
        groups[chain["answer"]].append(chain)
    ordered = sorted(groups.values(), key=lambda g: -len(g))
    selected = [group[0] for group in ordered[:4]]
    if len(selected) < 4:
        chosen = {id(c) for c in selected}
        for chain in shuffled:
            if id(chain) not in chosen:
                selected.append(chain)
                chosen.add(id(chain))
                if len(selected) == 4:
                    break
    return selected[:4]


def decide(views_answers: dict[str, list[str]]):
    """Global-first + veto on the given views. Returns {iid: {answer, route}}."""
    plur = defaultdict(dict)
    for instance_id, answers in views_answers.items():
        if answers:
            counts = Counter(answers)
            plur[instance_id.rsplit("_", 1)[-1]]["_".join(instance_id.split("_")[1:-1])] = \
                max(counts, key=lambda a: (counts[a], a))
    out = {}
    for instance_id, answers in views_answers.items():
        if not answers:
            out[instance_id] = {"answer": None, "route": "empty"}
            continue
        counts = Counter(answers)
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        others = [p for l, p in plur[problem].items() if l != lang]
        world = Counter(others).most_common(1) if others else []
        strong = world and world[0][1] / len(others) >= WORLD_STRENGTH
        if strong and world[0][0] in counts:
            out[instance_id] = {"answer": world[0][0], "route": "global"}
        elif counts.most_common(1)[0][1] / len(answers) >= LANDSLIDE:
            top = max(counts, key=lambda a: (counts[a], a))
            if strong and not match(world[0][0], top):
                out[instance_id] = {"answer": None, "route": "vetoed"}
            else:
                out[instance_id] = {"answer": top, "route": "local"}
        else:
            out[instance_id] = {"answer": None, "route": "residual"}
    return out, plur


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- diverse4_chains ---")
    def mk(answers):
        return [{"answer": a, "chain_text": f"c{i}", "question": "q"}
                for i, a in enumerate(answers)]
    pool = mk(["10"] * 8 + ["20"] * 4 + ["30"] * 2 + ["40", "50"])
    view = diverse4_chains("train_en_1", pool)
    check("four chains", len(view), 4)
    check("four distinct answers covered",
          len({c["answer"] for c in view}), 4)
    check("largest groups first",
          {c["answer"] for c in view} >= {"10", "20", "30"}, True)
    check("deterministic",
          [c["answer"] for c in diverse4_chains("train_en_1", pool)],
          [c["answer"] for c in view])
    small = mk(["10", "10", "20"])
    check("small pool passes through", len(diverse4_chains("x", small)), 3)
    two = mk(["10"] * 10 + ["20"] * 6)
    view2 = diverse4_chains("train_en_2", two)
    check("fillers keep frequency signal",
          Counter(c["answer"] for c in view2).most_common(1)[0][1] >= 2, True)

    print("--- decide (global-first + veto on views) ---")
    views = {
        "train_en_1": ["10", "20", "30", "40"],
        "train_de_1": ["10", "10", "10", "10"],
        "train_hi_1": ["10", "10", "10", "10"],
        "train_ru_1": ["10", "10", "10", "10"],
        "train_am_1": ["40", "40", "40", "40"],
    }
    out, _ = decide(views)
    check("flattened split committed by world", out["train_en_1"]["answer"], "10")
    check("route global", out["train_en_1"]["route"], "global")
    check("world-contradicted landslide vetoed", out["train_am_1"]["answer"], None)
    check("veto route", out["train_am_1"]["route"], "vetoed")
    views2 = dict(views)
    views2["train_hi_1"] = ["20", "20", "20", "20"]
    views2["train_ru_1"] = ["30", "30", "30", "30"]
    out2, _ = decide(views2)
    check("world split -> residual", out2["train_en_1"]["route"], "residual")

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
    chain_lookup = defaultdict(list)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(
                path, columns=["instance_id", "canonical_answer", "parse_status",
                               "question", "chain_text"]).to_pylist():
            if row["parse_status"] != "parsed":
                continue
            answer = norm(row["canonical_answer"])
            if answer is not None:
                chain_lookup[row["instance_id"]].append(
                    {"answer": answer, "question": row["question"],
                     "chain_text": row["chain_text"], "iid": row["instance_id"]})

    views = {instance_id: diverse4_chains(instance_id, chains)
             for instance_id, chains in chain_lookup.items()}
    views_answers = {i: [c["answer"] for c in v] for i, v in views.items()}
    decisions, plur = decide(views_answers)

    committed = {"n": 0, "correct": 0}
    per_tier = defaultdict(lambda: [0, 0, 0])
    for instance_id, decision in decisions.items():
        g = gold.get(instance_id)
        if g is None:
            continue
        t = TIER["_".join(instance_id.split("_")[1:-1])]
        per_tier[t][0] += 1
        if decision["answer"] is not None:
            per_tier[t][1] += 1
            per_tier[t][2] += match(decision["answer"], g)
    print("committed tier (diverse-4, global-first+veto):")
    for t in TIERS:
        n, com, cor = per_tier[t]
        print(f"  {t:<5} coverage {100*com/n:5.1f}%  precision {100*cor/max(com,1):5.1f}%")

    residual = [i for i, d in decisions.items()
                if d["route"] in ("residual", "vetoed") and gold.get(i)]
    total_chains = sum(len(views[i]) for i in residual)
    rng = random.Random(SEED + 7)
    demos = build_demo_bank(views_answers, decisions, chain_lookup, rng)
    have3 = sum(1 for d in demos.values() if len(d) == 3)
    print(f"residual pools: {len(residual)}; chains to score: {total_chains} x 8 passes")
    print(f"languages with full 3-demo banks: {have3}/37")

    if args.dry_run:
        sample = residual[0]
        lang = "_".join(sample.split("_")[1:-1])
        chain = views[sample][0]
        print("\n--- sample J3 prompt (first 500 chars) ---")
        print(build_prompt(chain["question"], chain["chain_text"], " Yes", " No",
                           demos.get(lang, []))[:500])
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
            raise ValueError(f"marker {marker!r} not single-token")
        return ids[0]

    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    flat = [(instance_id, k, chain) for instance_id in residual
            for k, chain in enumerate(views[instance_id])]
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
                    views[instance_id][k][key] = margin
                if start % (BATCH_SIZE * 40) == 0:
                    print(f"  {start}/{len(entries)}", flush=True)
                    torch.cuda.empty_cache()
        for instance_id in residual:
            for chain in views[instance_id]:
                normal, swapped = chain.get(name), chain.get(f"{name}_swap")
                chain[name] = ((normal + swapped) / 2
                               if normal is not None and swapped is not None else None)

    print()
    print("=" * 78)
    print("RESIDUAL TIER (diverse-4 config) — strategies on identical pools")
    print("=" * 78)
    rows = []
    for instance_id in residual:
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        local_counts = Counter(views_answers[instance_id])
        others = [p for l, p in plur[problem].items() if l != lang]
        world_share = {a: sum(match(p, a) for p in others) / max(len(others), 1)
                       for a in local_counts}
        judges = {}
        for name in CONDITIONS:
            scored = [c for c in views[instance_id] if c.get(name) is not None]
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
    results = {"residual_n": len(rows)}
    print(f"{'strategy':<16}{'ALL':>8}{'high':>8}{'mid':>8}{'low':>8}")
    for strategy in strategies:
        cells = {}
        for tier in ("ALL",) + TIERS:
            subset = [r for r in rows if tier in ("ALL", r["tier"])]
            judged = [r for r in subset if r[strategy] is not None]
            hits = sum(match(r[strategy], r["gold"]) for r in judged)
            cells[tier] = hits / max(len(judged), 1)
        results[strategy] = {t: round(v, 4) for t, v in cells.items()}
        print(f"{strategy:<16}" + "".join(f"{100*cells[t]:>7.1f}%"
                                          for t in ("ALL",) + TIERS))

    # end-to-end blend
    com_n = sum(per_tier[t][1] for t in TIERS)
    com_ok = sum(per_tier[t][2] for t in TIERS)
    tot = sum(per_tier[t][0] for t in TIERS)
    best = max(strategies, key=lambda s: results[s]["ALL"])
    res_ok = results[best]["ALL"] * len(rows)
    print(f"\nend-to-end @100% coverage (committed + best residual strategy "
          f"[{best}]): {100*(com_ok+res_ok)/tot:.1f}%")
    results["end_to_end_full_coverage"] = round((com_ok + res_ok) / tot, 4)
    (OUT_DIR / "stage7_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage7_items.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote stage7_results.json and stage7_items.jsonl")


if __name__ == "__main__":
    main()
