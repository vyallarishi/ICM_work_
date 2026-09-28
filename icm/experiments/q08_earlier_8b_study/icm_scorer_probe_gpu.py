"""E3 — scorer probe (GPU): can the mutual-predictability scorer be made to speak?

E0 verdict on the production scorer: per-variant margins are dominated by a
marker-preference bias ~5x the content component, and the debiased content
signal has AUC ~0.50 against gold at chain level. This probe rescores a
stratified sample of chains under measurement variations ONLY (the ICM
principle — score a chain's plausibility under labeled demonstrations — is
unchanged):

  A  repro        bf16, " A"/" B" markers, 8 pseudo-labeled demos (production setup)
  B  fp32         same as A, model in float32 (kills bf16 logit quantization)
  C  yes/no       bf16, " Yes"/" No" markers (semantically meaningful verdict tokens)
  D  instructed   bf16, " A"/" B", 1-line instruction + NO demos (is the demo
                  context helping or hurting?)

Every condition scores each chain twice (marker roles swapped) and reports the
debiased content margin = mean of the two role-consistent margins. Demos are
pseudo-labeled by pool plurality (label-blind); demo problems are excluded
from evaluation. Post-hoc AUC vs gold per language/condition.

Run via e3_probe.sbatch (1x H100). Writes e3_results.json + e3_chains.jsonl.gz.
"""

from __future__ import annotations

import glob
import gzip
import json
import random
import statistics as st
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pyarrow.parquet as pq

RUN = "/lustre10/scratch/rvyalla/sm-icm/runs/qwen3-8b-mono-heldout200-s200"
GEN = ("/lustre10/scratch/rvyalla/multilingual-icm/results/"
       "qwen3-8b-native-6shot-main/final/pooled/generated")
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
# Local snapshot (same one the SM-ICM inference jobs load; already verified
# against revision 49e3418f by the pipeline's preflight).
MODEL_DIR = "/lustre10/scratch/rvyalla/sm-icm/assets/models/Qwen3-8B-Base"

LANGS = {"en": "high", "de": "high", "ru": "mid", "hi": "mid",
         "am": "low", "sw": "low"}
N_DEMO_POS = 4
N_DEMO_NEG = 4
DEMO_MAX_CHARS = 700
CHAIN_MAX_CHARS = 1800
MAX_PROMPT_TOKENS = 6000
BATCH_SIZE = 16
SEED = 20260806


def norm(x) -> str | None:
    try:
        d = Decimal(str(x).replace(",", ""))
        return str(d.normalize()) if d.is_finite() else None
    except (InvalidOperation, TypeError):
        return None


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    ranked = [(v, 0) for v in neg] + [(v, 1) for v in pos]
    ranked.sort(key=lambda t: t[0])
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop][0] == ranked[index][0]:
            stop += 1
        rank_sum += (index + stop + 1) / 2 * sum(f for _, f in ranked[index:stop])
        index = stop
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def block(question: str, chain: str, marker: str | None = None) -> str:
    text = "[Q]\n" + question.strip() + "\n[R]\n" + chain.strip() + "\n[V]"
    return text + marker if marker is not None else text


def load_data():
    gold: dict[str, str] = {}
    for lang in LANGS:
        with open(f"{GEN}/{lang}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                gold[row["id"]] = norm(row.get("gold_number"))

    cols = ["instance_id", "canonical_answer", "parse_status", "question",
            "chain_text"]
    pools: dict[str, list[dict]] = defaultdict(list)
    for path in glob.glob(f"{RUN}/export/**/broad.parquet", recursive=True):
        for row in pq.read_table(path, columns=cols).to_pylist():
            lang = "_".join(row["instance_id"].split("_")[1:-1])
            answer = norm(row["canonical_answer"])
            if lang not in LANGS or row["parse_status"] != "parsed" or answer is None:
                continue
            row["answer"] = answer
            pools[row["instance_id"]].append(row)

    rng = random.Random(SEED)
    demos: dict[str, list[tuple[str, str, bool]]] = {}
    demo_problems: dict[str, set[str]] = defaultdict(set)
    for lang in LANGS:
        positives, negatives = [], []
        instance_ids = sorted(i for i in pools if f"_{lang}_" in i)
        rng.shuffle(instance_ids)
        for instance_id in instance_ids:
            pool = pools[instance_id]
            counts = Counter(c["answer"] for c in pool)
            top, top_n = counts.most_common(1)[0]
            if top_n / len(pool) < 0.8 or len(pool) < 8:
                continue  # only landslide pools give trustworthy pseudo-labels
            problem = instance_id.rsplit("_", 1)[-1]
            pos = [c for c in pool if c["answer"] == top
                   and len(c["chain_text"]) <= DEMO_MAX_CHARS]
            neg = [c for c in pool if c["answer"] != top
                   and len(c["chain_text"]) <= DEMO_MAX_CHARS]
            took = False
            if pos and len(positives) < N_DEMO_POS:
                positives.append((pos[0]["question"], pos[0]["chain_text"], True))
                took = True
            if neg and len(negatives) < N_DEMO_NEG:
                negatives.append((neg[0]["question"], neg[0]["chain_text"], False))
                took = True
            if took:
                demo_problems[lang].add(problem)
            if len(positives) >= N_DEMO_POS and len(negatives) >= N_DEMO_NEG:
                break
        merged = positives + negatives
        rng.shuffle(merged)
        demos[lang] = merged
        print(f"{lang}: {len(positives)} positive / {len(negatives)} negative demos "
              f"from {len(demo_problems[lang])} problems")

    chains = []
    for instance_id, pool in pools.items():
        lang = "_".join(instance_id.split("_")[1:-1])
        problem = instance_id.rsplit("_", 1)[-1]
        g = gold.get(instance_id)
        if g is None or problem in demo_problems[lang]:
            continue
        usable = [c for c in pool if len(c["chain_text"]) <= CHAIN_MAX_CHARS]
        gold_chains = [c for c in usable if c["answer"] == g]
        rest = [c for c in usable if c["answer"] != g]
        rng.shuffle(gold_chains)
        rng.shuffle(rest)
        for c in gold_chains[:2] + rest[:2]:
            chains.append({"instance_id": instance_id, "lang": lang,
                           "question": c["question"],
                           "chain_text": c["chain_text"],
                           "is_gold": c["answer"] == g})
    print(f"eval chains: {len(chains)} "
          f"({sum(c['is_gold'] for c in chains)} gold / "
          f"{sum(not c['is_gold'] for c in chains)} non-gold)")
    return demos, chains


def build_prompt(condition: str, lang_demos, chain, positive: str, negative: str) -> str:
    if condition == "D_instructed":
        preamble = ("<ICM>\nMark [V] with" + positive +
                    " if the reasoning is correct and" + negative +
                    " if it is incorrect.\n\n")
        return preamble + block(chain["question"], chain["chain_text"])
    blocks = [block(q, r, positive if label else negative)
              for q, r, label in lang_demos]
    return "<ICM>\n" + "\n\n".join(blocks + [block(chain["question"], chain["chain_text"])])


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    demos, chains = load_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def marker_id(marker: str) -> int:
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"marker {marker!r} is not a single token: {ids}")
        return ids[0]

    conditions = [
        ("A_repro", "bf16", " A", " B"),
        ("B_fp32", "fp32", " A", " B"),
        ("C_yesno", "bf16", " Yes", " No"),
        ("D_instructed", "bf16", " A", " B"),
    ]
    for c in chains:
        c["margins"] = {}

    def run_conditions(dtype_name: str, model) -> None:
        import inspect

        # Only materialize logits for the final position (repo's trick); the
        # full [batch, seq, vocab] tensor is ~29 GB and OOMs the GPU.
        forward_params = inspect.signature(model.forward).parameters
        last_logit_kwargs = {"logits_to_keep": 1} if "logits_to_keep" in forward_params else {}
        for name, dtype, positive, negative in conditions:
            if dtype != dtype_name:
                continue
            pos_id, neg_id = marker_id(positive), marker_id(negative)
            for role_swap in (False, True):
                # role_swap: the demo labels use the swapped marker as "correct"
                pos_marker, neg_marker = (negative, positive) if role_swap else (positive, negative)
                pos_tok, neg_tok = (neg_id, pos_id) if role_swap else (pos_id, neg_id)
                entries = []
                for index, chain in enumerate(chains):
                    prompt = build_prompt(name, demos[chain["lang"]], chain,
                                          pos_marker, neg_marker)
                    n_tokens = len(tokenizer.encode(prompt))
                    if n_tokens > MAX_PROMPT_TOKENS:
                        continue
                    entries.append((n_tokens, index, prompt))
                entries.sort(key=lambda e: e[0])  # length-sorted batches -> minimal padding
                key = f"{name}{'_swap' if role_swap else ''}"
                print(f"[{key}] scoring {len(entries)} chains", flush=True)
                batch_size = BATCH_SIZE // 2 if dtype_name == "fp32" else BATCH_SIZE
                for start in range(0, len(entries), batch_size):
                    batch = entries[start:start + batch_size]
                    enc = tokenizer([prompt for _, _, prompt in batch],
                                    return_tensors="pt", padding=True,
                                    pad_to_multiple_of=256).to(model.device)
                    with torch.inference_mode():
                        logits = model(**enc, use_cache=False, **last_logit_kwargs).logits
                        logits = logits[:, -1, :].float()
                        log_probs = torch.log_softmax(logits, dim=-1)
                    for row, (_, index, _) in enumerate(batch):
                        margin = float(log_probs[row, pos_tok] - log_probs[row, neg_tok])
                        chains[index]["margins"][key] = margin
                    if start % (batch_size * 50) == 0:
                        print(f"  {start}/{len(entries)}", flush=True)
                        torch.cuda.empty_cache()
                # Nothing is ever lost twice: persist all margins after every pass.
                snapshot = {c["instance_id"] + "|" + str(idx): c["margins"]
                            for idx, c in enumerate(chains)}
                (OUT_DIR / "e3_margins_partial.json").write_text(json.dumps(snapshot))

    for dtype_name, torch_dtype in (("bf16", "bfloat16"), ("fp32", "float32")):
        import torch
        print(f"loading model in {dtype_name}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=getattr(torch, torch_dtype), device_map="cuda",
        )
        model.eval()
        run_conditions(dtype_name, model)
        del model
        torch.cuda.empty_cache()

    # ---- analysis -------------------------------------------------------
    results: dict = {}
    print("\n" + "=" * 78)
    print("E3 RESULTS — debiased content margin AUC (gold vs non-gold chains)")
    print("=" * 78)
    header = f"{'condition':<14}" + "".join(f"{lang:>9}" for lang in LANGS) + f"{'ALL':>9}{'zero%':>8}"
    print(header)
    for name, _, _, _ in conditions:
        per_lang = {}
        all_gold, all_rest, zeros, total = [], [], 0, 0
        for lang in LANGS:
            gold_margins, rest_margins = [], []
            for c in chains:
                if c["lang"] != lang:
                    continue
                normal = c["margins"].get(name)
                swapped = c["margins"].get(f"{name}_swap")
                if normal is None or swapped is None:
                    continue
                content = (normal + swapped) / 2
                total += 1
                zeros += content == 0.0
                (gold_margins if c["is_gold"] else rest_margins).append(content)
            score = auc(gold_margins, rest_margins)
            per_lang[lang] = round(score, 3) if score is not None else None
            all_gold += gold_margins
            all_rest += rest_margins
        overall = auc(all_gold, all_rest)
        results[name] = {"per_lang": per_lang,
                         "overall_auc": round(overall, 3) if overall else None,
                         "zero_margin_pct": round(100 * zeros / max(total, 1), 1)}
        row = f"{name:<14}" + "".join(
            f"{per_lang[lang] if per_lang[lang] is not None else '—':>9}" for lang in LANGS)
        print(row + f"{overall:>9.3f}{100 * zeros / max(total, 1):>7.1f}%")

    (OUT_DIR / "e3_results.json").write_text(json.dumps(results, indent=2))
    with gzip.open(OUT_DIR / "e3_chains.jsonl.gz", "wt", encoding="utf-8") as handle:
        for c in chains:
            handle.write(json.dumps({k: v for k, v in c.items()
                                     if k != "chain_text"}, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT_DIR / 'e3_results.json'} and e3_chains.jsonl.gz")


if __name__ == "__main__":
    main()
