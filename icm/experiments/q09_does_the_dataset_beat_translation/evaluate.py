"""Q09 step 3 (GPU): accuracy of the base model and each fine-tuned arm on the test split.

One vLLM engine, the base model plus each LoRA adapter, IDENTICAL prompts and
decoding for all: greedy, the training prompt format, stop at the next
"Question:". The answer is parsed with the generation pipeline's own parser
and compared to the test-split gold EXACTLY (no tolerance).

    python -m icm.experiments.q09_does_the_dataset_beat_translation.evaluate \
        --adapter translated=<dir> --adapter method=<dir> [--limit-per-lang N]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ...paths import RUN_DIR

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
GEN_SCRIPTS = Path(__file__).resolve().parents[3] / "generation" / "scripts"


def paired_comparisons(results, rows, reference="translated", n_boot=10000, seed=0):
    """Accuracy difference of every arm vs the reference arm on the SAME prompts,
    with a paired bootstrap over prompts (95% CI), overall and per tier.
    PRE-REGISTERED H4: method_fusion - translated >= +1.0 point overall with the
    CI lower bound > 0, and method_lenmatch - translated CI lower bound > 0."""
    import numpy as np
    if reference not in results["arms"]:
        return {}
    rng = np.random.default_rng(seed)
    tiers = np.array([r["tier"] for r in rows])
    ref = np.array(results["arms"][reference]["correct"], dtype=float)
    out = {}
    print(f"\npaired difference vs {reference} (points, 95% bootstrap CI over the same prompts):")
    for name, arm in results["arms"].items():
        if name == reference:
            continue
        d = np.array(arm["correct"], dtype=float) - ref
        out[name] = {}
        for sl in ("ALL", "high", "mid", "low"):
            m = np.ones(len(d), bool) if sl == "ALL" else tiers == sl
            dd = d[m]
            boot = dd[rng.integers(0, len(dd), size=(n_boot, len(dd)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out[name][sl] = [round(100 * float(dd.mean()), 2), round(100 * float(lo), 2), round(100 * float(hi), 2)]
        print(f"  {name:<16}" + "  ".join(f"{sl} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for sl, v in out[name].items()))
    if "method_fusion" in out and "method_lenmatch" in out:
        f, l = out["method_fusion"]["ALL"], out["method_lenmatch"]["ALL"]
        out["H4_holds"] = f[0] >= 1.0 and f[1] > 0 and l[1] > 0
        print(f"H4 (fusion >= +1.0 with CI > 0, lenmatch CI > 0): {'HOLDS' if out['H4_holds'] else 'FAILS'}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--eval", default=str(RUN_DIR / "q09_eval.jsonl"))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--limit-per-lang", type=int, default=None, help="smoke test")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default=str(RUN_DIR / "q09_results.json"))
    a = ap.parse_args()

    sys.path.insert(0, str(GEN_SCRIPTS))
    from answer_parsing import extract_prediction_answer, normalize_number
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
    if a.limit_per_lang:
        seen = defaultdict(int); keep = []
        for r in rows:
            if seen[r["lang"]] < a.limit_per_lang:
                keep.append(r); seen[r["lang"]] += 1
        rows = keep
    prompts = [r["prompt"] for r in rows]
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=["\nQuestion:", "Question:"])
    llm = LLM(model=a.model_dir, dtype="bfloat16", max_model_len=4096, enable_lora=bool(a.adapter),
              max_lora_rank=a.rank, max_loras=1, seed=0, gpu_memory_utilization=0.9)

    arms = [("base", None)] + [tuple(s.split("=", 1)) for s in a.adapter]
    results = {"n_prompts": len(rows), "arms": {}}
    for i, (name, path) in enumerate(arms):
        lora = LoRARequest(name, i + 1, path) if path else None
        outs = llm.generate(prompts, params, lora_request=lora)
        by = defaultdict(lambda: [0, 0])
        samples, correct = [], []
        for r, o in zip(rows, outs):
            pred = extract_prediction_answer(o.outputs[0].text)[1]
            gold = normalize_number(r["gold"])
            ok = pred is not None and gold is not None and pred == gold
            correct.append(int(ok))
            for key in ("ALL", r["tier"], r["lang"]):
                by[key][0] += ok; by[key][1] += 1
            if len(samples) < 3:
                samples.append({"id": r["id"], "gold": r["gold"], "pred": str(pred), "text": o.outputs[0].text[:400]})
        results["arms"][name] = {"accuracy": {k: round(v[0] / v[1], 4) for k, v in by.items()},
                                 "n": {k: v[1] for k, v in by.items()}, "samples": samples, "correct": correct}
        print(f"{name:<12} ALL {by['ALL'][0] / by['ALL'][1]:.4f}  "
              + "  ".join(f"{t} {by[t][0] / max(by[t][1], 1):.4f}" for t in ("high", "mid", "low")), flush=True)
    langs = sorted(k for k in results["arms"]["base"]["accuracy"] if k not in ("ALL", "high", "mid", "low"))
    print(f"\n{'lang':<7}" + "".join(f"{n:>12}" for n, _ in arms))
    for l in langs:
        print(f"{l:<7}" + "".join(f"{results['arms'][n]['accuracy'][l]:>12.3f}" for n, _ in arms))
    results["paired"] = paired_comparisons(results, rows)
    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
