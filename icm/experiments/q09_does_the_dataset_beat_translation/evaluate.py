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
        samples = []
        for r, o in zip(rows, outs):
            pred = extract_prediction_answer(o.outputs[0].text)[1]
            gold = normalize_number(r["gold"])
            ok = pred is not None and gold is not None and pred == gold
            for key in ("ALL", r["tier"], r["lang"]):
                by[key][0] += ok; by[key][1] += 1
            if len(samples) < 3:
                samples.append({"id": r["id"], "gold": r["gold"], "pred": str(pred), "text": o.outputs[0].text[:400]})
        results["arms"][name] = {"accuracy": {k: round(v[0] / v[1], 4) for k, v in by.items()},
                                 "n": {k: v[1] for k, v in by.items()}, "samples": samples}
        print(f"{name:<12} ALL {by['ALL'][0] / by['ALL'][1]:.4f}  "
              + "  ".join(f"{t} {by[t][0] / max(by[t][1], 1):.4f}" for t in ("high", "mid", "low")), flush=True)
    langs = sorted(k for k in results["arms"]["base"]["accuracy"] if k not in ("ALL", "high", "mid", "low"))
    print(f"\n{'lang':<7}" + "".join(f"{n:>12}" for n, _ in arms))
    for l in langs:
        print(f"{l:<7}" + "".join(f"{results['arms'][n]['accuracy'][l]:>12.3f}" for n, _ in arms))
    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
