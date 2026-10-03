"""Q09 step 4 (GPU): re-run the evaluation SAVING every prediction, then check
that the arm differences are not artefacts of answer formatting.

Same engine, prompts, decoding and parser as evaluate.py (greedy, stop at the
next "Question:", max 1,024 tokens), so accuracies must reproduce q09_results.json.
Writes q09_predictions.jsonl (one row per prompt x arm: text, parsed answer,
tokens, finish reason) and q09_diagnostics.json.

PER ARM AND TIER:
  parsed        the strict parser found an answer
  has_answer_line  the output contains a "#### N" line
  truncated     generation hit the token cap
  acc_strict    accuracy as in evaluate.py
  acc_parsed    accuracy among parsed outputs (is the gap in reasoning or in parsing?)
  acc_lenient   strict, else the LAST number in the output (a format-blind score)
  latin_share   in non-Latin-script languages, share of letters that are Latin
                (reasoning drifting into English)
  out_tokens    mean output length
PAIRED (seed-averaged, bootstrap CI), under strict AND lenient scoring:
  method - translated, own_unlabelled - translated, method - own_unlabelled,
  and with --adapter gold_filtered_*: method - gold_filtered, gold_filtered - translated.
READING: if method - translated survives lenient scoring at about the same size,
the gain is not a formatting artefact.

    python -m icm.experiments.q09_does_the_dataset_beat_translation.diagnose_predictions \
        --adapter translated_s1=<dir> ... [--limit-per-lang N] | --self-test
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from ...paths import RUN_DIR
from .evaluate import GEN_SCRIPTS, MODEL_DIR

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
NON_LATIN = {"am", "ar", "bg", "bn", "el", "fa", "hi", "kk", "kn", "ko", "ky", "ml", "mr", "ru", "th", "uk",
             "ur", "zh-tw"}
COMPARISONS = (("method", "translated"), ("own_unlabelled", "translated"), ("method", "own_unlabelled"),
               ("method_lenmatch", "translated"), ("method", "gold_filtered"), ("gold_filtered", "translated"))


def last_number(text):
    m = NUMBER.findall(text or "")
    return m[-1].replace(",", "") if m else None


def latin_share(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("LATIN" in unicodedata.name(ch, "") for ch in letters) / len(letters)


def paired(scores, rows, n_boot=10000, seed=0):
    """scores: {adapter_name: [0/1 per prompt]} -> {comparison: {slice: [mean, lo, hi]}} in points."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tiers = np.array([r["tier"] for r in rows])
    arms = defaultdict(list)
    for name, s in scores.items():
        if name != "base":
            arms[re.sub(r"_s\d+$", "", name)].append(np.array(s, dtype=float))
    arms = {k: np.mean(v, axis=0) for k, v in arms.items()}
    out = {}
    for a, b in COMPARISONS:
        if a in arms and b in arms:
            d, res = arms[a] - arms[b], {}
            for sl in ("ALL", "high", "mid", "low"):
                dd = d if sl == "ALL" else d[tiers == sl]
                boot = dd[rng.integers(0, len(dd), size=(n_boot, len(dd)))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                res[sl] = [round(100 * float(dd.mean()), 2), round(100 * float(lo), 2), round(100 * float(hi), 2)]
            out[f"{a} - {b}"] = res
    return out


def summarise(preds, rows, normalize_number):
    """preds: {arm: [row dicts]} -> per-arm, per-tier diagnostics, and lenient 0/1 scores."""
    diag, lenient = {}, {}
    for arm, ps in preds.items():
        by = defaultdict(lambda: defaultdict(float)); n = defaultdict(int); nl = defaultdict(int)
        lenient[arm] = []
        for r, p in zip(rows, ps):
            gold = normalize_number(r["gold"])
            strict = p["pred"] is not None and p["pred"] == gold
            fallback = p["pred"] if p["pred"] is not None else normalize_number(last_number(p["text"]))
            len_ok = fallback is not None and fallback == gold
            lenient[arm].append(int(len_ok))
            for key in ("ALL", r["tier"]):
                d = by[key]; n[key] += 1
                d["parsed"] += p["pred"] is not None
                d["has_answer_line"] += "####" in p["text"]
                d["truncated"] += p["finish"] == "length"
                d["acc_strict"] += strict
                d["acc_lenient"] += len_ok
                d["correct_parsed"] += strict
                d["out_tokens"] += p["tokens"]
                if r["lang"] in NON_LATIN:
                    d["latin_share"] += latin_share(p["text"]); nl[key] += 1
        diag[arm] = {}
        for key, d in by.items():
            parsed = d["parsed"]
            diag[arm][key] = {"parsed": round(parsed / n[key], 4), "has_answer_line": round(d["has_answer_line"] / n[key], 4),
                              "truncated": round(d["truncated"] / n[key], 4), "acc_strict": round(d["acc_strict"] / n[key], 4),
                              "acc_parsed": round(d["correct_parsed"] / max(parsed, 1), 4),
                              "acc_lenient": round(d["acc_lenient"] / n[key], 4),
                              "latin_share_nonlatin_langs": round(d["latin_share"] / max(nl[key], 1), 4),
                              "out_tokens": round(d["out_tokens"] / n[key], 1)}
    return diag, lenient


def self_test():
    ok = True
    def check(name, got, want):
        nonlocal ok
        if got != want:
            ok = False
            print(f"FAIL {name}: {got!r} vs {want!r}")
    check("last number", last_number("so 3 + 4 = 7, total 1,080 eggs"), "1080")
    check("no number", last_number("none"), None)
    check("latin share", latin_share("abcअआ"), 0.6)
    norm = lambda x: None if x is None else str(x)
    rows = [{"gold": "7", "tier": "low", "lang": "hi"}, {"gold": "5", "tier": "low", "lang": "hi"}]
    preds = {"a_s1": [{"pred": None, "text": "answer is 7", "finish": "stop", "tokens": 3},
                      {"pred": "5", "text": "#### 5", "finish": "stop", "tokens": 2}]}
    diag, len_ = summarise(preds, rows, norm)
    check("strict", diag["a_s1"]["ALL"]["acc_strict"], 0.5)
    check("lenient recovers the unparsed one", len_["a_s1"], [1, 1])
    check("parsed rate", diag["a_s1"]["ALL"]["parsed"], 0.5)
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TEST FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--eval", default=str(RUN_DIR / "q09_eval.jsonl"))
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--limit-per-lang", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    if not self_test():
        sys.exit(1)
    if a.self_test:
        return

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
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, stop=["\nQuestion:", "Question:"])
    llm = LLM(model=a.model_dir, dtype="bfloat16", max_model_len=4096, enable_lora=bool(a.adapter),
              max_lora_rank=a.rank, max_loras=1, seed=0, gpu_memory_utilization=0.9)
    arms = [("base", None)] + [tuple(s.split("=", 1)) for s in a.adapter]
    preds, strict = {}, {}
    out_path = RUN_DIR / f"q09_predictions{a.tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as h:
        for i, (name, path) in enumerate(arms):
            lora = LoRARequest(name, i + 1, path) if path else None
            outs = llm.generate([r["prompt"] for r in rows], params, lora_request=lora)
            preds[name], strict[name] = [], []
            for r, o in zip(rows, outs):
                c = o.outputs[0]
                p = {"pred": extract_prediction_answer(c.text)[1], "text": c.text,
                     "finish": c.finish_reason, "tokens": len(c.token_ids)}
                preds[name].append(p)
                strict[name].append(int(p["pred"] is not None and p["pred"] == normalize_number(r["gold"])))
                h.write(json.dumps({"arm": name, "id": r["id"], "lang": r["lang"], "gold": r["gold"], **p},
                                   ensure_ascii=False, default=str) + "\n")   # pred is a Decimal
            print(f"{name:<22} strict {sum(strict[name]) / len(rows):.4f}", flush=True)

    diag, lenient = summarise(preds, rows, normalize_number)
    res = {"n_prompts": len(rows), "diagnostics": diag,
           "paired_strict": paired(strict, rows), "paired_lenient": paired(lenient, rows)}
    cols = ("parsed", "has_answer_line", "truncated", "acc_strict", "acc_parsed", "acc_lenient",
            "latin_share_nonlatin_langs", "out_tokens")
    print(f"\n{'arm':<22}{'tier':<6}" + "".join(f"{c[:12]:>13}" for c in cols))
    for arm in diag:
        for t in ("ALL", "high", "mid", "low"):
            print(f"{arm:<22}{t:<6}" + "".join(f"{diag[arm][t][c]:>13}" for c in cols))
    for kind in ("paired_strict", "paired_lenient"):
        print(f"\n{kind} (seed-averaged), points [95% CI]:")
        for k, v in res[kind].items():
            print(f"  {k:<30}" + "  ".join(f"{sl} {x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]" for sl, x in v.items()))
    Path(RUN_DIR / f"q09_diagnostics{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path} and {RUN_DIR / f'q09_diagnostics{a.tag}.json'}")


if __name__ == "__main__":
    main()
