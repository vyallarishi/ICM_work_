"""Label-free per-language statistics from the 2-shot records (PREREG_native_vs_sft.md definitions):
  A      cross-lingual agreement: share of problems where the language's plurality matches the majority of the
         other languages' pluralities in the same record (match with 0.5% tolerance; no gold)
  S      self-consistency: mean within-pool plurality share
  plur_acc  plurality accuracy against gold (reference only; not used by the rule)
Records: full66 (67 languages, jury = other 66) and full37 (37, jury = other 36). For full66 also A with the
jury restricted to the core 25 (reproduces existing_data_analyses D_agreement_bias 'estimated' as a check).
Read-only; CPU job (agreement.sbatch). Output: agreement.json.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/rvyalla/multilingual_icm/ICM_work_")
from icm.method.data import match, norm  # noqa: E402

RES = Path("/scratch/rvyalla/multilingual-icm/results")
F = Path("/home/rvyalla/multilingual_icm/data/findings")
OUT = Path(__file__).resolve().parent / "agreement.json"
RECORDS = {"full66": RES / "qwen3-14b-full66-7473", "full37": RES / "qwen3-14b-full37-1500"}


def pluralities(path):
    """{original_id: (plurality, share)} and {original_id: gold} over usable chains."""
    out, gold = {}, {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        answers = [norm(c.get("pred_number")) for c in r.get("candidates", [])
                   if (c.get("text") or "").strip() and c.get("finish_reason") != "length"]
        answers = [a for a in answers if a is not None]
        if not answers:
            continue
        cnt = Counter(answers)
        top = max(cnt, key=lambda a: (cnt[a], a))
        out[int(r["original_id"])] = (top, cnt[top] / len(answers))
        gold[int(r["original_id"])] = norm(r.get("gold_number"))
    return out, gold


def majority(answers):
    """Majority of a list of normalised answers, with tolerance-aware grouping; None if empty."""
    groups = []  # [representative, count]
    for a in answers:
        for g in groups:
            if match(a, g[0]):
                g[1] += 1
                break
        else:
            groups.append([a, 1])
    return max(groups, key=lambda g: (g[1], g[0]))[0] if groups else None


def agreement(plur, langs, jury):
    """A(L) for each L in langs, jury = the other languages in `jury`."""
    A = {}
    problems = set.union(*(set(plur[l]) for l in langs))
    for L in langs:
        hits = n = 0
        for p in problems:
            if p not in plur[L]:
                continue
            others = [plur[J][p][0] for J in jury if J != L and p in plur[J]]
            if len(others) < 2:
                continue
            n += 1
            hits += match(plur[L][p][0], majority(others))
        A[L] = hits / n if n else None
    return A


def main():
    sel = json.load(open(F / "results/language_selection.json"))
    core25 = ["en"] + [l["code"] for v in sel["core"]["tiers"].values() for l in v["langs"]]
    result = {}
    for name, root in RECORDS.items():
        plur, gold = {}, {}
        for path in sorted((root / "final/pooled/generated").glob("*.jsonl")):
            plur[path.stem], gold[path.stem] = pluralities(path)
            print(f"{name}/{path.stem}: {len(plur[path.stem])} pools", flush=True)
        langs = sorted(plur)
        A = agreement(plur, langs, langs)
        A25 = agreement(plur, [l for l in langs if l in core25], core25) if name == "full66" else {}
        result[name] = {}
        for L in langs:
            S = sum(s for _, s in plur[L].values()) / len(plur[L])
            acc = sum(match(a, gold[L][p]) for p, (a, _) in plur[L].items()) / len(plur[L])
            result[name][L] = {"A": A[L], "S": S, "plur_acc": acc, "n_pools": len(plur[L])}
            if L in A25:
                result[name][L]["A_core25_jury"] = A25[L]
        print(name, "done", flush=True)
    json.dump(result, open(OUT, "w"), indent=1)
    # check against the existing 25-jury estimate
    est = json.load(open(F / "results/existing_data_analyses.json"))["D_agreement_bias"]["per_language"]
    diffs = [abs(result["full66"][l]["A_core25_jury"] - est[l]["estimated"]) for l in est if l in result["full66"]]
    print(f"check vs D_agreement_bias.estimated: max |diff| = {max(diffs):.4f} over {len(diffs)} languages")


if __name__ == "__main__":
    main()
