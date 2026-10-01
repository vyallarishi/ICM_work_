"""Per-language health of a generation run, before scaling (CPU, seconds).

For every language in <root>/final/pooled/generated/, including languages the
tier table does not know yet:

    items, chains, parse-usable %, truncated %, raw chain accuracy,
    landslide %, and coverage / precision after cross-lingual voting
    (icm.method.vote, no regeneration)

Gold is read only for the accuracy and precision columns.

    python tools/pilot_report.py --root /scratch/.../qwen3-14b-pilot13 [--out report.json]
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

from icm.method import data, vote
from icm.method.metrics import metrics


def raw_stats(path):
    """Counts straight from the generated file, before any filtering."""
    items = chains = usable = truncated = correct = 0
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        gold = data.norm(row.get("gold_number"))
        items += 1
        for c in row.get("candidates", []):
            chains += 1
            if c.get("finish_reason") == "length":
                truncated += 1
            a = data.norm(c.get("pred_number"))
            if a is not None and (c.get("text") or "").strip():
                usable += 1
                correct += bool(gold is not None and data.match(a, gold))
    return {"items": items, "chains": chains, "usable_pct": 100 * usable / max(chains, 1),
            "truncated_pct": 100 * truncated / max(chains, 1),
            "chain_accuracy_pct": 100 * correct / max(usable, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    langs = sorted(Path(p).stem for p in glob.glob(f"{a.root}/final/pooled/generated/*.jsonl"))
    for lang in langs:                      # let load_pools accept every language present
        data.TIER.setdefault(lang, "new")
    pools = data.load(a.root)
    state, contested, _ = vote.converge(pools)

    rows = {}
    for lang in langs:
        r = raw_stats(f"{a.root}/final/pooled/generated/{lang}.jsonl")
        mine = {i: p for i, p in pools.items() if p["lang"] == lang}
        if mine:
            sub_state = {i: state[i] for i in mine}
            m = metrics(mine, sub_state, [i for i in contested if i in mine])
            r.update({"pools": len(mine),
                      "landslide_pct": 100 * sum(p["share"] >= data.LANDSLIDE for p in mine.values()) / len(mine),
                      "coverage_pct": 100 * m["coverage"], "precision_pct": 100 * m["precision"],
                      "abstained": sum(sub_state[i]["answer"] is None for i in mine)})
        rows[lang] = r

    cols = ["items", "chains", "usable_pct", "truncated_pct", "chain_accuracy_pct",
            "landslide_pct", "coverage_pct", "precision_pct", "abstained"]
    print(f"{'lang':<7}" + "".join(f"{c:>19}" for c in cols))
    for lang, r in rows.items():
        print(f"{lang:<7}" + "".join(f"{r.get(c, float('nan')):>19.1f}" if isinstance(r.get(c), float)
                                     else f"{r.get(c, '-'):>19}" for c in cols))
    print("\nread: a language with low chain accuracy and high abstention has little to vote with;"
          " a high truncated % means max_new_tokens is cutting it off.")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
