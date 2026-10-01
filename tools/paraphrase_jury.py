"""Matched version of the paraphrase control: the SAME target pools, three juries.

The paraphrase control (q04/english_paraphrase_control.py) compared two
different populations: English paraphrase pools labelled by other paraphrases
vs 37-language pools labelled by other languages. This script holds the
target fixed (the real 37-language pools on the 142 problems that have
paraphrases) and varies only who votes:

    languages   the other 36 languages                (the record)
    paraphrases the English paraphrase views only
    both        languages + paraphrases

Coverage / precision are reported for the real pools by tier, and for the
English pools alone. CPU, seconds.

    python tools/paraphrase_jury.py [--chains data/.../stage31_chains.jsonl]
"""
import argparse
import json
from collections import Counter, defaultdict

from icm.method import data, vote
from icm.method.metrics import metrics
from icm.method.regenerate import extract_answer
from icm.paths import RUN_DIR

FULL37 = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37"


def build_paraphrase_pools(chains_path, gold):
    """Exactly as english_paraphrase_control.phase_decide builds them."""
    para_chains = defaultdict(list)
    for line in open(chains_path, encoding="utf-8"):
        r = json.loads(line)
        ans = extract_answer(r["text"])
        if ans is not None and r["text"].strip():
            para_chains[(r["problem"], r["view"])].append({"answer": ans, "text": r["text"]})
    pools = {}
    for (prob, view), chains in para_chains.items():
        if not chains or prob not in gold:
            continue
        iid = f"train_v{view:02d}_{prob}"
        view_chains = data.diverse8_view(iid, chains)
        counts = Counter(c["answer"] for c in view_chains)
        pools[iid] = {"gold": gold[prob], "lang": f"v{view:02d}", "question": "",
                      "chains": view_chains,
                      "share": counts.most_common(1)[0][1] / len(view_chains),
                      "plurality": max(counts, key=lambda a: (counts[a], a))}
    return pools


def score(real, state, contested, label):
    out = {}
    groups = {"ALL": set(real)}
    for tier in ("high", "mid", "low"):
        groups[tier] = {i for i, p in real.items() if data.TIER[p["lang"]] == tier}
    groups["en"] = {i for i, p in real.items() if p["lang"] == "en"}
    for g, ids in groups.items():
        m = metrics({i: real[i] for i in ids}, {i: state[i] for i in ids},
                    [i for i in contested if i in ids])
        out[g] = m
    print(f"{label:<12}" + "".join(f"  {g}: cov {100*out[g]['coverage']:5.1f} prec {100*out[g]['precision']:5.1f}"
                                  for g in ("ALL", "high", "mid", "low", "en")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", default=str(RUN_DIR / "stage31_chains.jsonl"))
    ap.add_argument("--out", default=str(RUN_DIR / "paraphrase_jury.json"))
    a = ap.parse_args()

    real_all = data.load(FULL37)
    gold = {i.rsplit("_", 1)[-1]: p["gold"] for i, p in real_all.items() if p["lang"] == "en"}
    para = build_paraphrase_pools(a.chains, gold)
    problems = {i.rsplit("_", 1)[-1] for i in para}
    real = {i: p for i, p in real_all.items() if i.rsplit("_", 1)[-1] in problems}
    print(f"real pools {len(real)} ({len({p['lang'] for p in real.values()})} languages) on "
          f"{len(problems)} problems; paraphrase pools {len(para)} "
          f"(views per problem: median {sorted(Counter(i.rsplit('_', 1)[-1] for i in para).values())[len(problems)//2]})\n")

    results = {}
    st, ct, _ = vote.converge(real)
    results["languages"] = score(real, st, ct, "languages")

    st_b, ct_b = {}, []
    for lang in sorted({p["lang"] for p in real.values()}):
        mine = {i: p for i, p in real.items() if p["lang"] == lang}
        s, c, _ = vote.converge({**mine, **para})
        st_b.update({i: s[i] for i in mine}); ct_b += [i for i in c if i in mine]
    results["paraphrases"] = score(real, st_b, ct_b, "paraphrases")

    s, c, _ = vote.converge({**real, **para})
    results["both"] = score(real, {i: s[i] for i in real}, [i for i in c if i in real], "both")

    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
