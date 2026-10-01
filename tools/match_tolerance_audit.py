"""How much does the 0.5% answer tolerance matter on real data? (CPU, reads every chain)

`match(a, b)` treats answers within 0.5% as equal, so "200" == "201" and
"1000" == "1004". Two consequences to measure on a generation run:

  1. inside pools: distinct integer answers that match() merges (they vote
     together and the veto treats them as one)
  2. against gold: chains / committed answers that count as CORRECT only
     because of the tolerance (precision inflation)

Also counts chains lost because the '####' answer line carried units
("#### 42 dollars"), which the parser turns into no answer.

    python tools/match_tolerance_audit.py --root <generation run> [--out audit.json]
"""
import argparse
import glob
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

from icm.method import data, vote

UNITS_RE = re.compile(r"####\s*\S*\d")


def exact(a, b):
    return Decimal(a) == Decimal(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tol_pairs = Counter(); tol_gold_chains = Counter(); units_lost = Counter(); n_chains = Counter()
    per_lang_pools = Counter()
    for path in sorted(glob.glob(f"{a.root}/final/pooled/generated/*.jsonl")):
        lang = Path(path).stem
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            gold = data.norm(row.get("gold_number"))
            answers = set()
            for c in row.get("candidates", []):
                n_chains[lang] += 1
                ans = data.norm(c.get("pred_number"))
                text = c.get("text") or ""
                if ans is None and UNITS_RE.search(text):
                    units_lost[lang] += 1
                if ans is None:
                    continue
                answers.add(ans)
                if gold is not None and data.match(ans, gold) and not exact(ans, gold):
                    tol_gold_chains[lang] += 1
            per_lang_pools[lang] += 1
            ans = sorted(answers)
            for i in range(len(ans)):
                for j in range(i + 1, len(ans)):
                    if data.match(ans[i], ans[j]):
                        tol_pairs[lang] += 1
                        break

    # committed answers correct only by tolerance, after the full vote
    languages = {**data.LANGUAGES_50, **{l: "new" for l in per_lang_pools}}
    pools = data.load(a.root, languages=languages)
    state, contested, _ = vote.converge(pools)
    tol_commits = Counter(); commits = Counter(); exact_wrong = Counter()
    for iid, s in state.items():
        if s["answer"] is None:
            continue
        lang = pools[iid]["lang"]; commits[lang] += 1
        if data.match(s["answer"], pools[iid]["gold"]) and not exact(s["answer"], pools[iid]["gold"]):
            tol_commits[lang] += 1
    tot_commits = sum(commits.values())
    print(f"{'lang':<7}{'pools':>7}{'pools w/ merged answers':>25}{'chains right by tol':>21}"
          f"{'commits right by tol':>22}{'chains lost to units':>22}")
    rows = {}
    for lang in sorted(per_lang_pools):
        rows[lang] = {"pools": per_lang_pools[lang], "pools_with_tolerance_merge": tol_pairs[lang],
                      "chains_correct_only_by_tolerance": tol_gold_chains[lang],
                      "commits_correct_only_by_tolerance": tol_commits[lang], "commits": commits[lang],
                      "chains_lost_to_units": units_lost[lang], "chains": n_chains[lang]}
        r = rows[lang]
        print(f"{lang:<7}{r['pools']:>7}{r['pools_with_tolerance_merge']:>25}"
              f"{r['chains_correct_only_by_tolerance']:>21}{r['commits_correct_only_by_tolerance']:>22}"
              f"{r['chains_lost_to_units']:>22}")
    tc = sum(tol_commits.values())
    print(f"\nTOTAL commits {tot_commits}; correct only by tolerance {tc} "
          f"= {100 * tc / max(tot_commits, 1):.3f} points of precision; "
          f"pools with a tolerance merge {sum(tol_pairs.values())}; "
          f"chains lost to units {sum(units_lost.values())} of {sum(n_chains.values())}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
