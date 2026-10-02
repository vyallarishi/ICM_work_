"""H7 step 3 (CPU): the pipeline audits itself on problems with derived answers.

The variant pools (h7_build_variants + generation) are labelled by the SAME
pipeline (v2 rules, voting only) with no gold. Their gold answers were derived
by re-executing the source solution's calculations on new numbers, so label
precision can be measured without any human label. It is compared with label
precision on the ORIGINAL problems the variants came from (same problems, same
languages, same rules), from the 37 x 1,500 record.

PRE-REGISTERED:
  (a) overall precision on variants is within 3 points of precision on their
      source originals (the pipeline's quality transfers to new problems), and
  (b) committed answers equal to the SOURCE problem's answer (memorisation of
      the original instead of solving the variant) are <= 2% of commits.
If (a) holds, derived-answer variants are a label-free audit of the pipeline
on any new data, the scaling claim of the paper.

    python -m icm.experiments.q10_pipeline_hypotheses.h7_audit
"""
import json

from ...method import data, vote
from ...method.rules import V2
from ...paths import RUN_DIR
from .common import ROOT, by_tier, exact, write

VARIANT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-variants500"


def main():
    index = json.loads((RUN_DIR / "h7_variants_index.json").read_text())
    var = data.load(VARIANT_ROOT, V2)
    st, _, _ = vote.converge(var, rules=V2)
    var_ans = {i: s["answer"] for i, s in st.items()}

    # the same source problems and languages in the record
    orig_all = data.load(ROOT, V2)
    so, _, _ = vote.converge(orig_all, rules=V2)
    pairs = {}
    for iid in var:
        lang, vi = var[iid]["lang"], iid.rsplit("_", 1)[-1]
        oid = f"train_{lang}_{index[vi]['source_original_id']}"
        if oid in orig_all:
            pairs[iid] = oid
    orig = {pairs[i]: orig_all[pairs[i]] for i in pairs}
    res = {"variants": by_tier({i: var[i] for i in pairs}, {i: var_ans[i] for i in pairs}),
           "originals_same_problems": by_tier(orig, {o: so[o]["answer"] for o in orig}),
           "variants_all": by_tier(var, var_ans)}
    commits = [i for i in pairs if var_ans[i] is not None]
    memo = sum(exact(var_ans[i], index[i.rsplit("_", 1)[-1]]["source_answer"]) and
               not exact(var_ans[i], var[i]["gold"]) for i in commits)
    chains = [c for i in pairs for c in var[i]["chains"]]
    memo_chains = sum(exact(c["answer"], index[i.rsplit("_", 1)[-1]]["source_answer"])
                      for i in pairs for c in var[i]["chains"])
    res["memorisation"] = {"commits_equal_source_answer": memo, "commits": len(commits),
                           "rate": round(memo / max(len(commits), 1), 4),
                           "chains_with_source_answer_rate": round(memo_chains / max(len(chains), 1), 4)}
    v, o = res["variants"]["ALL"], res["originals_same_problems"]["ALL"]
    gap = 100 * (v["precision"] - o["precision"])
    res["verdict"] = {"precision_gap_points": round(gap, 2), "H7a_holds": abs(gap) <= 3.0,
                      "H7b_holds": res["memorisation"]["rate"] <= 0.02}
    print(f"{'':<26}{'n':>7}{'coverage':>10}{'precision':>11}")
    for name in ("variants", "originals_same_problems"):
        for t in ("ALL", "high", "mid", "low"):
            r = res[name][t]
            print(f"{name + ' ' + t:<26}{r['n']:>7}{100 * r['coverage']:>10.2f}{100 * r['precision']:>11.2f}")
    print(f"\nmemorisation: {memo}/{len(commits)} commits equal the source answer ({100 * res['memorisation']['rate']:.2f}%); "
          f"{100 * res['memorisation']['chains_with_source_answer_rate']:.2f}% of chains")
    print(f"H7a precision gap {gap:+.2f} points -> {'HOLDS' if res['verdict']['H7a_holds'] else 'FAILS'} (bar 3);  "
          f"H7b memorisation -> {'HOLDS' if res['verdict']['H7b_holds'] else 'FAILS'} (bar 2%)")
    write("h7_audit.json", res)


if __name__ == "__main__":
    main()
