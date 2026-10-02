"""H1: the audited rule fixes (v2) do not cost label quality.

PRE-REGISTERED. On 37 x 1,500, voting only (no regeneration, so the rule sets
are compared on identical information), scored exactly against gold:
    v2 precision >= record precision - 0.2 points  AND
    v2 coverage  >= record coverage  - 0.5 points,  overall.
If it holds, the 50-language run uses v2. Also reported: each switch alone,
and the record WITH regeneration (the headline), for reference.

Why voting only: regeneration answers were generated for the record's
contested set; v2's fixed view changes that set, so v2 + regeneration would
be missing answers for some pools and look worse for a reason unrelated to
the rules.

    python -m icm.experiments.q10_pipeline_hypotheses.h1_rules
"""
from ...method import data, vote
from ...method.regenerate import load_regen
from ...method.rules import BY_NAME
from ...paths import RUN_DIR
from ..q09_does_the_dataset_beat_translation.build_sft_sets import committed_answers
from .common import ROOT, by_tier, write


def main():
    out = {}
    for name in ("record", "only_view", "only_veto", "only_exact", "v2"):
        rules = BY_NAME[name]
        pools = data.load(ROOT, rules)
        state, _, _ = vote.converge(pools, rules=rules)
        out[name] = by_tier(pools, {i: s["answer"] for i, s in state.items()})
        a = out[name]["ALL"]
        print(f"{name:<12} voting only   coverage {100 * a['coverage']:.2f}  precision {100 * a['precision']:.2f}", flush=True)
    pools = data.load(ROOT)
    final = committed_answers({k: dict(v) for k, v in pools.items()}, load_regen(RUN_DIR / "stage25_regen.jsonl"))
    out["record_with_regeneration"] = by_tier(pools, final)
    a = out["record_with_regeneration"]["ALL"]
    print(f"{'record+regen':<12}               coverage {100 * a['coverage']:.2f}  precision {100 * a['precision']:.2f}")
    r, v = out["record"]["ALL"], out["v2"]["ALL"]
    dp, dc = 100 * (v["precision"] - r["precision"]), 100 * (v["coverage"] - r["coverage"])
    out["verdict"] = {"precision_change": round(dp, 2), "coverage_change": round(dc, 2),
                      "H1_holds": dp >= -0.2 and dc >= -0.5}
    print(f"\nH1: v2 - record = precision {dp:+.2f}, coverage {dc:+.2f}  ->  "
          f"{'HOLDS' if out['verdict']['H1_holds'] else 'FAILS'} (bar: >= -0.2 / >= -0.5)")
    write("h1_rules.json", out)


if __name__ == "__main__":
    main()
