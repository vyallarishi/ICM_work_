"""Run the whole method.

    python -m icm.method.run gen    --root <pools> --model-dir <hf model> --out <regen.jsonl>   (GPU)
    python -m icm.method.run decide --root <pools> --regen <regen.jsonl> --out <results.json>  (CPU)

With no arguments both use the headline run's paths (the /scratch pools and
data/ICM_work_/runs/stage25_regen.jsonl), so `python -m icm.method.run decide` reproduces
recorded_results/stage25_results.json and writes data/ICM_work_/runs/method_results.json.
"""
import argparse
import json

from ..paths import RUN_DIR
from .data import LANGUAGES_50, TIER, load, match
from .metrics import metrics
from .regenerate import ALPHA, build_jobs, generate, load_regen, weighted_decide
from .rules import BY_NAME, RECORD
from .vote import World, converge

DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
DEFAULT_MODEL = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_REGEN = str(RUN_DIR / "stage25_regen.jsonl")


def decide(pools, regen, world=None, rules=RECORD):
    """Vote, then re-decide the contested pools with the regenerated answers.

    Returns {"baseline": metrics after voting, "with_regen": metrics after
    regeneration plus a recovery breakdown}. Same structure as the recorded
    headline JSON.
    """
    world = world or World(pools)
    state, contested, oscillators = converge(pools, world, rules)
    base = metrics(pools, state, contested, rules)

    new_state = {i: dict(s) for i, s in state.items()}
    for iid in contested:
        others = world.context(new_state, iid)
        new_state[iid]["answer"] = weighted_decide(
            pools[iid], others, regen.get(iid, []), ALPHA,
            iid in oscillators, rules)
    row = metrics(pools, new_state, contested, rules)

    abst = [i for i in contested if state[i]["answer"] is None]
    recovered = [i for i in abst if new_state[i]["answer"] is not None]
    rec_ok = sum(match(new_state[i]["answer"], pools[i]["gold"], rules)
                 for i in recovered)
    wrong_commits = withdrawn = 0
    for iid in contested:
        if state[iid]["answer"] is None:
            continue
        was_ok = match(state[iid]["answer"], pools[iid]["gold"], rules)
        now = new_state[iid]["answer"]
        if was_ok and now is None:
            withdrawn += 1
        elif was_ok and not match(now, pools[iid]["gold"], rules):
            wrong_commits += 1
    tier_rows = {}
    for tier in ("high", "mid", "low"):
        t_abst = [i for i in abst if LANGUAGES_50.get(pools[i]["lang"]) == tier]
        t_rec = [i for i in t_abst if new_state[i]["answer"] is not None]
        t_ok = sum(match(new_state[i]["answer"], pools[i]["gold"], rules)
                   for i in t_rec)
        tier_rows[tier] = {"abstained": len(t_abst), "recovered": len(t_rec),
                           "correct": t_ok}
    row.update({"abstained_before": len(abst), "recovered": len(recovered),
                "recovered_correct": rec_ok,
                "recovery_precision": round(rec_ok / max(len(recovered), 1), 4),
                "wrong_commits_added": wrong_commits + len(recovered) - rec_ok,
                "correct_withdrawn": withdrawn,
                "recovery_by_tier": tier_rows})
    return {"baseline": base, "with_regen": row}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", choices=("record", "all"), default="record",
                        help="record = the 37 languages of the headline; all = plus the 13 added in 2026-10")
    parser.add_argument("--rules", choices=sorted(BY_NAME), default="record",
                        help="record = the headline's rules; v2 = the audited fixes (icm/method/rules.py)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="GPU: regenerate chains for the contested pools")
    g.add_argument("--root", default=DEFAULT_ROOT)
    g.add_argument("--model-dir", default=DEFAULT_MODEL)
    g.add_argument("--out", default=DEFAULT_REGEN)
    g.add_argument("--dry-run", action="store_true", help="build prompts, print one, stop")
    d = sub.add_parser("decide", help="CPU: vote + regeneration decision -> results JSON")
    d.add_argument("--root", default=DEFAULT_ROOT)
    d.add_argument("--regen", default=DEFAULT_REGEN)
    d.add_argument("--out", default=str(RUN_DIR / "method_results.json"))
    args = parser.parse_args()

    rules = BY_NAME[args.rules]
    languages = LANGUAGES_50 if args.languages == "all" else TIER
    pools = load(args.root, rules, languages)
    if args.cmd == "gen":
        state, contested, oscillators = converge(pools, rules=rules)
        print(f"after voting ({rules.name} rules): {metrics(pools, state, contested, rules)}")
        jobs = build_jobs(pools, state, contested, rules)
        abstained = [i for i in contested if state[i]["answer"] is None]
        print(f"pools {len(pools)}; contested {len(contested)} "
              f"(abstained {len(abstained)}, oscillators {len(oscillators)}); "
              f"regen jobs {len(jobs)}")
        if args.dry_run:
            print(f"\n--- sample prompt (first 300) ---\n{jobs[0][1][:300]}")
            return
        generate(jobs, args.model_dir, args.out, rules)
        print(f"wrote {args.out} ({len(jobs)} pools)")
    else:
        results = decide(pools, load_regen(args.regen), rules=rules)
        with open(args.out, "w") as handle:
            json.dump(results, handle, indent=2)
        print(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
